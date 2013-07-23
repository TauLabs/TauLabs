/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup TelemetryModule Telemetry Module
 * @{ 
 *
 * @file       telemetry.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Telemetry module, handles telemetry and UAVObject updates
 * @see        The GNU Public License (GPL) Version 3
 *
 *****************************************************************************/
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include "openpilot.h"
#include "flighttelemetrystats.h"
#include "gcstelemetrystats.h"
#include "modulesettings.h"

// Private constants
#define MAX_QUEUE_SIZE   TELEM_QUEUE_SIZE
#define STACK_SIZE_BYTES PIOS_TELEM_STACK_SIZE
#define TASK_PRIORITY_RX (tskIDLE_PRIORITY + 2)
#define TASK_PRIORITY_TX (tskIDLE_PRIORITY + 2)
#define TASK_PRIORITY_TXPRI (tskIDLE_PRIORITY + 2)
#define REQ_TIMEOUT_MS 250
#define MAX_RETRIES 2
#define STATS_UPDATE_PERIOD_MS 4000
#define CONNECTION_TIMEOUT_MS 8000

// Private types

// Private variables
static uintptr_t telemetryPort;
static xQueueHandle queue;

#if defined(PIOS_TELEM_PRIORITY_QUEUE)
static xQueueHandle priorityQueue;
static xTaskHandle telemetryTxPriTaskHandle;
static void telemetryTxPriTask(void *parameters);
#else
#define priorityQueue queue
#endif

static xTaskHandle telemetryTxTaskHandle;
static xTaskHandle telemetryRxTaskHandle;
static uint32_t txErrors;
static uint32_t txRetries;
static uint32_t timeOfLastObjectUpdate;
static UAVTalkConnection uavTalkCon;

// Private functions
static void telemetryTxTask(void *parameters);
static void telemetryRxTask(void *parameters);
static int32_t transmitData(uint8_t * data, int32_t length);
static void registerObject(UAVObjHandle obj);
static void updateObject(UAVObjHandle obj, int32_t eventType);
static int32_t setUpdatePeriod(UAVObjHandle obj, int32_t updatePeriodMs);
static void processObjEvent(UAVObjEvent * ev);
static void updateTelemetryStats();
static void gcsTelemetryStatsUpdated();
static void updateSettings();
static uintptr_t getComPort();

/**
 * Initialise the telemetry module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t TelemetryStart(void)
{
	// Process all registered objects and connect queue for updates
	UAVObjIterate(&registerObject);
    
	// Listen to objects of interest
	GCSTelemetryStatsConnectQueue(priorityQueue);
    
	// Start telemetry tasks
	xTaskCreate(telemetryTxTask, (signed char *)"TelTx", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY_TX, &telemetryTxTaskHandle);
	xTaskCreate(telemetryRxTask, (signed char *)"TelRx", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY_RX, &telemetryRxTaskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_TELEMETRYTX, telemetryTxTaskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_TELEMETRYRX, telemetryRxTaskHandle);

#if defined(PIOS_TELEM_PRIORITY_QUEUE)
	xTaskCreate(telemetryTxPriTask, (signed char *)"TelPriTx", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY_TXPRI, &telemetryTxPriTaskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_TELEMETRYTXPRI, telemetryTxPriTaskHandle);
#endif

	return 0;
}

/**
 * Initialise the telemetry module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t TelemetryInitialize(void)
{
	FlightTelemetryStatsInitialize();
	GCSTelemetryStatsInitialize();

	// Initialize vars
	timeOfLastObjectUpdate = 0;

	// Create object queues
	queue = xQueueCreate(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));
#if defined(PIOS_TELEM_PRIORITY_QUEUE)
	priorityQueue = xQueueCreate(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));
#endif

	// Update telemetry settings
	telemetryPort = PIOS_COM_TELEM_RF;
	updateSettings();
    
	// Initialise UAVTalk
	uavTalkCon = UAVTalkInitialize(&transmitData);
    
	// Create periodic event that will be used to update the telemetry stats
	txErrors = 0;
	txRetries = 0;
	UAVObjEvent ev;
	memset(&ev, 0, sizeof(UAVObjEvent));
	EventPeriodicQueueCreate(&ev, priorityQueue, STATS_UPDATE_PERIOD_MS);

	return 0;
}

MODULE_INITCALL(TelemetryInitialize, TelemetryStart)

/**
 * Register a new object, adds object to local list and connects the queue depending on the object's
 * telemetry settings.
 * \param[in] obj Object to connect
 */
static void registerObject(UAVObjHandle obj)
{
	if (UAVObjIsMetaobject(obj)) {
		/* Only connect change notifications for meta objects.  No periodic updates */
		UAVObjConnectQueue(obj, priorityQueue, EV_MASK_ALL_UPDATES);
		return;
	} else {
		UAVObjMetadata metadata;
		UAVObjUpdateMode updateMode;
		UAVObjGetMetadata(obj, &metadata);
		updateMode = UAVObjGetTelemetryUpdateMode(&metadata);

		/* Only create a periodic event for objects that are periodic */
		if ((updateMode == UPDATEMODE_PERIODIC) ||
			(updateMode == UPDATEMODE_THROTTLED)) {
			// Setup object for periodic updates
			UAVObjEvent ev = {
				.obj    = obj,
				.instId = UAVOBJ_ALL_INSTANCES,
				.event  = EV_UPDATED_PERIODIC,
			};
			EventPeriodicQueueCreate(&ev, queue, 0);
		}

		// Setup object for telemetry updates
		updateObject(obj, EV_NONE);
	}
}

/**
 * Update object's queue connections and timer, depending on object's settings
 * \param[in] obj Object to updates
 */
static void updateObject(UAVObjHandle obj, int32_t eventType)
{
	UAVObjMetadata metadata;
	UAVObjUpdateMode updateMode;
	int32_t eventMask;

	if (UAVObjIsMetaobject(obj)) {
		/* This function updates the periodic updates for the object.
		 * Meta Objects cannot have periodic updates.
		 */
		PIOS_Assert(false);
		return;
	}

	// Get metadata
	UAVObjGetMetadata(obj, &metadata);
	updateMode = UAVObjGetTelemetryUpdateMode(&metadata);

	// Setup object depending on update mode
	switch (updateMode) {
	case UPDATEMODE_PERIODIC:
		// Set update period
		setUpdatePeriod(obj, metadata.telemetryUpdatePeriod);
		// Connect queue
		eventMask = EV_UPDATED_PERIODIC | EV_UPDATED_MANUAL | EV_UPDATE_REQ;
		UAVObjConnectQueue(obj, priorityQueue, eventMask);
		break;
	case UPDATEMODE_ONCHANGE:
		// Set update period
		setUpdatePeriod(obj, 0);
		// Connect queue
		eventMask = EV_UPDATED | EV_UPDATED_MANUAL | EV_UPDATE_REQ;
		UAVObjConnectQueue(obj, priorityQueue, eventMask);
		break;
	case UPDATEMODE_THROTTLED:
		if ((eventType == EV_UPDATED_PERIODIC) || (eventType == EV_NONE)) {
			// If we received a periodic update, we can change back to update on change
			eventMask = EV_UPDATED | EV_UPDATED_MANUAL | EV_UPDATE_REQ;
			// Set update period on initialization and metadata change
			if (eventType == EV_NONE)
				setUpdatePeriod(obj, metadata.telemetryUpdatePeriod);
		} else {
			eventMask = getEventMask(obj, priorityQueue);
			if (eventMask & EV_UPDATED_PERIODIC) {
				// If periodic flag is already set then we have previously sent an update during
				// the timeout period and this update would be missed, so set the dirty flag.
				// Once setting EV_UPDATED_THROTTLED_DIRTY, there is no need to listen for EV_UPDATED.
				eventMask = EV_UPDATED_PERIODIC | EV_UPDATE_REQ | EV_UPDATED_THROTTLED_DIRTY;
			} else { //If periodic is not set then we just received an object
				// update so switch to periodic for the timeout period to prevent
				// sending more updates.  Listen to the EV_UPDATED flag still to
				// catch any updates that would overwise never get sent.
				eventMask = EV_UPDATED_PERIODIC | EV_UPDATED | EV_UPDATED_MANUAL | EV_UPDATE_REQ;
			}
		}
		UAVObjConnectQueue(obj, priorityQueue, eventMask);
		break;
	case UPDATEMODE_MANUAL:
		// Set update period
		setUpdatePeriod(obj, 0);
		// Connect queue
		eventMask = EV_UPDATED_MANUAL | EV_UPDATE_REQ;
		UAVObjConnectQueue(obj, priorityQueue, eventMask);
		break;
	}
}

/**
 * Processes queue events
 */
static void processObjEvent(UAVObjEvent * ev)
{
	UAVObjMetadata metadata;
	UAVObjUpdateMode updateMode;
	FlightTelemetryStatsData flightStats;
	int32_t retries;
	int32_t success;

	if (ev->obj == 0) {
		updateTelemetryStats();
	} else if (ev->obj == GCSTelemetryStatsHandle()) {
		gcsTelemetryStatsUpdated();
	} else {
		FlightTelemetryStatsGet(&flightStats);
		// Get object metadata
		UAVObjGetMetadata(ev->obj, &metadata);
		updateMode = UAVObjGetTelemetryUpdateMode(&metadata);

		// Act on event
		retries = 0;
		success = -1;
		if (ev->event == EV_UPDATED || ev->event == EV_UPDATED_MANUAL || ((ev->event == EV_UPDATED_PERIODIC) && (updateMode != UPDATEMODE_THROTTLED))) {
			// Send update to GCS (with retries)
			while (retries < MAX_RETRIES && success == -1) {
				success = UAVTalkSendObject(uavTalkCon, ev->obj, ev->instId, UAVObjGetTelemetryAcked(&metadata), REQ_TIMEOUT_MS);	// call blocks until ack is received or timeout
				++retries;
			}
			// Update stats
			txRetries += (retries - 1);
			if (success == -1) {
				++txErrors;
			}
		} else if (ev->event == EV_UPDATE_REQ) {
			// Request object update from GCS (with retries)
			while (retries < MAX_RETRIES && success == -1) {
				success = UAVTalkSendObjectRequest(uavTalkCon, ev->obj, ev->instId, REQ_TIMEOUT_MS);	// call blocks until update is received or timeout
				++retries;
			}
			// Update stats
			txRetries += (retries - 1);
			if (success == -1) {
				++txErrors;
			}
		} else if (ev->event == EV_UPDATED_PERIODIC && updateMode == UPDATEMODE_THROTTLED) {
			// Get the event mask
			int32_t eventMask = getEventMask(ev->obj, priorityQueue);

			if (eventMask & EV_UPDATED_THROTTLED_DIRTY) { // If EV_UPDATED_THROTTLED_DIRTY flag is set then send the data like normal.
				// Send update to GCS (with retries)
				while (retries < MAX_RETRIES && success == -1) {
					success = UAVTalkSendObject(uavTalkCon, ev->obj, ev->instId, UAVObjGetTelemetryAcked(&metadata), REQ_TIMEOUT_MS);	// call blocks until ack is received or timeout
					++retries;
				}
				// Update stats
				txRetries += (retries - 1);
				if (success == -1) {
					++txErrors;
				}
			}
		}
		// If this is a metaobject then make necessary telemetry updates
		if (UAVObjIsMetaobject(ev->obj)) {
			updateObject(UAVObjGetLinkedObj(ev->obj), EV_NONE);	// linked object will be the actual object the metadata are for
		} else {
			if (updateMode == UPDATEMODE_THROTTLED) {
				// If this is UPDATEMODE_THROTTLED, the event mask changes on every event.
				updateObject(ev->obj, ev->event);
			}
		}
	}
}

/**
 * Telemetry transmit task, regular priority
 */
static void telemetryTxTask(void *parameters)
{
	UAVObjEvent ev;

	// Loop forever
	while (1) {
		// Wait for queue message
		if (xQueueReceive(queue, &ev, portMAX_DELAY) == pdTRUE) {
			// Process event
			processObjEvent(&ev);
		}
	}
}

/**
 * Telemetry transmit task, high priority
 */
#if defined(PIOS_TELEM_PRIORITY_QUEUE)
static void telemetryTxPriTask(void *parameters)
{
	UAVObjEvent ev;

	// Loop forever
	while (1) {
		// Wait for queue message
		if (xQueueReceive(priorityQueue, &ev, portMAX_DELAY) == pdTRUE) {
			// Process event
			processObjEvent(&ev);
		}
	}
}
#endif

/**
 * Telemetry transmit task. Processes queue events and periodic updates.
 */
static void telemetryRxTask(void *parameters)
{

	// Task loop
	while (1) {
		uintptr_t inputPort = getComPort();

		if (inputPort) {
			// Block until data are available
			uint8_t serial_data[1];
			uint16_t bytes_to_process;

			bytes_to_process = PIOS_COM_ReceiveBuffer(inputPort, serial_data, sizeof(serial_data), 500);
			if (bytes_to_process > 0) {
				for (uint8_t i = 0; i < bytes_to_process; i++) {
					UAVTalkProcessInputStream(uavTalkCon,serial_data[i]);
				}
			}
		} else {
			vTaskDelay(5);
		}
	}
}

/**
 * Transmit data buffer to the modem or USB port.
 * \param[in] data Data buffer to send
 * \param[in] length Length of buffer
 * \return -1 on failure
 * \return number of bytes transmitted on success
 */
static int32_t transmitData(uint8_t * data, int32_t length)
{
	uintptr_t outputPort = getComPort();

	if (outputPort)
		return PIOS_COM_SendBuffer(outputPort, data, length);

	return -1;
}

/**
 * Set update period of object (it must be already setup for periodic updates)
 * \param[in] obj The object to update
 * \param[in] updatePeriodMs The update period in ms, if zero then periodic updates are disabled
 * \return 0 Success
 * \return -1 Failure
 */
static int32_t setUpdatePeriod(UAVObjHandle obj, int32_t updatePeriodMs)
{
	UAVObjEvent ev;

	// Add object for periodic updates
	ev.obj = obj;
	ev.instId = UAVOBJ_ALL_INSTANCES;
	ev.event = EV_UPDATED_PERIODIC;
	return EventPeriodicQueueUpdate(&ev, queue, updatePeriodMs);
}

/**
 * Called each time the GCS telemetry stats object is updated.
 * Trigger a flight telemetry stats update if a connection is not
 * yet established.
 */
static void gcsTelemetryStatsUpdated()
{
	FlightTelemetryStatsData flightStats;
	GCSTelemetryStatsData gcsStats;
	FlightTelemetryStatsGet(&flightStats);
	GCSTelemetryStatsGet(&gcsStats);
	if (flightStats.Status != FLIGHTTELEMETRYSTATS_STATUS_CONNECTED || gcsStats.Status != GCSTELEMETRYSTATS_STATUS_CONNECTED) {
		updateTelemetryStats();
	}
}

/**
 * Update telemetry statistics and handle connection handshake
 */
static void updateTelemetryStats()
{
	UAVTalkStats utalkStats;
	FlightTelemetryStatsData flightStats;
	GCSTelemetryStatsData gcsStats;
	uint8_t forceUpdate;
	uint8_t connectionTimeout;
	uint32_t timeNow;

	// Get stats
	UAVTalkGetStats(uavTalkCon, &utalkStats);
	UAVTalkResetStats(uavTalkCon);

	// Get object data
	FlightTelemetryStatsGet(&flightStats);
	GCSTelemetryStatsGet(&gcsStats);

	// Update stats object
	if (flightStats.Status == FLIGHTTELEMETRYSTATS_STATUS_CONNECTED) {
		flightStats.RxDataRate = (float)utalkStats.rxBytes / ((float)STATS_UPDATE_PERIOD_MS / 1000.0f);
		flightStats.TxDataRate = (float)utalkStats.txBytes / ((float)STATS_UPDATE_PERIOD_MS / 1000.0f);
		flightStats.RxFailures += utalkStats.rxErrors;
		flightStats.TxFailures += txErrors;
		flightStats.TxRetries += txRetries;
		txErrors = 0;
		txRetries = 0;
	} else {
		flightStats.RxDataRate = 0;
		flightStats.TxDataRate = 0;
		flightStats.RxFailures = 0;
		flightStats.TxFailures = 0;
		flightStats.TxRetries = 0;
		txErrors = 0;
		txRetries = 0;
	}

	// Check for connection timeout
	timeNow = TICKS2MS(xTaskGetTickCount());
	if (utalkStats.rxObjects > 0) {
		timeOfLastObjectUpdate = timeNow;
	}
	if ((timeNow - timeOfLastObjectUpdate) > CONNECTION_TIMEOUT_MS) {
		connectionTimeout = 1;
	} else {
		connectionTimeout = 0;
	}

	// Update connection state
	forceUpdate = 1;
	if (flightStats.Status == FLIGHTTELEMETRYSTATS_STATUS_DISCONNECTED) {
		// Wait for connection request
		if (gcsStats.Status == GCSTELEMETRYSTATS_STATUS_HANDSHAKEREQ) {
			flightStats.Status = FLIGHTTELEMETRYSTATS_STATUS_HANDSHAKEACK;
		}
	} else if (flightStats.Status == FLIGHTTELEMETRYSTATS_STATUS_HANDSHAKEACK) {
		// Wait for connection
		if (gcsStats.Status == GCSTELEMETRYSTATS_STATUS_CONNECTED) {
			flightStats.Status = FLIGHTTELEMETRYSTATS_STATUS_CONNECTED;
		} else if (gcsStats.Status == GCSTELEMETRYSTATS_STATUS_DISCONNECTED) {
			flightStats.Status = FLIGHTTELEMETRYSTATS_STATUS_DISCONNECTED;
		}
	} else if (flightStats.Status == FLIGHTTELEMETRYSTATS_STATUS_CONNECTED) {
		if (gcsStats.Status != GCSTELEMETRYSTATS_STATUS_CONNECTED || connectionTimeout) {
			flightStats.Status = FLIGHTTELEMETRYSTATS_STATUS_DISCONNECTED;
		} else {
			forceUpdate = 0;
		}
	} else {
		flightStats.Status = FLIGHTTELEMETRYSTATS_STATUS_DISCONNECTED;
	}

	// Update the telemetry alarm
	if (flightStats.Status == FLIGHTTELEMETRYSTATS_STATUS_CONNECTED) {
		AlarmsClear(SYSTEMALARMS_ALARM_TELEMETRY);
	} else {
		AlarmsSet(SYSTEMALARMS_ALARM_TELEMETRY, SYSTEMALARMS_ALARM_ERROR);
	}

	// Update object
	FlightTelemetryStatsSet(&flightStats);

	// Force telemetry update if not connected
	if (forceUpdate) {
		FlightTelemetryStatsUpdated();
	}
}

/**
 * Update the telemetry settings, called on startup.
 * FIXME: This should be in the TelemetrySettings object. But objects
 * have too much overhead yet. Also the telemetry has no any specific
 * settings, etc. Thus the ModuleSettings object which contains the
 * telemetry port speed is used for now.
 */
static void updateSettings()
{
	
	if (telemetryPort) {
		// Retrieve settings
		uint8_t speed;
		ModuleSettingsTelemetrySpeedGet(&speed);

		// Set port speed
		switch (speed) {
		case MODULESETTINGS_TELEMETRYSPEED_2400:
			PIOS_COM_ChangeBaud(telemetryPort, 2400);
			break;
		case MODULESETTINGS_TELEMETRYSPEED_4800:
			PIOS_COM_ChangeBaud(telemetryPort, 4800);
			break;
		case MODULESETTINGS_TELEMETRYSPEED_9600:
			PIOS_COM_ChangeBaud(telemetryPort, 9600);
			break;
		case MODULESETTINGS_TELEMETRYSPEED_19200:
			PIOS_COM_ChangeBaud(telemetryPort, 19200);
			break;
		case MODULESETTINGS_TELEMETRYSPEED_38400:
			PIOS_COM_ChangeBaud(telemetryPort, 38400);
			break;
		case MODULESETTINGS_TELEMETRYSPEED_57600:
			PIOS_COM_ChangeBaud(telemetryPort, 57600);
			break;
		case MODULESETTINGS_TELEMETRYSPEED_115200:
			PIOS_COM_ChangeBaud(telemetryPort, 115200);
			break;
		}
	}
}

/**
 * Determine input/output com port as highest priority available 
 */
static uintptr_t getComPort() {
#if defined(PIOS_INCLUDE_USB)
	if ( PIOS_COM_Available(PIOS_COM_TELEM_USB) )
		return PIOS_COM_TELEM_USB;
	else
#endif /* PIOS_INCLUDE_USB */
		if ( PIOS_COM_Available(telemetryPort) )
			return telemetryPort;
		else
			return 0;
}

/**
  * @}
  * @}
  */
