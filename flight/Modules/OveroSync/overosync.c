/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup OveroSyncModule OveroSync Module
 * @{ 
 *
 * @file       overosync.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      Communication with an Overo via SPI
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
#include "modulesettings.h"
#include "overosync.h"
#include "overosyncstats.h"
#include "systemstats.h"
#include "pios_thread.h"
#include "pios_queue.h"

// Private constants
#define MAX_QUEUE_SIZE   200
#define STACK_SIZE_BYTES 512
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW

// Private types

// Private variables
static struct pios_queue *queue;
static UAVTalkConnection uavTalkCon;
static struct pios_thread *overoSyncTaskHandle;
static bool module_enabled;

// Private functions
static void    overoSyncTask(void *parameters);
static int32_t pack_data(uint8_t * data, int32_t length);
static void    register_object(UAVObjHandle obj);
static void    send_settings(UAVObjHandle obj);

// External variables
extern uint32_t pios_com_overo_id;
extern uint32_t pios_overo_id;

struct overosync {
	uint32_t sent_bytes;
	uint32_t sent_objects;
	uint32_t failed_objects;
	uint32_t received_objects;
	bool     sending_settings;
};

struct overosync *overosync;

/**
 * Initialise the overo sync module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t OveroSyncInitialize(void)
{
#ifdef MODULE_OveroSync_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_OVEROSYNC] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (!module_enabled)
		return -1;

	// Create object queues
	queue = PIOS_Queue_Create(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));
	
	OveroSyncStatsInitialize();

	// Initialise UAVTalk
	uavTalkCon = UAVTalkInitialize(&pack_data);

	return 0;
}

/**
 * Start the overo sync module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t OveroSyncStart(void)
{
	//Check if module is enabled or not
	if (module_enabled == false) {
		return -1;
	}
	
	overosync = (struct overosync *) PIOS_malloc(sizeof(*overosync));
	if(overosync == NULL)
		return -1;

	overosync->sent_bytes = 0;

	// Process all registered objects and connect queue for updates
	UAVObjIterate(&register_object);
	
	// Start telemetry tasks
	overoSyncTaskHandle = PIOS_Thread_Create(overoSyncTask, "OveroSync", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
	
	TaskMonitorAdd(TASKINFO_RUNNING_OVEROSYNC, overoSyncTaskHandle);
	
	return 0;
}

MODULE_INITCALL(OveroSyncInitialize, OveroSyncStart)
;
/**
 * Register a new object, adds object to local list and connects the queue depending on the object's
 * telemetry settings.
 * \param[in] obj Object to connect
 */
static void register_object(UAVObjHandle obj)
{
	int32_t eventMask;
	eventMask = EV_UPDATED | EV_UPDATED_MANUAL | EV_UPDATE_REQ | EV_UNPACKED;
	UAVObjConnectQueue(obj, queue, eventMask);
}

/**
 * Register a new object, adds object to local list and connects the queue depending on the object's
 * telemetry settings.
 * \param[in] obj Object to connect
 */
static void send_settings(UAVObjHandle obj)
{
	if (UAVObjIsSettings(obj)) {
		UAVTalkSendObjectTimestamped(uavTalkCon, obj, 0, false, 0);
	}
}

/**
 * Telemetry transmit task, regular priority
 *
 * Logic: We need to double buffer the DMA transfers.  Pack the buffer until either
 * 1) it is full (and then we should record the number of missed events then)
 * 2) the current transaction is done (we should immediately schedule since we are slave)
 * when done packing the buffer we should call PIOS_SPI_TransferBlock, change the active buffer
 * and then take the semaphrore
 */
static void overoSyncTask(void *parameters)
{
	UAVObjEvent ev;

	// Kick off SPI transfers (once one is completed another will automatically transmit)
	overosync->sent_objects = 0;
	overosync->failed_objects = 0;
	overosync->received_objects = 0;
	
	uint32_t lastUpdateTime = PIOS_Thread_Systime();
	uint32_t updateTime;

	bool initialized = false;
	uint8_t last_connected = OVEROSYNCSTATS_CONNECTED_FALSE;

	// Loop forever
	while (1) {
		// Wait for queue message
		if (PIOS_Queue_Receive(queue, &ev, PIOS_QUEUE_TIMEOUT_MAX) == true) {

			// For the first seconds do not send updates to allow the
			// overo to boot.  Then enable it and act normally.
			if (!initialized && PIOS_Thread_Systime() < 5000) {
				continue;
			} else if (!initialized) {
				initialized = true;
				PIOS_OVERO_Enable(pios_overo_id);
			}

			// Process event.  This calls transmitData
			UAVTalkSendObjectTimestamped(uavTalkCon, ev.obj, ev.instId, false, 0);
			
			updateTime = PIOS_Thread_Systime();
			if(((uint32_t) (updateTime - lastUpdateTime)) > 1000) {
				// Update stats.  This will trigger a local send event too
				OveroSyncStatsData syncStats;
				syncStats.Send = overosync->sent_bytes;
				syncStats.Connected = syncStats.Send > 500 ? OVEROSYNCSTATS_CONNECTED_TRUE : OVEROSYNCSTATS_CONNECTED_FALSE;
				syncStats.DroppedUpdates = overosync->failed_objects;
				syncStats.Packets = PIOS_OVERO_GetPacketCount(pios_overo_id);
				OveroSyncStatsSet(&syncStats);
				overosync->failed_objects = 0;
				overosync->sent_bytes = 0;
				lastUpdateTime = updateTime;

				// When first connected, send all the settings.  Right now this
				// will fail since all the settings will overfill the buffer and
				if (last_connected == OVEROSYNCSTATS_CONNECTED_FALSE &&
					syncStats.Connected == OVEROSYNCSTATS_CONNECTED_TRUE) {
					UAVObjIterate(&send_settings);
				}

				// Because the previous code only happens on connection and the
				// remote logging program doesn't send the settings to the log
				// when arming starts we send all settings every thirty seconds
				static uint32_t second_count = 0;
				if (second_count ++ > 30) {
					UAVObjIterate(&send_settings);
					second_count = 0;
				}
				last_connected = syncStats.Connected;
			}

			// TODO: Check the receive buffer
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
static int32_t pack_data(uint8_t * data, int32_t length)
{
	if( PIOS_COM_SendBufferNonBlocking(pios_com_overo_id, data, length) < 0)
		goto fail;

	overosync->sent_bytes += length;

	return length;

fail:
	overosync->failed_objects++;
	return -1;
}

/**
  * @}
  * @}
  */
