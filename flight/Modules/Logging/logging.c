/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup Logging Logging Module
 * @{ 
 *
 * @file       logging.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      Forward a set of UAVObjects when updated out a PIOS_COM port
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
#include "pios_thread.h"
#include "pios_queue.h"
#include "uavobjectmanager.h"

#include "pios_streamfs.h"

#include "attitudeactual.h"
#include "accels.h"
#include "gyros.h"
#include "baroaltitude.h"
#include "flightstatus.h"
#include "gpsposition.h"
#include "gpstime.h"
#include "magnetometer.h"
#include "loggingsettings.h"
#include "loggingstats.h"

// Private constants
#define MAX_QUEUE_SIZE   200
#define STACK_SIZE_BYTES 1200
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW

// Linked-list macros
#define LL_APPEND(head,add)                                                      \
do {                                                                             \
  __typeof__(head) _tmp;                                                         \
  (add)->next=NULL;                                                              \
  if (head) {                                                                    \
    _tmp = head;                                                                 \
    while (_tmp->next) { _tmp = _tmp->next; }                                    \
    _tmp->next=(add);                                                            \
  } else {                                                                       \
    (head)=(add);                                                                \
  }                                                                              \
} while (0)

#define LL_FOREACH(head,el)                                                      \
    for(el=head;el;el=el->next)

// Private types

// linked list entry for storing the last log times
struct UAVOLogInfo {
	struct UAVOLogInfo *next;
	UAVObjHandle obj;
	uint32_t last_log;
	uint16_t logging_period;
} __attribute__((packed));

// Private variables
static UAVTalkConnection uavTalkCon;
static struct pios_thread *loggingTaskHandle;
static bool module_enabled;
static struct pios_queue *queue;
static struct UAVOLogInfo *log_info;
LoggingSettingsData settings;

// Private functions
static void    loggingTask(void *parameters);
static int32_t send_data(uint8_t *data, int32_t length);
static void    register_object(UAVObjHandle obj);
static void settingsUpdatedCb(UAVObjEvent * ev);

// Local variables
static uintptr_t logging_com_id;
static uint32_t written_bytes;

// External variables
extern uintptr_t streamfs_id;

/**
 * Initialise the Logging module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t LoggingInitialize(void)
{
	// TODO: make selectable
	logging_com_id = PIOS_COM_LOGGING;

#ifdef MODULE_Logging_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_LOGGING] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (!logging_com_id)
		module_enabled = false;

	if (!module_enabled)
		return -1;

	// Create object queues
	queue = PIOS_Queue_Create(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));

	LoggingStatsInitialize();
	LoggingSettingsInitialize();

	// Get settings a connect update callback
	LoggingSettingsGet(&settings);
	LoggingSettingsConnectCallback(settingsUpdatedCb);

	// Initialise UAVTalk
	uavTalkCon = UAVTalkInitialize(&send_data);

	return 0;
}

/**
 * Start the Logging module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t LoggingStart(void)
{
	//Check if module is enabled or not
	if (module_enabled == false) {
		return -1;
	}

	// Process all registered objects and connect queue for updates
	UAVObjIterate(&register_object);

	// Start logging task
	loggingTaskHandle = PIOS_Thread_Create(loggingTask, "Logging", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);

	TaskMonitorAdd(TASKINFO_RUNNING_LOGGING, loggingTaskHandle);
	
	return 0;
}

MODULE_INITCALL(LoggingInitialize, LoggingStart);

static void loggingTask(void *parameters)
{
	UAVObjEvent ev;
	struct UAVOLogInfo *info;
	uint32_t time_now;

	bool armed = false;
	bool write_open = false;
	bool read_open = false;
	int32_t read_sector = 0;
	uint8_t read_data[LOGGINGSTATS_FILESECTOR_NUMELEM];

	//PIOS_STREAMFS_Format(streamfs_id);

	LoggingStatsData loggingData;
	LoggingStatsGet(&loggingData);
	loggingData.BytesLogged = 0;
	loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
	loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);

	LoggingSettingsGet(&settings);

	if (settings.LogBehavior == LOGGINGSETTINGS_LOGBEHAVIOR_LOGONSTART) {
		loggingData.Operation = LOGGINGSTATS_OPERATION_LOGGING;
	} else {
		loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
	}

	LoggingStatsSet(&loggingData);

	// Loop forever
	while (1) {

		LoggingStatsGet(&loggingData);

		// Check for change in armed state if logging on armed
		LoggingSettingsGet(&settings);
		if (settings.LogBehavior == LOGGINGSETTINGS_LOGBEHAVIOR_LOGONARM) {
			FlightStatusData flightStatus;
			FlightStatusGet(&flightStatus);

			if (flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMED && !armed) {
				// Start logging because just armed
				loggingData.Operation = LOGGINGSTATS_OPERATION_LOGGING;
				armed = true;
				LoggingStatsSet(&loggingData);
			} else if (flightStatus.Armed == FLIGHTSTATUS_ARMED_DISARMED && armed) {
				loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
				armed = false;
				LoggingStatsSet(&loggingData);
			}
		}

		switch (loggingData.Operation) {
		case LOGGINGSTATS_OPERATION_LOGGING:
			// Close the file if it is open for reading
			if (read_open) {
				PIOS_STREAMFS_Close(streamfs_id);
				read_open = false;
			}
			// Open the file if it is not open for writing
			if (!write_open) {
				if (PIOS_STREAMFS_OpenWrite(streamfs_id) != 0) {
					loggingData.Operation = LOGGINGSTATS_OPERATION_ERROR;
					continue;
				} else {
					write_open = true;
				}
				loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
				loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);
				LoggingStatsSet(&loggingData);
			}

			// Log the registered objects. Use a loop so we can write multiple UAVOs
			for (int ii = 0; ii < 16; ii++) {
				if (PIOS_Queue_Receive(queue, &ev, PIOS_QUEUE_TIMEOUT_MAX) == true) {

					// find log info entry and log the object if found
					LL_FOREACH(log_info, info) {
						if (info->obj == ev.obj) {
							time_now = PIOS_Thread_Systime();
							// logging_period == 1 means log every sample, so we can log faster than 1kHz
							if (time_now - info->last_log > info->logging_period || info->logging_period == 1) {
								UAVTalkSendObjectTimestamped(uavTalkCon, ev.obj, ev.instId, false, 0);
								info->last_log = time_now;
							}
							break;
						}
					}
				}
			}

			LoggingStatsBytesLoggedSet(&written_bytes);
			break;

		case LOGGINGSTATS_OPERATION_DOWNLOAD:
			if (!read_open) {
				// Start reading
				if (PIOS_STREAMFS_OpenRead(streamfs_id, loggingData.FileRequest) != 0) {
					loggingData.Operation = LOGGINGSTATS_OPERATION_ERROR;
				} else {
					read_open = true;
					read_sector = -1;
				}
			}

			if (read_open && read_sector == loggingData.FileSectorNum) {
				// Request received for same sector. Reupdate.
				memcpy(loggingData.FileSector, read_data, LOGGINGSTATS_FILESECTOR_NUMELEM);
				loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
			} else if (read_open && (read_sector + 1) == loggingData.FileSectorNum) {
				int32_t bytes_read = PIOS_COM_ReceiveBuffer(logging_com_id, read_data, LOGGINGSTATS_FILESECTOR_NUMELEM, 1);

				if (bytes_read < 0 || bytes_read > LOGGINGSTATS_FILESECTOR_NUMELEM) {
					// close on error
					loggingData.Operation = LOGGINGSTATS_OPERATION_ERROR;
					PIOS_STREAMFS_Close(streamfs_id);
					read_open = false;
				} else if (bytes_read < LOGGINGSTATS_FILESECTOR_NUMELEM) {

					// Check it has really run out of bytes by reading again
					int32_t bytes_read2 = PIOS_COM_ReceiveBuffer(logging_com_id, &read_data[bytes_read], LOGGINGSTATS_FILESECTOR_NUMELEM - bytes_read, 1);
					memcpy(loggingData.FileSector, read_data, LOGGINGSTATS_FILESECTOR_NUMELEM);

					if ((bytes_read + bytes_read2) < LOGGINGSTATS_FILESECTOR_NUMELEM) {
						// indicate end of file
						loggingData.Operation = LOGGINGSTATS_OPERATION_COMPLETE;
						PIOS_STREAMFS_Close(streamfs_id);
						read_open = false;
					} else {
						// Indicate sent
						loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
					}
				} else {
					// Indicate sent
					loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
					memcpy(loggingData.FileSector, read_data, LOGGINGSTATS_FILESECTOR_NUMELEM);
				}
				read_sector = loggingData.FileSectorNum;
			}
			LoggingStatsSet(&loggingData);
			// fall-through to default case
		default:
			// Empty the queue when we are not logging. This also makes sure that we
			// are not hogging the processor
			PIOS_Queue_Receive(queue, &ev, PIOS_QUEUE_TIMEOUT_MAX);
			// Close the file if necessary
			if (write_open) {
				PIOS_STREAMFS_Close(streamfs_id);
				loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
				loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);
				LoggingStatsSet(&loggingData);
				write_open = false;
			}
		}
	}
}

/**
 * Forward data from UAVTalk out the serial port
 * \param[in] data Data buffer to send
 * \param[in] length Length of buffer
 * \return -1 on failure
 * \return number of bytes transmitted on success
 */
static int32_t send_data(uint8_t *data, int32_t length)
{
	if( PIOS_COM_SendBuffer(logging_com_id, data, length) < 0)
		return -1;

	written_bytes += length;

	return length;
}

/**
 * Register a new object, adds object to local list and connects the queue depending on the object's
 * telemetry settings.
 * \param[in] obj Object to connect
 */
static void register_object(UAVObjHandle obj)
{
	// check whether we want to log this object
	UAVObjMetadata meta_data;
	if (UAVObjGetMetadata(obj, &meta_data) < 0)
		return;

	if (meta_data.loggingUpdatePeriod == 0)
		return;

	// register callback
	int32_t eventMask;
	eventMask = EV_UPDATED | EV_UNPACKED;
	UAVObjConnectQueue(obj, queue, eventMask);

	// create log info entry
	struct UAVOLogInfo *info;
	info = (struct UAVOLogInfo *) PIOS_malloc_no_dma(sizeof(struct UAVOLogInfo));
	if (info == NULL)
		return;

	info->obj = obj;
	info->last_log = 0;
	// store update period, so we don't have to fetch the metadata during logging
	info->logging_period = meta_data.loggingUpdatePeriod;
	LL_APPEND(log_info, info);
}

/** Callback to update settings
 */
static void settingsUpdatedCb(UAVObjEvent * ev)
{
	LoggingSettingsGet(&settings);
}

/**
  * @}
  * @}
  */
