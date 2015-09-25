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
#include "utlist.h"
#include "misc_math.h"
#include "timeutils.h"

#include "pios_streamfs.h"
#include <pios_board_info.h>

#include "flightstatus.h"
#include "loggingsettings.h"
#include "loggingstats.h"

// Private constants
#define STACK_SIZE_BYTES 1000
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW
const char DIGITS[16] = "0123456789abcdef";

// Threshold to decide whether a private queue is used for an UAVO type.
// This is somewhat arbitrary; 50ms seems to be a good tradeoff between
// memory requirements and logging performance.
#define PRIVATE_QUEUE_THRESHOLD 50

#define LOGGING_PERIOD_MS 10
#define SHARED_QUEUE_SIZE 64
#define PRIVATE_QUEUE_SIZE 2

// Private types

// linked list entry for storing the last log times
struct UAVOLogInfo {
	struct UAVOLogInfo *next;
	UAVObjHandle obj;
	uint16_t logging_period;
	uint32_t next_log;
	struct pios_queue *queue;
} __attribute__((packed));

// Private variables
static UAVTalkConnection uavTalkCon;
static struct pios_thread *loggingTaskHandle;
static bool module_enabled;
static struct UAVOLogInfo *log_info;
static LoggingSettingsData settings;
static LoggingStatsData loggingData;
struct pios_queue *shared_queue;

// Private functions
static void    loggingTask(void *parameters);
static int32_t send_data(uint8_t *data, int32_t length);
static int info_cmp(struct UAVOLogInfo *info_1, struct UAVOLogInfo *info_2);
static void register_object(UAVObjHandle obj);
static void writeHeader();
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

	// create shared queue
	shared_queue = PIOS_Queue_Create(SHARED_QUEUE_SIZE, sizeof(UAVObjEvent));
	if (!shared_queue){
		return -1;
	}

	// Process all registered objects and connect queue for updates
	UAVObjIterate(&register_object);

	// Sort the list, so that entries with smaller logging periods and objects
	// with private queues come first
	LL_SORT(log_info, info_cmp);

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

	bool armed = false;
	bool write_open = false;
	bool read_open = false;
	uint32_t now = PIOS_Thread_Systime();
	int32_t read_sector = 0;
	uint8_t read_data[LOGGINGSTATS_FILESECTOR_NUMELEM];

	//PIOS_STREAMFS_Format(streamfs_id);

	LoggingStatsGet(&loggingData);
	loggingData.BytesLogged = 0;
	loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
	loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);

	LoggingSettingsGet(&settings);

	if (settings.LogBehavior == LOGGINGSETTINGS_LOGBEHAVIOR_LOGONSTART) {
		loggingData.Operation = LOGGINGSTATS_OPERATION_LOGGING_INITIALIZING;
	} else {
		loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
	}

	LoggingStatsSet(&loggingData);

	// Loop forever
	while (1) {

		LoggingStatsGet(&loggingData);

		// Check for change in armed state if logging on armed
		if (settings.LogBehavior == LOGGINGSETTINGS_LOGBEHAVIOR_LOGONARM) {
			FlightStatusData flightStatus;
			FlightStatusGet(&flightStatus);

			if (flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMED && !armed) {
				// Start logging because just armed
				loggingData.Operation = LOGGINGSTATS_OPERATION_LOGGING_INITIALIZING;
				armed = true;
				LoggingStatsSet(&loggingData);
			} else if (flightStatus.Armed == FLIGHTSTATUS_ARMED_DISARMED && armed) {
				loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
				armed = false;
				LoggingStatsSet(&loggingData);
			}
		}

		switch (loggingData.Operation) {
		case LOGGINGSTATS_OPERATION_FORMAT:
			// Format the file system
			if (read_open  || write_open) {
				PIOS_STREAMFS_Close(streamfs_id);
				read_open = false;
				write_open = false;
			}

			PIOS_STREAMFS_Format(streamfs_id);
			loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
			LoggingStatsSet(&loggingData);
			break;
		case LOGGINGSTATS_OPERATION_LOGGING_INITIALIZING:
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

				// Write information at start of the log file
				writeHeader();

				// When the file is first created, traverse the linked list and
				// log all UAVObjects, settings and data alike.
				LL_FOREACH(log_info, info) {
					// Iterate over all instances.
					uint16_t numInstances = UAVObjGetNumInstances(info->obj);
					for (int instId=0; instId<numInstances; instId++) {
						UAVTalkSendObjectTimestamped(uavTalkCon, info->obj, instId, false, 0);
					}
					// use a random offset, so that not all objects are logged at the same time
					info->next_log =  PIOS_Thread_Systime() + randomize_int(info->logging_period);
				}
			}
			// Empty the queues
			while(PIOS_Queue_Receive(shared_queue, &ev, 0))
			LL_FOREACH(log_info, info) {
				if (info->queue == shared_queue){
					// The list is sorted based on the logging period, so we can stop as soon
					// as we have an item without a private queue (as subsequent items won't 
					// have a private queue either).
					break;
				}
				while(PIOS_Queue_Receive(info->queue, &ev, 0)){};
			}
			LoggingStatsBytesLoggedSet(&written_bytes);
			loggingData.Operation = LOGGINGSTATS_OPERATION_LOGGING;
			LoggingStatsSet(&loggingData);
			break;
		case LOGGINGSTATS_OPERATION_LOGGING:
			{
				// Sleep between writing
				PIOS_Thread_Sleep_Until(&now, LOGGING_PERIOD_MS);

				// Log the objects with private queues
				for (int i=0; i<PRIVATE_QUEUE_SIZE; i++){
					LL_FOREACH(log_info, info) {
						if (info->queue == shared_queue){
							break; // the list is sorted (see comment above)
						}
						if(PIOS_Queue_Receive(info->queue, &ev, 0) == true) {
							UAVTalkSendObjectTimestamped(uavTalkCon, ev.obj, ev.instId, false, 0);
						}
					}
				}

				// Log the objects registred to the shared queue
				while (PIOS_Queue_Receive(shared_queue, &ev, 0) == true) {
					UAVTalkSendObjectTimestamped(uavTalkCon, ev.obj, ev.instId, false, 0);
				}
				LoggingStatsBytesLoggedSet(&written_bytes);

				now = PIOS_Thread_Systime();
			}
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
			//  Makes sure that we are not hogging the processor
			PIOS_Thread_Sleep(10);
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
 * Function for sorting linked list based on logging period and makes
 * sure that objects with private queues come first.
 */
static int info_cmp(struct UAVOLogInfo *info_1, struct UAVOLogInfo *info_2)
{
	if ((info_1->queue == shared_queue) && (info_2->queue != shared_queue)){
		return true;
	}
	if ((info_1->queue != shared_queue) && (info_2->queue == shared_queue)){
		return false;
	}

	return info_1->logging_period > info_2->logging_period;
}

/**
 * @brief Callback for adding an object to the logging queue
 * @param ev the event
 */
static void obj_updated_callback(UAVObjEvent * ev)
{
	struct UAVOLogInfo *info;
	uint32_t now;

	if (loggingData.Operation != LOGGINGSTATS_OPERATION_LOGGING){
		// We are not logging, so all events are discarded
		return;
	}
	LL_FOREACH(log_info, info) {
		if (info->obj == ev->obj){
			now = PIOS_Thread_Systime();
			if (info->next_log <= now){
				// compute the time of the next logging operation
				while (info->next_log < now) {
					info->next_log += info->logging_period;
				};
				// Ad the item to either the shared or the private queue
				// Note: PIOS_Queue_Send() copies the object, so it isn't
				// necessary to do the copy here.
				PIOS_Queue_Send(info->queue, ev, 0);
			}
			return;
		}
	}
}

/**
 * Register a new object, adds object to local list and connects the update callback
 * \param[in] obj Object to connect
 */
static void register_object(UAVObjHandle obj)
{
	struct pios_queue *queue;

	// check whether we want to log this object
	UAVObjMetadata meta_data;
	if (UAVObjGetMetadata(obj, &meta_data) < 0)
		return;

	if (meta_data.loggingUpdatePeriod == 0)
		return;

	// create log info entry
	struct UAVOLogInfo *info;
	info = (struct UAVOLogInfo *) PIOS_malloc_no_dma(sizeof(struct UAVOLogInfo));
	if (info == NULL){
		return;
	}

	info->obj = obj;
	// nothing can be logged faster than the logging period
	info->logging_period = MAX(meta_data.loggingUpdatePeriod, LOGGING_PERIOD_MS);

	if (meta_data.loggingUpdatePeriod <= PRIVATE_QUEUE_THRESHOLD && !UAVObjIsSettings(obj)){
		// this data object is logged frequently: use a private queue
		queue =  PIOS_Queue_Create(PRIVATE_QUEUE_SIZE, sizeof(UAVObjEvent));
		if (queue == NULL){
			// queue allocation failed
			PIOS_free((void *)info);
			return;
		}
		info->queue = queue;
	}
	else {
		// this is slowly logged data object or a settings object: use shared queue
		info->queue = shared_queue;
	}

	int32_t event_mask = EV_UPDATED | EV_UNPACKED;
	UAVObjConnectCallback(obj, obj_updated_callback, event_mask);
	LL_APPEND(log_info, info);
}


/**
 * Write log file header
 * see firmwareinfotemplate.c
 */
static void writeHeader()
{
	int pos;
#define STR_BUF_LEN 45
	char tmp_str[STR_BUF_LEN];
	char *info_str;
	char this_char;
	DateTimeT date_time;

	const struct pios_board_info * bdinfo = &pios_board_info_blob;

	// Header
	#define LOG_HEADER "Tau Labs git hash:\n"
	send_data((uint8_t *)LOG_HEADER, strlen(LOG_HEADER));

	// Commit tag name
	info_str = (char*)(bdinfo->fw_base + bdinfo->fw_size + 14);
	send_data((uint8_t*)info_str, strlen(info_str));

	// Git commit hash
	pos = 0;
	tmp_str[pos++] = ':';
	for (int i = 0; i < 4; i++){
		this_char = *(char*)(bdinfo->fw_base + bdinfo->fw_size + 7 - i);
		tmp_str[pos++] = DIGITS[(this_char & 0xF0) >> 4];
		tmp_str[pos++] = DIGITS[(this_char & 0x0F)];
	}
	send_data((uint8_t*)tmp_str, pos);

	// Date
	date_from_timestamp(*(uint32_t *)(bdinfo->fw_base + bdinfo->fw_size + 8), &date_time);
	uint8_t len = snprintf(tmp_str, STR_BUF_LEN, " %d%02d%02d\n", 1900 + date_time.year, date_time.mon + 1, date_time.mday);
	send_data((uint8_t*)tmp_str, len);

	// UAVO SHA1
	pos = 0;
	for (int i = 0; i < 20; i++){
		this_char = *(char*)(bdinfo->fw_base + bdinfo->fw_size + 60 + i);
		tmp_str[pos++] = DIGITS[(this_char & 0xF0) >> 4];
		tmp_str[pos++] = DIGITS[(this_char & 0x0F)];
	}
	tmp_str[pos++] = '\n';
	send_data((uint8_t*)tmp_str, pos);
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
