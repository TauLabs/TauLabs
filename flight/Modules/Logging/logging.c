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
#include "pios_mutex.h"
#include "uavobjectmanager.h"
#include "misc_math.h"
#include "timeutils.h"
#include "uavobjectmanager.h"

#include "pios_streamfs.h"
#include <pios_board_info.h>

#include "accels.h"
#include "actuatorcommand.h"
#include "airspeedactual.h"
#include "attitudeactual.h"
#include "baroaltitude.h"
#include "flightbatterystate.h"
#include "flightstatus.h"
#include "gpsposition.h"
#include "gpstime.h"
#include "gpssatellites.h"
#include "gyros.h"
#include "loggingsettings.h"
#include "loggingstats.h"
#include "magnetometer.h"
#include "manualcontrolcommand.h"
#include "positionactual.h"
#include "systemalarms.h"
#include "systemident.h"
#include "velocityactual.h"
#include "waypointactive.h"

// Private constants
#define STACK_SIZE_BYTES 1200
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW
const char DIGITS[16] = "0123456789abcdef";

#define LOGGING_PERIOD_MS 10
#define LOGGING_QUEUE_SIZE 64

// Private types

// Private variables
static UAVTalkConnection uavTalkCon;
static struct pios_thread *loggingTaskHandle;
static bool module_enabled;
static LoggingSettingsData settings;
static LoggingStatsData loggingData;
struct pios_queue *logging_queue;
static struct pios_recursive_mutex *mutex;

// Private functions
static void    loggingTask(void *parameters);
static int32_t send_data(uint8_t *data, int32_t length);
static uint16_t get_minimum_logging_period();
static void unregister_object(UAVObjHandle obj);
static void register_object(UAVObjHandle obj);
static void register_default_profile();
static void logSettings(UAVObjHandle obj);
static void SettingsUpdatedCb(UAVObjEvent * ev);
static void writeHeader();

// Local variables
static uintptr_t logging_com_id;
static uint32_t written_bytes;
static bool destination_spi_flash;

#if defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC)
// External variables
extern uintptr_t streamfs_id;
#endif

/**
 * Initialise the Logging module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t LoggingInitialize(void)
{
	if (PIOS_COM_OPENLOG) {
		logging_com_id = PIOS_COM_OPENLOG;
		destination_spi_flash = false;
	}
	else if (PIOS_COM_SPIFLASH) {
		logging_com_id = PIOS_COM_SPIFLASH;
		destination_spi_flash = true;
	}
		
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

	// create logging queue
	logging_queue = PIOS_Queue_Create(LOGGING_QUEUE_SIZE, sizeof(UAVObjEvent));
	if (!logging_queue){
		return -1;
	}

	// Create mutex
	mutex = PIOS_Recursive_Mutex_Create();
	if (mutex == NULL){
		return -2;
	}

	// Start logging task
	loggingTaskHandle = PIOS_Thread_Create(loggingTask, "Logging", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);

	TaskMonitorAdd(TASKINFO_RUNNING_LOGGING, loggingTaskHandle);
	
	return 0;
}

MODULE_INITCALL(LoggingInitialize, LoggingStart);

static void loggingTask(void *parameters)
{
	UAVObjEvent ev;

	bool armed = false;
	uint32_t now = PIOS_Thread_Systime();

#if defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC)
	bool write_open = false;
	bool read_open = false;
	int32_t read_sector = 0;
	uint8_t read_data[LOGGINGSTATS_FILESECTOR_NUMELEM];
#endif

	// Get settings and connect callback
	LoggingSettingsGet(&settings);
	LoggingSettingsConnectCallback(SettingsUpdatedCb);

	LoggingStatsGet(&loggingData);
	loggingData.BytesLogged = 0;
	
#if defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC)
	if (destination_spi_flash)
	{
		loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
		loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);
	}
#endif

	if (settings.LogBehavior == LOGGINGSETTINGS_LOGBEHAVIOR_LOGONSTART) {
		loggingData.Operation = LOGGINGSTATS_OPERATION_INITIALIZING;
	} else {
		loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
	}

	LoggingStatsSet(&loggingData);

	// Loop forever
	while (1) 
	{
		LoggingStatsGet(&loggingData);

		// Check for change in armed state if logging on armed

		if (settings.LogBehavior == LOGGINGSETTINGS_LOGBEHAVIOR_LOGONARM) {
			FlightStatusData flightStatus;
			FlightStatusGet(&flightStatus);

			if (flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMED && !armed) {
				// Start logging because just armed
				loggingData.Operation = LOGGINGSTATS_OPERATION_INITIALIZING;
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
#if defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC)
			if (destination_spi_flash){
				if (read_open || write_open) {
					PIOS_STREAMFS_Close(streamfs_id);
					read_open = false;
					write_open = false;
				}

				PIOS_STREAMFS_Format(streamfs_id);
				loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
				loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);
			}
#endif /* defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC) */
			loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
			LoggingStatsSet(&loggingData);
			break;
		case LOGGINGSTATS_OPERATION_INITIALIZING:
			// Unregister all objects
			UAVObjIterate(&unregister_object);
			// Register objects to be logged
			switch (settings.Profile) {
				case LOGGINGSETTINGS_PROFILE_DEFAULT:
					register_default_profile();
					break;
				case LOGGINGSETTINGS_PROFILE_CUSTOM:
					UAVObjIterate(&register_object);
					break;
			}
#if defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC)
			if (destination_spi_flash){
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
			}
			else {
				read_open = false;
				write_open = true;
			}
#endif /* defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC) */

			// Write information at start of the log file
			writeHeader();

			// Log settings
			if (settings.LogSettingsOnStart == LOGGINGSETTINGS_LOGSETTINGSONSTART_TRUE){
				UAVObjIterate(&logSettings);
			}

			// Empty the queue
			while(PIOS_Queue_Receive(logging_queue, &ev, 0))

			LoggingStatsBytesLoggedSet(&written_bytes);
			loggingData.Operation = LOGGINGSTATS_OPERATION_LOGGING;
			LoggingStatsSet(&loggingData);
			break;
		case LOGGINGSTATS_OPERATION_LOGGING:
			{
				// Sleep between writing
				PIOS_Thread_Sleep_Until(&now, LOGGING_PERIOD_MS);

				// Log the objects registred to the shared queue
				for (int i=0; i<LOGGING_QUEUE_SIZE; i++) {
					if (PIOS_Queue_Receive(logging_queue, &ev, 0) == true) {
						UAVTalkSendObjectTimestamped(uavTalkCon, ev.obj, ev.instId, false, 0);
					}
					else {
						break;
					}
				}

				LoggingStatsBytesLoggedSet(&written_bytes);

				now = PIOS_Thread_Systime();
			}
			break;
		case LOGGINGSTATS_OPERATION_DOWNLOAD:
#if defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC)
			if (destination_spi_flash) {
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
			}
#endif /* defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC) */

			// fall-through to default case
		default:
			//  Makes sure that we are not hogging the processor
			PIOS_Thread_Sleep(10);
#if defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC)
			if (destination_spi_flash) {
				// Close the file if necessary
				if (write_open) {
					PIOS_STREAMFS_Close(streamfs_id);
					loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
					loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);
					LoggingStatsSet(&loggingData);
					write_open = false;
				}
			}
#endif /* defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC) */
		}
	}
}

/**
 * Log all settings objects
 * \param[in] obj Object to log
*/
static void logSettings(UAVObjHandle obj)
{
	if (UAVObjIsSettings(obj)) {
		UAVTalkSendObjectTimestamped(uavTalkCon, obj, 0, false, 0);
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
 * @brief Callback for adding an object to the logging queue
 * @param ev the event
 */
static void obj_updated_callback(UAVObjEvent * ev)
{
	if (loggingData.Operation != LOGGINGSTATS_OPERATION_LOGGING){
		// We are not logging, so all events are discarded
		return;
	}
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	PIOS_Queue_Send(logging_queue, ev, 0);
	PIOS_Recursive_Mutex_Unlock(mutex);
}


/**
 * Get the minimum logging period in milliseconds
*/
static uint16_t get_minimum_logging_period()
{
	uint16_t max_freq = 5;
	switch (settings.MaxLogRate) {
		case LOGGINGSETTINGS_MAXLOGRATE_5:
			max_freq = 5;
			break;
		case LOGGINGSETTINGS_MAXLOGRATE_10:
			max_freq = 10;
			break;
		case LOGGINGSETTINGS_MAXLOGRATE_25:
			max_freq = 25;
			break;
		case LOGGINGSETTINGS_MAXLOGRATE_50:
			max_freq = 50;
			break;
		case LOGGINGSETTINGS_MAXLOGRATE_100:
			max_freq = 100;
			break;
		case LOGGINGSETTINGS_MAXLOGRATE_250:
			max_freq = 250;
			break;
		case LOGGINGSETTINGS_MAXLOGRATE_500:
			max_freq = 500;
			break;
		case LOGGINGSETTINGS_MAXLOGRATE_1000:
			max_freq = 1000;
			break;
	}
	uint16_t min_period = 1000 / max_freq;
	return min_period;
}


/**
 * Unregister an object
 * \param[in] obj Object to unregister
 */
static void unregister_object(UAVObjHandle obj) {
	UAVObjDisconnectCallback(obj, obj_updated_callback);
}


/**
 * Register a new object: connect the update callback
 * \param[in] obj Object to connect
 */
static void register_object(UAVObjHandle obj)
{
	// check whether we want to log this object
	UAVObjMetadata meta_data;
	if (UAVObjGetMetadata(obj, &meta_data) < 0){
		return;
	}

	if (meta_data.loggingUpdatePeriod == 0){
		return;
	}

	uint16_t period = MAX(meta_data.loggingUpdatePeriod, get_minimum_logging_period());
	if (period == 1) {
		// log every update
		UAVObjConnectCallback(obj, obj_updated_callback, EV_UPDATED | EV_UNPACKED);
	}
	else {
		// log updates throttled
		UAVObjConnectCallbackThrottled(obj, obj_updated_callback, EV_UPDATED | EV_UNPACKED, period);
	}
}


/**
 * Register objects for the default logging profile
 */
static void register_default_profile()
{
	const uint32_t DEFAULT_PERIOD = 10;

	// For the default profile, we limit things to 100Hz (for now)
	uint16_t min_period = MAX(get_minimum_logging_period(), DEFAULT_PERIOD);

	// Objects for which we log all changes (use 100Hz to limit max data rate)
	UAVObjConnectCallbackThrottled(FlightStatusHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, DEFAULT_PERIOD);
	UAVObjConnectCallbackThrottled(SystemAlarmsHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, DEFAULT_PERIOD);
	if (WaypointActiveHandle()) {
		UAVObjConnectCallbackThrottled(WaypointActiveHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, DEFAULT_PERIOD);
	}

	if (SystemIdentHandle()){
		UAVObjConnectCallbackThrottled(SystemIdentHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, DEFAULT_PERIOD);
	}

	// Log fast
	UAVObjConnectCallbackThrottled(AccelsHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, min_period);
	UAVObjConnectCallbackThrottled(GyrosHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, min_period);

	// Log a bit slower
	UAVObjConnectCallbackThrottled(AttitudeActualHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 5 * min_period);
	UAVObjConnectCallbackThrottled(MagnetometerHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 5 * min_period);
	UAVObjConnectCallbackThrottled(ManualControlCommandHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 5 * min_period);
	UAVObjConnectCallbackThrottled(ActuatorCommandHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 5 * min_period);

	// Log slow
	if (FlightBatteryStateHandle()) {
		UAVObjConnectCallbackThrottled(FlightBatteryStateHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 10 * min_period);
	}
	if (BaroAltitudeHandle()) {
		UAVObjConnectCallbackThrottled(BaroAltitudeHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 10 * min_period);
	}
	if (AirspeedActualHandle()) {
		UAVObjConnectCallbackThrottled(AirspeedActualHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 10 * min_period);
	}
	if (GPSPositionHandle()) {
		UAVObjConnectCallbackThrottled(GPSPositionHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 10 * min_period);
	}
	if (PositionActualHandle()) {
		UAVObjConnectCallbackThrottled(PositionActualHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 10 * min_period);
	}
	if (VelocityActualHandle()) {
		UAVObjConnectCallbackThrottled(VelocityActualHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 10 * min_period);
	}

	// Log very slow
	if (GPSTimeHandle()) {
		UAVObjConnectCallbackThrottled(GPSTimeHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 50 * min_period);
	}

	// Log very very slow
	if (GPSSatellitesHandle()) {
		UAVObjConnectCallbackThrottled(GPSSatellitesHandle(), obj_updated_callback, EV_UPDATED | EV_UNPACKED, 500 * min_period);
	}
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

/**
 * Callback triggered when the module settings are updated
 */
static void SettingsUpdatedCb(UAVObjEvent * ev)
{
	LoggingSettingsGet(&settings);
}

/**
  * @}
  * @}
  */
