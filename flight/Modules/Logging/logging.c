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
#include "timeutils.h"
#include "uavobjectmanager.h"

#include "pios_streamfs.h"
#include <pios_board_info.h>

#include "airspeedactual.h"
#include "attitudeactual.h"
#include "accels.h"
#include "actuatorcommand.h"
#include "flightstatus.h"
#include "gyros.h"
#include "baroaltitude.h"
#include "flightstatus.h"
#include "gpsposition.h"
#include "gpstime.h"
#include "gpssatellites.h"
#include "magnetometer.h"
#include "manualcontrolcommand.h"
#include "positionactual.h"
#include "loggingsettings.h"
#include "loggingstats.h"
#include "velocityactual.h"
#include "waypoint.h"
#include "waypointactive.h"

// Private constants
#define STACK_SIZE_BYTES 1200
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW
const char DIGITS[16] = "0123456789abcdef";

// Private types

// Private variables
static UAVTalkConnection uavTalkCon;
static struct pios_thread *loggingTaskHandle;
static bool module_enabled;
static LoggingSettingsData settings;
static bool flightstatus_updated = false;
static bool waypoint_updated = false;

// Private functions
static void    loggingTask(void *parameters);
static int32_t send_data(uint8_t *data, int32_t length);
static void logSettings(UAVObjHandle obj);
static void SettingsUpdatedCb(UAVObjEvent * ev);
static void FlightStatusUpdatedCb(UAVObjEvent * ev);
static void WaypointActiveUpdatedCb(UAVObjEvent * ev);
static void writeHeader();

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

	// Start logging task
	loggingTaskHandle = PIOS_Thread_Create(loggingTask, "Logging", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);

	TaskMonitorAdd(TASKINFO_RUNNING_LOGGING, loggingTaskHandle);
	
	return 0;
}

MODULE_INITCALL(LoggingInitialize, LoggingStart);

static void loggingTask(void *parameters)
{
	bool armed = false;
	bool write_open = false;
	bool read_open = false;
	bool first_run = true;
	int32_t read_sector = 0;
	uint8_t read_data[LOGGINGSTATS_FILESECTOR_NUMELEM];

	//PIOS_STREAMFS_Format(streamfs_id);

	// Get settings and connect callback
	LoggingSettingsGet(&settings);
	LoggingSettingsConnectCallback(SettingsUpdatedCb);

	// Connect callbacks for UAVOs being logged on change
	FlightStatusConnectCallback(FlightStatusUpdatedCb);
	WaypointActiveConnectCallback(WaypointActiveUpdatedCb);

	LoggingStatsData loggingData;
	LoggingStatsGet(&loggingData);
	loggingData.BytesLogged = 0;
	loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
	loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);

	if (settings.LogBehavior == LOGGINGSETTINGS_LOGBEHAVIOR_LOGONSTART) {
		if (PIOS_STREAMFS_OpenWrite(streamfs_id) != 0) {
			loggingData.Operation = LOGGINGSTATS_OPERATION_ERROR;
			write_open = false;
		} else {
			loggingData.Operation = LOGGINGSTATS_OPERATION_LOGGING;
			write_open = true;
			first_run = true;
		}
	} else {
		loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
	}

	LoggingStatsSet(&loggingData);

	int i = 0;
	// Loop forever
	while (1) {

		// Sleep for some time depending on logging rate
		switch(settings.MaxLogRate){
			case LOGGINGSETTINGS_MAXLOGRATE_5:
				PIOS_Thread_Sleep(200);
				break;
			case LOGGINGSETTINGS_MAXLOGRATE_10:
				PIOS_Thread_Sleep(100);
				break;
			case LOGGINGSETTINGS_MAXLOGRATE_25:
				PIOS_Thread_Sleep(40);
				break;
			case LOGGINGSETTINGS_MAXLOGRATE_50:
				PIOS_Thread_Sleep(20);
				break;
			case LOGGINGSETTINGS_MAXLOGRATE_100:
				PIOS_Thread_Sleep(10);
				break;
			default:
				PIOS_Thread_Sleep(1000);
		}

		LoggingStatsGet(&loggingData);

		// Check for change in armed state if logging on armed

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


		// If currently downloading a log, close the file
		if (loggingData.Operation == LOGGINGSTATS_OPERATION_LOGGING && read_open) {
			PIOS_STREAMFS_Close(streamfs_id);
			read_open = false;
		}

		if (loggingData.Operation == LOGGINGSTATS_OPERATION_LOGGING && !write_open) {
			if (PIOS_STREAMFS_OpenWrite(streamfs_id) != 0) {
				loggingData.Operation = LOGGINGSTATS_OPERATION_ERROR;
			} else {
				write_open = true;
			}
			loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
			loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);
			LoggingStatsSet(&loggingData);
		} else if (loggingData.Operation != LOGGINGSTATS_OPERATION_LOGGING && write_open) {
			PIOS_STREAMFS_Close(streamfs_id);
			loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
			loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);
			LoggingStatsSet(&loggingData);
			write_open = false;
		}

		switch (loggingData.Operation) {
		case LOGGINGSTATS_OPERATION_LOGGING:
			if (!write_open)
				continue;

			if (first_run){
				// Write information at start of the log file
				writeHeader();

				// Log settings
				if (settings.LogSettingsOnStart == LOGGINGSETTINGS_LOGSETTINGSONSTART_TRUE){
					UAVObjIterate(&logSettings);
				}

				// Log some data objects that are unlikely to change during flight
				// Waypoints
				for (int i = 0; i < UAVObjGetNumInstances(WaypointHandle()); i++) {
					UAVTalkSendObjectTimestamped(uavTalkCon, WaypointHandle(), i, false, 0);
				}

				// Trigger logging for objects that are logged on change
				flightstatus_updated = true;
				waypoint_updated = true;

				first_run = false;
			}

			// Log objects on change
			if (flightstatus_updated){
				UAVTalkSendObjectTimestamped(uavTalkCon, FlightStatusHandle(), 0, false, 0);
				flightstatus_updated = false;
			}

			if (waypoint_updated){
				UAVTalkSendObjectTimestamped(uavTalkCon, WaypointActiveHandle(), 0, false, 0);
				waypoint_updated = false;
			}

			// Log very fast
			UAVTalkSendObjectTimestamped(uavTalkCon, AccelsHandle(), 0, false, 0);
			UAVTalkSendObjectTimestamped(uavTalkCon, GyrosHandle(), 0, false, 0);

			// Log a bit slower
			if ((i % 2) == 0) {
				UAVTalkSendObjectTimestamped(uavTalkCon, AttitudeActualHandle(), 0, false, 0);
				UAVTalkSendObjectTimestamped(uavTalkCon, MagnetometerHandle(), 0, false, 0);
				UAVTalkSendObjectTimestamped(uavTalkCon, ManualControlCommandHandle(), 0, false, 0);
				UAVTalkSendObjectTimestamped(uavTalkCon, ActuatorCommandHandle(), 0, false, 0);
			}

			// Log slower
			if ((i % 10) == 1) {
				UAVTalkSendObjectTimestamped(uavTalkCon, AirspeedActualHandle(), 0, false, 0);
				UAVTalkSendObjectTimestamped(uavTalkCon, BaroAltitudeHandle(), 0, false, 0);
				UAVTalkSendObjectTimestamped(uavTalkCon, GPSPositionHandle(), 0, false, 0);
				UAVTalkSendObjectTimestamped(uavTalkCon, PositionActualHandle(), 0, false, 0);
				UAVTalkSendObjectTimestamped(uavTalkCon, VelocityActualHandle(), 0, false, 0);
			}

			// Log slow
			if ((i % 50) == 2) {
				UAVTalkSendObjectTimestamped(uavTalkCon, GPSTimeHandle(), 0, false, 0);
			}

			// Log very slow
			if ((i % 500) == 3) {
				UAVTalkSendObjectTimestamped(uavTalkCon, GPSSatellitesHandle(), 0, false, 0);
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

		}

		i++;
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
 * Callback triggered when FlightStatus is updated
 */
static void FlightStatusUpdatedCb(UAVObjEvent * ev)
{
	flightstatus_updated = true;
}

/**
 * Callback triggered when WaypointActive is updated
 */
static void WaypointActiveUpdatedCb(UAVObjEvent * ev)
{
	waypoint_updated = true;
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
  * @}
  * @}
  */
