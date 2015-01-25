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
#define STACK_SIZE_BYTES 900
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW

// Private types

// Private variables
static UAVTalkConnection uavTalkCon;
static struct pios_thread *loggingTaskHandle;
static bool module_enabled;

// Private functions
static void    loggingTask(void *parameters);
static int32_t send_data(uint8_t *data, int32_t length);

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
	int32_t read_sector = 0;
	uint8_t read_data[LOGGINGSTATS_FILESECTOR_NUMELEM];

	//PIOS_STREAMFS_Format(streamfs_id);

	LoggingStatsData loggingData;
	LoggingStatsGet(&loggingData);
	loggingData.BytesLogged = 0;
	loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
	loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);

	LoggingSettingsData settings;
	LoggingSettingsGet(&settings);

	if (settings.LogBehavior == LOGGINGSETTINGS_LOGBEHAVIOR_LOGONSTART) {
		if (PIOS_STREAMFS_OpenWrite(streamfs_id) != 0) {
			loggingData.Operation = LOGGINGSTATS_OPERATION_ERROR;
			write_open = false;
		} else {
			loggingData.Operation = LOGGINGSTATS_OPERATION_LOGGING;
			write_open = true;
		}
	} else {
		loggingData.Operation = LOGGINGSTATS_OPERATION_IDLE;
	}

	LoggingStatsSet(&loggingData);

	int i = 0;
	// Loop forever
	while (1) {

		// Do not update anything at more than 40 Hz
		PIOS_Thread_Sleep(20);

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

			UAVTalkSendObjectTimestamped(uavTalkCon, AttitudeActualHandle(), 0, false, 0);
			UAVTalkSendObjectTimestamped(uavTalkCon, AccelsHandle(), 0, false, 0);
			UAVTalkSendObjectTimestamped(uavTalkCon, GyrosHandle(), 0, false, 0);
			UAVTalkSendObjectTimestamped(uavTalkCon, MagnetometerHandle(), 0, false, 0);

			if ((i % 10) == 0) {
				UAVTalkSendObjectTimestamped(uavTalkCon, BaroAltitudeHandle(), 0, false, 0);
				UAVTalkSendObjectTimestamped(uavTalkCon, GPSPositionHandle(), 0, false, 0);
			}

			if ((i % 50) == 1) {
				UAVTalkSendObjectTimestamped(uavTalkCon, GPSTimeHandle(), 0, false, 0);	
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
