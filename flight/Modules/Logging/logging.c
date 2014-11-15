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
#include "gpsposition.h"
#include "gpstime.h"
#include "loggingstats.h"

// Private constants
#define STACK_SIZE_BYTES 600
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

	if (PIOS_STREAMFS_OpenWrite(streamfs_id) != 0)
		return -1;

	// Start logging task
	loggingTaskHandle = PIOS_Thread_Create(loggingTask, "Logging", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);

	TaskMonitorAdd(TASKINFO_RUNNING_LOGGING, loggingTaskHandle);
	
	return 0;
}

MODULE_INITCALL(LoggingInitialize, LoggingStart);

static void loggingTask(void *parameters)
{

	LoggingStatsData loggingData;
	LoggingStatsGet(&loggingData);
	loggingData.BytesLogged = 0;
	loggingData.MinFileId = PIOS_STREAMFS_MinFileId(streamfs_id);
	loggingData.MaxFileId = PIOS_STREAMFS_MaxFileId(streamfs_id);
	LoggingStatsSet(&loggingData);

	int i = 0;
	// Loop forever
	while (1) {

		// Do not update anything at more than 40 Hz
		PIOS_Thread_Sleep(20);

		UAVTalkSendObjectTimestamped(uavTalkCon, AttitudeActualHandle(), 0, false, 0);
		UAVTalkSendObjectTimestamped(uavTalkCon, AccelsHandle(), 0, false, 0);
		UAVTalkSendObjectTimestamped(uavTalkCon, GyrosHandle(), 0, false, 0);

		if ((i % 10) == 0) {
			UAVTalkSendObjectTimestamped(uavTalkCon, BaroAltitudeHandle(), 0, false, 0);
			UAVTalkSendObjectTimestamped(uavTalkCon, GPSPositionHandle(), 0, false, 0);
		}

		if ((i % 50) == 1) {
			UAVTalkSendObjectTimestamped(uavTalkCon, GPSTimeHandle(), 0, false, 0);	
		}

		LoggingStatsBytesLoggedSet(&written_bytes);

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
