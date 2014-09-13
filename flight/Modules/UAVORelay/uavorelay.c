/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup UAVORelay UAVORelay Module
 * @{ 
 *
 * @file       uavorelay.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
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
#include "pios_can.h"

#include "cameradesired.h"
#include "pios_thread.h"
#include "pios_queue.h"

// Private constants
#define STACK_SIZE_BYTES 512
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW

// Private types

// Private variables
<<<<<<< HEAD
static struct pios_queue *queue;
static UAVTalkConnection uavTalkCon;
=======
static xQueueHandle queue;
>>>>>>> UAVORelay: receive CAN messages for gimbal
static struct pios_thread *uavoRelayTaskHandle;
static bool module_enabled;

// Private functions
static void    uavoRelayTask(void *parameters);

// External variables
extern uintptr_t pios_can_id;

/**
 * Initialise the UAVORelay module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t UAVORelayInitialize(void)
{

#ifdef MODULE_UAVORelay_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_UAVORELAY] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (!module_enabled)
		return -1;

	// Create object queues
	queue = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_GIMBAL);

	CameraDesiredInitialize();

	return 0;
}

/**
 * Start the UAVORelay module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t UAVORelayStart(void)
{
	//Check if module is enabled or not
	if (module_enabled == false) {
		return -1;
	}
	
	// Start relay task
	uavoRelayTaskHandle = PIOS_Thread_Create(
			uavoRelayTask, "UAVORelay", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);

	TaskMonitorAdd(TASKINFO_RUNNING_UAVORELAY, uavoRelayTaskHandle);
	
	return 0;
}

MODULE_INITCALL(UAVORelayInitialize, UAVORelayStart)
;

static void uavoRelayTask(void *parameters)
{

	// Loop forever
	while (1) {

		struct pios_can_gimbal_message bgc_message;

		// Wait for queue message
		if (PIOS_Queue_Receive(queue, &bgc_message, 5) == true) {
			CameraDesiredData cameraDesired;
			CameraDesiredGet(&cameraDesired);

			cameraDesired.Declination = bgc_message.setpoint_pitch;
			cameraDesired.Bearing = bgc_message.setpoint_yaw;
			cameraDesired.Roll = bgc_message.fc_roll;
			cameraDesired.Pitch = bgc_message.fc_pitch;
			cameraDesired.Yaw = bgc_message.fc_yaw;
			CameraDesiredSet(&cameraDesired);
		}

	}
}


/**
  * @}
  * @}
  */
