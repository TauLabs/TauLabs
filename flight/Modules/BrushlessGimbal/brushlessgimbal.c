/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Gimbal output module
 * @brief Output the desired speeds
 * @{
 *
 * @file       actuator.c
 * @author     Tau Labs, http://github.com/TauLabs Copyright (C) 2013.
 * @brief      Drives the gimbal outputs
 *
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
#include "actuatordesired.h"

// Private constants
#define MAX_QUEUE_SIZE 2

#if defined(PIOS_ACTUATOR_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_ACTUATOR_STACK_SIZE
#else
#define STACK_SIZE_BYTES 1312
#endif

#define TASK_PRIORITY (tskIDLE_PRIORITY+4)
#define FAILSAFE_TIMEOUT_MS 100

// Private types

// Private variables
static xQueueHandle queue;
static xTaskHandle taskHandle;

// Private functions
static void brushlessGimbalTask(void* parameters);

/**
 * @brief Module initialization
 * @return 0
 */
int32_t BrushlessGimbalStart()
{
	// Start main task
	xTaskCreate(brushlessGimbalTask, (signed char*)"BrushlessGimbal", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_ACTUATOR, taskHandle);
	PIOS_WDG_RegisterFlag(PIOS_WDG_ACTUATOR);

	return 0;
}

/**
 * @brief Module initialization
 * @return 0
 */
int32_t BrushlessGimbalInitialize()
{
	// Listen for ActuatorDesired updates (Primary input to this module)
	ActuatorDesiredInitialize();
	queue = xQueueCreate(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));
	ActuatorDesiredConnectQueue(queue);

	return 0;
}
MODULE_INITCALL(BrushlessGimbalInitialize, BrushlessGimbalStart)

/**
 * @brief Main Actuator module task
 *
 * Universal matrix based mixer for VTOL, helis and fixed wing.
 * Converts desired roll,pitch,yaw and throttle to servo/ESC outputs.
 *
 * Because of how the Throttle ranges from 0 to 1, the motors should too!
 *
 * Note this code depends on the UAVObjects for the mixers being all being the same
 * and in sequence. If you change the object definition, make sure you check the code!
 *
 * @return -1 if error, 0 if success
 */
static void brushlessGimbalTask(void* parameters)
{
	UAVObjEvent ev;

	PIOS_Brushless_SetUpdateRate(30000);

	while (1)
	{
		PIOS_WDG_UpdateFlag(PIOS_WDG_ACTUATOR);

		// Wait until the ActuatorDesired object is updated
		xQueueReceive(queue, &ev, 1);

		ActuatorDesiredData actuatorDesired;
		ActuatorDesiredGet(&actuatorDesired);

		PIOS_Brushless_SetSpeed(0, actuatorDesired.Pitch * 30);
		PIOS_Brushless_SetSpeed(1, actuatorDesired.Roll * 30);
	}
}

/**
 * @}
 * @}
 */
