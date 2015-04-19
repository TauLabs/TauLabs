/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup OsdCan OSD CAN bus interface
 * @{
 *
 * @file       osdcan.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
 * @brief      Relay messages between OSD and FC
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
#include "pios_thread.h"
#include "pios_can.h"

#include "attitudeactual.h"

// Private constants
#define MAX_QUEUE_SIZE 2

#define STACK_SIZE_BYTES 1312
#define TASK_PRIORITY PIOS_THREAD_PRIO_NORMAL
#define FAILSAFE_TIMEOUT_MS 100

// Private types

// Private variables
static struct pios_queue *queue;
static struct pios_thread *taskHandle;

// Private functions
static void osdCanTask(void* parameters);

/**
 * @brief Module initialization
 * @return 0
 */
static int32_t OsdCanStart()
{
	// Start main task
	taskHandle = PIOS_Thread_Create(osdCanTask, "OsdCan", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
	//TaskMonitorAdd(TASKINFO_RUNNING_ACTUATOR, taskHandle);
	//PIOS_WDG_RegisterFlag(PIOS_WDG_ACTUATOR);

	return 0;
}

extern uintptr_t pios_can_id;

/**
 * @brief Module initialization
 * @return 0
 */
static int32_t OsdCanInitialize()
{
	// Listen for ActuatorDesired updates (Primary input to this module)
	AttitudeActualInitialize();

	// Create object queues
	queue = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_ATTITUDE_ROLL_PITCH);

	return 0;
}
MODULE_INITCALL(OsdCanInitialize, OsdCanStart);

/**
 * @brief Gimbal output control task
 */
static void osdCanTask(void* parameters)
{

	// Loop forever
	while (1) {

		struct pios_can_roll_pitch_message roll_pitch_message;

		// Wait for queue message
		if (PIOS_Queue_Receive(queue, &roll_pitch_message, 10) == true) {

			PIOS_LED_Toggle(PIOS_LED_ALARM);

			AttitudeActualData attitudeActual;
			AttitudeActualGet(&attitudeActual);
			attitudeActual.Roll = roll_pitch_message.fc_roll;
			attitudeActual.Pitch = roll_pitch_message.fc_pitch;
			AttitudeActualSet(&attitudeActual);
		}

	}

}



/**
 * @}
 * @}
 */
