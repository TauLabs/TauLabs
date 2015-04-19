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
#include "baroaltitude.h"
#include "flightstatus.h"

// Private constants
#define MAX_QUEUE_SIZE 2

#define STACK_SIZE_BYTES 1312
#define TASK_PRIORITY PIOS_THREAD_PRIO_NORMAL
#define FAILSAFE_TIMEOUT_MS 100

// Private types

// Private variables
static struct pios_queue *queue_roll_pitch;
static struct pios_queue *queue_yaw;
static struct pios_queue *queue_altitude;
static struct pios_queue *queue_flightstatus;
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
	queue_roll_pitch = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_ATTITUDE_ROLL_PITCH);
	queue_yaw = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_ATTITUDE_YAW);
	queue_flightstatus = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_FLIGHTSTATUS);
	queue_altitude = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_ALT);

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
		struct pios_can_yaw_message pios_can_yaw_message;
		struct pios_can_flightstatus_message pios_can_flightstatus_message;
		struct pios_can_alt_message pios_can_alt_message;

		// Wait for queue message
		if (PIOS_Queue_Receive(queue_roll_pitch, &roll_pitch_message, 0) == true) {
			AttitudeActualData attitudeActual;
			AttitudeActualGet(&attitudeActual);
			attitudeActual.Roll = roll_pitch_message.fc_roll;
			attitudeActual.Pitch = roll_pitch_message.fc_pitch;
			AttitudeActualSet(&attitudeActual);
		}

		if (PIOS_Queue_Receive(queue_yaw, &pios_can_yaw_message, 0) == true) {
			AttitudeActualData attitudeActual;
			AttitudeActualGet(&attitudeActual);
			attitudeActual.Yaw = pios_can_yaw_message.fc_yaw;
			AttitudeActualSet(&attitudeActual);
		}

		if (PIOS_Queue_Receive(queue_altitude, &pios_can_alt_message, 0) == true) {
			BaroAltitudeAltitudeSet(&pios_can_alt_message.fc_alt);
		}

		if (PIOS_Queue_Receive(queue_flightstatus, &pios_can_flightstatus_message, 0) == true) {
			FlightStatusData flightStatus;
			FlightStatusGet(&flightStatus);
			flightStatus.FlightMode = pios_can_flightstatus_message.flight_mode;
			flightStatus.Armed = pios_can_flightstatus_message.armed;
			FlightStatusSet(&flightStatus);
		}

		PIOS_Thread_Sleep(1);
	}

}



/**
 * @}
 * @}
 */
