/**
 ******************************************************************************
 * @addtogroup Modules Tau Labs Modules
 * @{
 * @addtogroup ControlModule Control Module
 * @brief Process the control sources and select the appropriate one.
 * @{
 *
 * @file       control.c
 * @author     Tau Labs, http://github.com/TauLabs Copyright (C) 2013.
 * @brief      ManualControl module. Handles safety R/C link and flight mode.
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

#include "control.h"
#include "failsafe_control.h"
#include "tablet_control.h"
#include "transmitter_control.h"

#include "flightstatus.h"
#include "systemalarms.h"

// Private constants
#if defined(PIOS_CONTROL_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_MANUAL_STACK_SIZE
#else
#define STACK_SIZE_BYTES 1424
#endif

#define TASK_PRIORITY (tskIDLE_PRIORITY+4)
#define UPDATE_PERIOD_MS 20
#define THROTTLE_FAILSAFE -0.1f

// Private variables
static xTaskHandle taskHandle;
static portTickType lastSysTime;

// Private functions
static void controlTask(void *parameters);
static bool ok_to_arm(void);

// Private functions for control events
static int32_t control_event_arm();
static int32_t control_event_arming();
static int32_t control_event_disarm();

/**
 * Module starting
 */
int32_t ControlStart()
{
	// Start main task
	xTaskCreate(controlTask, (signed char *)"Control", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_MANUALCONTROL, taskHandle);
	PIOS_WDG_RegisterFlag(PIOS_WDG_MANUAL);

	return 0;
}

/**
 * Module initialization
 */
int32_t ControlInitialize()
{
	failsafe_control_initialize();
	transmitter_control_initialize();
	tablet_control_initialize();


	return 0;
}

MODULE_INITCALL(ControlInitialize, ControlStart);

/**
 * Module task
 */
static void controlTask(void *parameters)
{
	/* Make sure disarmed on power up */
	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);
	flightStatus.Armed = FLIGHTSTATUS_ARMED_DISARMED;

	// Main task loop
	lastSysTime = xTaskGetTickCount();

	// Select failsafe before run
	failsafe_control_select(true);

	while (1) {

		// Process periodic data for each of the controllers, including reading
		// all available inputs
		failsafe_control_update();
		transmitter_control_update();
		tablet_control_update();

		static enum control_selection last_control_selection = CONTROL_SELECTION_FAILSAFE;
		enum control_events control_events = CONTROL_EVENTS_NONE;

		// Control logic to select the valid controller
		enum control_selection control_selection = transmitter_control_selected_controller();
		bool reset_controller = control_selection != last_control_selection;

		switch(control_selection) {
		case CONTROL_SELECTION_FAILSAFE:
			failsafe_control_select(reset_controller);
			control_events = failsafe_control_get_events();
			break;
		case CONTROL_SELECTION_TRANSMITTER:
			transmitter_control_select(reset_controller);
			control_events = transmitter_control_get_events();
			break;
		case CONTROL_SELECTION_TABLET:
			if (tablet_control_select(reset_controller) == 0) {
				control_events = tablet_control_get_events();
			} else {
				// Failure in tablet control.  This would be better if done
				// at the selection stage before the tablet is even used.
				failsafe_control_select(false);
				control_events = failsafe_control_get_events();
			}
			break;
		}
		last_control_selection = control_selection;

		// TODO: This can evolve into a full FSM like I2C possibly
		switch(control_events) {
		case CONTROL_EVENTS_NONE:
			break;
		case CONTROL_EVENTS_ARM:
			control_event_arm();
			break;
		case CONTROL_EVENTS_ARMING:
			control_event_arming();
			break;
		case CONTROL_EVENTS_DISARM:
			control_event_disarm();
			break;
		}

		// Wait until next update
		vTaskDelayUntil(&lastSysTime, UPDATE_PERIOD_MS / portTICK_RATE_MS);
		PIOS_WDG_UpdateFlag(PIOS_WDG_MANUAL);
	}
}

//! When the control system requests to arm the FC
static int32_t control_event_arm()
{
	if(ok_to_arm()) {
		FlightStatusData flightStatus;
		FlightStatusGet(&flightStatus);
		if (flightStatus.Armed != FLIGHTSTATUS_ARMED_ARMED) {
			flightStatus.Armed = FLIGHTSTATUS_ARMED_ARMED;
			FlightStatusSet(&flightStatus);
		}
	}
	return 0;
}

//! When the control system requests to start arming the FC
static int32_t control_event_arming()
{
	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);
	if (flightStatus.Armed != FLIGHTSTATUS_ARMED_ARMING) {
		flightStatus.Armed = FLIGHTSTATUS_ARMED_ARMING;
		FlightStatusSet(&flightStatus);
	}
	return 0;
}

//! When the control system requests to disarm the FC
static int32_t control_event_disarm()
{
	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);
	if (flightStatus.Armed != FLIGHTSTATUS_ARMED_DISARMED) {
		flightStatus.Armed = FLIGHTSTATUS_ARMED_DISARMED;
		FlightStatusSet(&flightStatus);
	}
	return 0;
}


/**
 * @brief Determine if the aircraft is safe to arm based on alarms
 * @returns True if safe to arm, false otherwise
 */
static bool ok_to_arm(void)
{
	// read alarms
	SystemAlarmsData alarms;
	SystemAlarmsGet(&alarms);

	// Check each alarm
	for (int i = 0; i < SYSTEMALARMS_ALARM_NUMELEM; i++)
	{
		if (alarms.Alarm[i] >= SYSTEMALARMS_ALARM_ERROR &&
			i != SYSTEMALARMS_ALARM_GPS &&
			i != SYSTEMALARMS_ALARM_TELEMETRY)
		{
			return false;
		}
	}

	return true;
}

/**
  * @}
  * @}
  */
