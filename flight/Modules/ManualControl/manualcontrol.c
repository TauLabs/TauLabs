/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Control Control Module
 * @{
 * @brief      Highest level control module which decides the control state
 *
 * This module users the values from the transmitter and its settings to
 * to determine if the system is currently controlled by failsafe, transmitter,
 * or a tablet.  The transmitter values are read out and stored in @ref
 * ManualControlCommand.  The tablet sends values via @ref TabletInfo which
 * may be used if the flight mode switch is in the appropriate position. The
 * transmitter settings come from @ref ManualControlSettings.
 *
 * @file       manualcontrol.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
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
#include "manualcontrolcommand.h"
#include "manualcontrolsettings.h"
#include "systemalarms.h"

// Private constants
#if defined(PIOS_MANUAL_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_MANUAL_STACK_SIZE
#else
#define STACK_SIZE_BYTES 1000
#endif

#define TASK_PRIORITY (tskIDLE_PRIORITY+4)
#define UPDATE_PERIOD_MS 20

// Private variables
static xTaskHandle taskHandle;
static portTickType lastSysTime;

// Private functions
static void manualControlTask(void *parameters);
static bool ok_to_arm(void);
static FlightStatusControlSourceOptions control_source_select();

// Private functions for control events
static int32_t control_event_arm();
static int32_t control_event_arming();
static int32_t control_event_disarm();

/**
 * Module starting
 */
int32_t ManualControlStart()
{
	// Start main task
	xTaskCreate(manualControlTask, (signed char *)"Control", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_MANUALCONTROL, taskHandle);
	PIOS_WDG_RegisterFlag(PIOS_WDG_MANUAL);

	return 0;
}

/**
 * Module initialization
 */
int32_t ManualControlInitialize()
{
	failsafe_control_initialize();
	transmitter_control_initialize();
	tablet_control_initialize();


	return 0;
}

MODULE_INITCALL(ManualControlInitialize, ManualControlStart);

/**
 * Module task
 */
static void manualControlTask(void *parameters)
{
	/* Make sure disarmed on power up */
	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);
	flightStatus.Armed = FLIGHTSTATUS_ARMED_DISARMED;
	FlightStatusSet(&flightStatus);

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

		// Initialize to invalid value to ensure first update sets FlightStatus
		static FlightStatusControlSourceOptions last_control_selection = -1;
		enum control_events control_events = CONTROL_EVENTS_NONE;

		// Control logic to select the valid controller
		FlightStatusControlSourceOptions control_selection = control_source_select();
		bool reset_controller = control_selection != last_control_selection;

		// This logic would be better collapsed into control_source_select but
		// in the case of the tablet we need to be able to abort and fall back
		// to failsafe for now
		switch(control_selection) {
		case FLIGHTSTATUS_CONTROLSOURCE_TRANSMITTER:
			transmitter_control_select(reset_controller);
			control_events = transmitter_control_get_events();
			break;
		case FLIGHTSTATUS_CONTROLSOURCE_TABLET:
		{
			static bool tablet_previously_succeeded = false;
			if (tablet_control_select(reset_controller) == 0) {
				control_events = tablet_control_get_events();
				tablet_previously_succeeded = true;
			} else {
				// Failure in tablet control.  This would be better if done
				// at the selection stage before the tablet is even used.
				failsafe_control_select(reset_controller || tablet_previously_succeeded);
				control_events = failsafe_control_get_events();
				tablet_previously_succeeded = false;
			}
			break;
		}
		case FLIGHTSTATUS_CONTROLSOURCE_FAILSAFE:
		default:
			failsafe_control_select(reset_controller);
			control_events = failsafe_control_get_events();
			break;
		}
		if (control_selection != last_control_selection) {
			FlightStatusControlSourceSet(&control_selection);
			last_control_selection = control_selection;
		}

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
		vTaskDelayUntil(&lastSysTime, MS2TICKS(UPDATE_PERIOD_MS));
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
 * @brief control_source_select Determine which sub-module to use
 * for the main control source of the flight controller.
 * @returns @ref FlightStatusControlSourceOptions indicating the selected
 * mode
 *
 * This function is the ultimate one that controls what happens and
 * selects modes such as failsafe, transmitter control, geofencing
 * and potentially other high level modes in the future
 */
static FlightStatusControlSourceOptions control_source_select()
{
	ManualControlCommandData cmd;
	ManualControlCommandGet(&cmd);
	if (cmd.Connected != MANUALCONTROLCOMMAND_CONNECTED_TRUE) {
		return FLIGHTSTATUS_CONTROLSOURCE_FAILSAFE;
	} else if (transmitter_control_get_flight_mode() ==
	           MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_TABLETCONTROL) {
		return FLIGHTSTATUS_CONTROLSOURCE_TABLET;
	} else {
		return FLIGHTSTATUS_CONTROLSOURCE_TRANSMITTER;
	}

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
