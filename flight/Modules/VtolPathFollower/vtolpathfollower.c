/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup VtolPathFollower VTOL path follower module
 * @{
 *
 * @file       vtolpathfollower.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      Compute attitude to achieve a path for VTOL aircrafts
 *
 * Runs the VTOL follower FSM which then calls the lower VTOL navigation
 * control algorithms as appropriate.
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
#include "physical_constants.h"
#include "misc_math.h"
#include "paths.h"
#include "pid.h"

#include "vtol_follower_priv.h"

#include "acceldesired.h"
#include "altitudeholdsettings.h"
#include "modulesettings.h"
#include "pathdesired.h"        // object that will be updated by the module
#include "flightstatus.h"
#include "pathstatus.h"
#include "stabilizationdesired.h"
#include "systemsettings.h"
#include "velocitydesired.h"
#include "vtolpathfollowersettings.h"
#include "vtolpathfollowerstatus.h"
#include "coordinate_conversions.h"

// Private constants
#define MAX_QUEUE_SIZE 4
#define STACK_SIZE_BYTES 1548
#define TASK_PRIORITY (tskIDLE_PRIORITY+2)

// Private types

// Private variables
static xTaskHandle pathfollowerTaskHandle;
static VtolPathFollowerSettingsData guidanceSettings;

// Private functions
static void vtolPathFollowerTask(void *parameters);
static bool module_enabled = false;

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t VtolPathFollowerStart()
{
	if (module_enabled) {
		// Start main task
		xTaskCreate(vtolPathFollowerTask, (signed char *)"VtolPathFollower", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &pathfollowerTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_PATHFOLLOWER, pathfollowerTaskHandle);
	}

	return 0;
}

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t VtolPathFollowerInitialize()
{
#ifdef MODULE_VtolPathFollower_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_VTOLPATHFOLLOWER] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (!module_enabled) {
		return -1;
	}

	AccelDesiredInitialize();
	AltitudeHoldSettingsInitialize();
	PathDesiredInitialize();
	PathStatusInitialize();
	VelocityDesiredInitialize();
	VtolPathFollowerSettingsInitialize();
	VtolPathFollowerStatusInitialize();
	
	return 0;
}

MODULE_INITCALL(VtolPathFollowerInitialize, VtolPathFollowerStart);

extern struct pid vtol_pids[VTOL_PID_NUM];

/**
 * Module thread, should not return.
 */
static void vtolPathFollowerTask(void *parameters)
{
	SystemSettingsData systemSettings;
	FlightStatusData flightStatus;

	portTickType lastUpdateTime;
	
	VtolPathFollowerSettingsConnectCallback(vtol_follower_control_settings_updated);
	AltitudeHoldSettingsConnectCallback(vtol_follower_control_settings_updated);
	vtol_follower_control_settings_updated(NULL);
	
	VtolPathFollowerSettingsGet(&guidanceSettings);
	
	// Main task loop
	lastUpdateTime = xTaskGetTickCount();
	while (1) {

		SystemSettingsGet(&systemSettings);
		if ( (systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_VTOL) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_QUADP) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_QUADX) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_HEXA) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_HEXAX) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_HEXACOAX) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_OCTO) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_OCTOV) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_OCTOCOAXP) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_OCTOCOAXX) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_TRI) )
		{
			// This should be a critical alarm since the system will not attempt to
			// control in this situation.
			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER,SYSTEMALARMS_ALARM_CRITICAL);
			vTaskDelay(1000);
			continue;
		}

		// Continue collecting data if not enough time
		vTaskDelayUntil(&lastUpdateTime, MS2TICKS(guidanceSettings.UpdatePeriod));
		
		static uint8_t last_flight_mode;
		FlightStatusGet(&flightStatus);

		static bool fsm_running = false;

		if (flightStatus.FlightMode != last_flight_mode) {
			// The mode has changed

			last_flight_mode = flightStatus.FlightMode;

			switch(flightStatus.FlightMode) {
			case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:
				vtol_follower_fsm_activate_goal(GOAL_LAND_HOME);
				fsm_running = true;
				break;
			case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
				vtol_follower_fsm_activate_goal(GOAL_HOLD_POSITION);
				fsm_running = true;
				break;
			case FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER:
				// TODO: currently when in this mode the follower just
				// attempts to fly the path segments blindly which means
				// the FSM cannot be utilized in a meaningful way. It might
				// be better when flying in path planner mode for the path
				// planner to specify the goals in PathDesired so things like
				// RTH can be used. However, for now this isn't critical.
				vtol_follower_fsm_activate_goal(GOAL_FLY_PATH);
				fsm_running = true;
				break;
			default:
				vtol_follower_fsm_activate_goal(GOAL_LAND_NONE);
				fsm_running = false;
				break;
			}
		}

		if (fsm_running) {
			vtol_follower_fsm_update();
		} else {
			for (uint32_t i = 0; i < VTOL_PID_NUM; i++)
				pid_zero(&vtol_pids[i]);
		
			// Track throttle before engaging this mode.  Cheap system ident
			StabilizationDesiredThrottleGet(&vtol_pids[DOWN_VELOCITY].iAccumulator);
			vtol_pids[DOWN_VELOCITY].iAccumulator *= 1000.0f; // pid library scales up accumulator by 1000
		}

		AlarmsClear(SYSTEMALARMS_ALARM_PATHFOLLOWER);

	}
}

/**
 * @}
 * @}
 */

