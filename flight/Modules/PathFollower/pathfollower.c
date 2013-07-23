/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup PathFollowerModule Path Follower Module
 * @{ 
 *
 * @file       pathfollower.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @brief      This module compared @ref PositionActuatl to @ref ActiveWaypoint 
 * and sets @ref AttitudeDesired.  It only does this when the FlightMode field
 * of @ref ManualControlCommand is Auto.
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

/**
 * Input object: @ref PathDesired and @ref PositionActual
 * Output object: @ref StabilizationDesired
 *
 * Calculate the value of @ref StabilizationDesired to get on the desired path.
 */

#include "openpilot.h"

#include "airspeedactual.h"
#include "fixedwingairspeeds.h"
#include "fixedwingpathfollowersettingscc.h"
#include "flightstatus.h"
#include "pathdesired.h"
#include "systemsettings.h"

#include "fixedwingpathfollower.h"
#include "helicopterpathfollower.h"
#include "multirotorpathfollower.h"
#include "dubinscartpathfollower.h"

#include "coordinate_conversions.h"

// Private constants
#define MAX_QUEUE_SIZE 4
#define STACK_SIZE_BYTES 750
#define TASK_PRIORITY (tskIDLE_PRIORITY+2)
#define CRITICAL_ERROR_THRESHOLD_MS 5000	//Time in [ms] before an error becomes a critical error

enum pathFollowerTypes {
	FIXEDWING, MULTIROTOR, HELICOPTER, DUBINSCART, HOLONOMIC, NONHOLONOMIC,
	DISABLED
};

// Private variables
static xTaskHandle PathFollowerTaskHandle;
static uint8_t flightMode = FLIGHTSTATUS_FLIGHTMODE_MANUAL;
static bool followerEnabled = false;
bool flightStatusUpdate = false;
static uint8_t pathFollowerType;

// Private functions
static void PathFollowerTask(void *parameters);
static void FlightStatusUpdatedCb(UAVObjEvent * ev);

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t PathFollowerStart()
{
	// Start main task
	if (followerEnabled) {
		// Start main task
		xTaskCreate(PathFollowerTask, (signed char *)"PathFollower",
			    STACK_SIZE_BYTES / 4, NULL, TASK_PRIORITY,
			    &PathFollowerTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_PATHFOLLOWER,
			       PathFollowerTaskHandle);
	}

	return 0;
}

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t PathFollowerInitialize()
{
	HwSettingsInitialize();
	uint8_t optionalModules[HWSETTINGS_OPTIONALMODULES_NUMELEM];
	HwSettingsOptionalModulesGet(optionalModules);

	// Select algorithm based on vehicle type
	SystemSettingsInitialize();
	uint8_t systemSettingsAirframeType;
	SystemSettingsAirframeTypeGet(&systemSettingsAirframeType);
	switch(systemSettingsAirframeType) {
		case SYSTEMSETTINGS_AIRFRAMETYPE_FIXEDWING:
		case SYSTEMSETTINGS_AIRFRAMETYPE_FIXEDWINGELEVON:
		case SYSTEMSETTINGS_AIRFRAMETYPE_FIXEDWINGVTAIL:
			pathFollowerType = FIXEDWING;
			break;
		case SYSTEMSETTINGS_AIRFRAMETYPE_TRI:
		case SYSTEMSETTINGS_AIRFRAMETYPE_QUADX:
		case SYSTEMSETTINGS_AIRFRAMETYPE_QUADP:
		case SYSTEMSETTINGS_AIRFRAMETYPE_HEXA:
		case SYSTEMSETTINGS_AIRFRAMETYPE_HEXAX:
		case SYSTEMSETTINGS_AIRFRAMETYPE_HEXACOAX:
		case SYSTEMSETTINGS_AIRFRAMETYPE_OCTO:
		case SYSTEMSETTINGS_AIRFRAMETYPE_OCTOV:
		case SYSTEMSETTINGS_AIRFRAMETYPE_OCTOCOAXP:
		case SYSTEMSETTINGS_AIRFRAMETYPE_OCTOCOAXX:
			pathFollowerType = MULTIROTOR;
			break;
		case SYSTEMSETTINGS_AIRFRAMETYPE_GROUNDVEHICLECAR:
		case SYSTEMSETTINGS_AIRFRAMETYPE_GROUNDVEHICLEMOTORCYCLE:
			pathFollowerType = DUBINSCART;
			break;
		case SYSTEMSETTINGS_AIRFRAMETYPE_GROUNDVEHICLEDIFFERENTIAL:
			pathFollowerType = HOLONOMIC;
			break;
		case SYSTEMSETTINGS_AIRFRAMETYPE_HELICP:
			pathFollowerType = HELICOPTER;
			break;
		case SYSTEMSETTINGS_AIRFRAMETYPE_VTOL:
			pathFollowerType = HOLONOMIC;
			break;
		default:
			// Cannot activate, prevent system arming
			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER, SYSTEMALARMS_ALARM_CRITICAL);
			pathFollowerType = DISABLED;
			return -1;
			break;
	}

	if (optionalModules[HWSETTINGS_OPTIONALMODULES_FIXEDWINGPATHFOLLOWER] ==
	    HWSETTINGS_OPTIONALMODULES_ENABLED) {
		FixedWingPathFollowerSettingsCCInitialize();
		FixedWingAirspeedsInitialize();
		AirspeedActualInitialize();
		PathDesiredInitialize();

		// TODO: Index into array of functions
		switch (pathFollowerType) {
		case FIXEDWING:
			initializeFixedWingPathFollower();
			break;
		case MULTIROTOR:
//			initializeMultirotorPathFollower();
			break;
		case HELICOPTER:
//			initializeHelicopterPathFollower();
			break;
		case HOLONOMIC:
			break;
		case DUBINSCART:
//			initializeDubinsCartPathFollower();
			break;
		default:
			PIOS_DEBUG_Assert(0);
			return -1;
			break;
		}

		FlightStatusConnectCallback(FlightStatusUpdatedCb);

		followerEnabled = true;
		return 0;
	}

	return -1;
}

MODULE_INITCALL(PathFollowerInitialize, PathFollowerStart)

/**
 * Module thread, should not return.
 */
static void PathFollowerTask(void *parameters)
{
	portTickType lastUpdateTime;
	FixedWingPathFollowerSettingsCCData fixedwingpathfollowerSettings;

	// Main task loop
	lastUpdateTime = xTaskGetTickCount();
	while (1) {
		// TODO: Refactor this into the fixed wing method as a callback
		FixedWingPathFollowerSettingsCCGet(&fixedwingpathfollowerSettings);

		vTaskDelayUntil(&lastUpdateTime, MS2TICKS(fixedwingpathfollowerSettings.UpdatePeriod));

		if (flightStatusUpdate)
			FlightStatusFlightModeGet(&flightMode);

		// Depending on vehicle type, call appropriate path follower
		// TODO: Index into array of methods
		switch (pathFollowerType) {
		case FIXEDWING:
			updateFixedWingDesiredStabilization(flightMode, fixedwingpathfollowerSettings);
			break;
//		case MULTIROTOR:
//			// Set alarm, currently untested
//			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER, SYSTEMALARMS_ALARM_ERROR);
//			updateMultirotorDesiredStabilization(flightMode, fixedwingpathfollowerSettings);
//			break;
//		case HELICOPTER:
//			// Unready
//			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER, SYSTEMALARMS_ALARM_CRITICAL);
//			//updateHelicopterDesiredStabilization(fixedwingpathfollowerSettings);
//		case HOLONOMIC:
//			// Unready
//			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER, SYSTEMALARMS_ALARM_CRITICAL);
//			break;
//		case DUBINSCART:
//			updateDubinsCartDesiredStabilization(flightMode, fixedwingpathfollowerSettings);
//			break;
		default:
			//Something has gone wrong, we shouldn't be able to get to this point
			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER, SYSTEMALARMS_ALARM_CRITICAL);
			break;
		}
	}
}

// Triggered by changes in FlightStatus
static void FlightStatusUpdatedCb(UAVObjEvent * ev)
{
	flightStatusUpdate = true;
}

/**
 * @}
 * @}
 */
