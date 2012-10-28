/**
 ******************************************************************************
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
 * Input object: ???
 * Output object: AttitudeDesired
 *
 * This module will periodically update the value of the AttitudeDesired object.
 *
 * The module executes in its own thread in this example.
 *
 * Modules have no API, all communication to other modules is done through UAVObjects.
 * However modules may use the API exposed by shared libraries.
 * See the OpenPilot wiki for more details.
 * http://www.openpilot.org/OpenPilot_Application_Architecture
 *
 */

#include "openpilot.h"

#include "hwsettings.h"
#include "attitudeactual.h"
#include "positionactual.h"
#include "velocityactual.h"
#include "manualcontrol.h"
#include "flightstatus.h"
#include "airspeedactual.h"
#include "homelocation.h"
#include "stabilizationdesired.h"	// object that will be updated by the module
#include "pathdesired.h"	// object that will be updated by the module
#include "systemsettings.h"
#include "fixedwingpathfollowersettings.h"

#include "fixedwingpathfollower.h"
#include "helicopterpathfollower.h"
#include "multirotorpathfollower.h"
#include "dubinscartpathfollower.h"

#include "CoordinateConversions.h"

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
//static void FixedWingPathFollowerParamsUpdatedCb(UAVObjEvent * ev);
static void FlightStatusUpdatedCb(UAVObjEvent * ev);
//static void updateSteadyStateAttitude();

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

	// Conditions when this runs:
	// 1. ???

	//Test for vehicle type
	SystemSettingsInitialize();
	uint8_t systemSettingsAirframeType;
	SystemSettingsAirframeTypeGet(&systemSettingsAirframeType);
	if ((systemSettingsAirframeType ==
	     SYSTEMSETTINGS_AIRFRAMETYPE_FIXEDWING)
	    || (systemSettingsAirframeType ==
		SYSTEMSETTINGS_AIRFRAMETYPE_FIXEDWINGELEVON)
	    || (systemSettingsAirframeType ==
		SYSTEMSETTINGS_AIRFRAMETYPE_FIXEDWINGVTAIL)) {
		pathFollowerType = FIXEDWING;
	} else
	    if ((systemSettingsAirframeType == SYSTEMSETTINGS_AIRFRAMETYPE_TRI)
		|| (systemSettingsAirframeType ==
		    SYSTEMSETTINGS_AIRFRAMETYPE_QUADX)
		|| (systemSettingsAirframeType ==
		    SYSTEMSETTINGS_AIRFRAMETYPE_QUADP)
		|| (systemSettingsAirframeType ==
		    SYSTEMSETTINGS_AIRFRAMETYPE_HEXA)
		|| (systemSettingsAirframeType ==
		    SYSTEMSETTINGS_AIRFRAMETYPE_HEXAX)
		|| (systemSettingsAirframeType ==
		    SYSTEMSETTINGS_AIRFRAMETYPE_HEXACOAX)
		|| (systemSettingsAirframeType ==
		    SYSTEMSETTINGS_AIRFRAMETYPE_OCTO)
		|| (systemSettingsAirframeType ==
		    SYSTEMSETTINGS_AIRFRAMETYPE_OCTOV)
		|| (systemSettingsAirframeType ==
		    SYSTEMSETTINGS_AIRFRAMETYPE_OCTOCOAXP)
		|| (systemSettingsAirframeType ==
		    SYSTEMSETTINGS_AIRFRAMETYPE_OCTOCOAXX)) {
		pathFollowerType = MULTIROTOR;
	} else
	    if ((systemSettingsAirframeType ==
		 SYSTEMSETTINGS_AIRFRAMETYPE_GROUNDVEHICLECAR)
		|| (systemSettingsAirframeType ==
		    SYSTEMSETTINGS_AIRFRAMETYPE_GROUNDVEHICLEMOTORCYCLE)) {
		pathFollowerType = DUBINSCART;
	} else
	    if ((systemSettingsAirframeType ==
		 SYSTEMSETTINGS_AIRFRAMETYPE_GROUNDVEHICLEDIFFERENTIAL)) {
		pathFollowerType = HOLONOMIC;
	} else
	    if ((systemSettingsAirframeType ==
		 SYSTEMSETTINGS_AIRFRAMETYPE_HELICP)) {
		pathFollowerType = HELICOPTER;
	} else
	    if ((systemSettingsAirframeType ==
		 SYSTEMSETTINGS_AIRFRAMETYPE_VTOL)) {
		pathFollowerType = HOLONOMIC;
	} else {		//WHAT ABOUT CUSTOM MIXERS?
		pathFollowerType = DISABLED;
		return -1;	//HUH??? RETURNING -1 STILL LEADS TO THE MODULE BEING ACTIVATED
	}

	if (optionalModules[HWSETTINGS_OPTIONALMODULES_FIXEDWINGPATHFOLLOWER] ==
	    HWSETTINGS_OPTIONALMODULES_ENABLED) {
		FixedWingPathFollowerSettingsInitialize();
		AirspeedActualInitialize();
		PathDesiredInitialize();

		//VVVVVVVVVVVVVVV
//              pathFollowerTypeInitialize[pathFollowerType]; <-- THIS NEEDS TO
//              BE DONE LIKE THIS, WITH A VIRTUAL FUNCTION INSTEAD OF A SWITCH
		switch (pathFollowerType) {
		case FIXEDWING:
			initializeFixedWingPathFollower();
			break;
		case MULTIROTOR:
			initializeMultirotorPathFollower();
		case HELICOPTER:
			initializeHelicopterPathFollower();
		case HOLONOMIC:
			break;
		case DUBINSCART:
			initializeDubinsCartPathFollower();
			break;
		default:
			//Something has gone wrong, we shouldn't be able to get to this point
			AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,
				  SYSTEMALARMS_ALARM_CRITICAL);
			break;
		}
		//^^^^^^^^^^^^

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
	FixedWingPathFollowerSettingsData fixedwingpathfollowerSettings;

//      FixedWingPathFollowerSettingsConnectCallback(FixedWingPathFollowerParamsUpdatedCb);
//      PathDesiredConnectCallback(FixedWingPathFollowerParamsUpdatedCb);

	// Main task loop
	lastUpdateTime = xTaskGetTickCount();
	while (1) {
		//IT WOULD BE NICE NOT TO DO THIS EVERY LOOP.
		FixedWingPathFollowerSettingsGet
		    (&fixedwingpathfollowerSettings);

		// Wait.
		vTaskDelayUntil(&lastUpdateTime,
				fixedwingpathfollowerSettings.UpdatePeriod /
				portTICK_RATE_MS);

		// Check flightmode
		if (flightStatusUpdate) {
			FlightStatusFlightModeGet(&flightMode);
		}
		//Depending on vehicle type, call appropriate path follower
		switch (pathFollowerType) {
		case FIXEDWING:
			updateFixedWingDesiredStabilization(flightMode,
							    fixedwingpathfollowerSettings);
			break;
		case MULTIROTOR:
			// @todo: is this really done? Added Alarmset just in case
			AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,
				  SYSTEMALARMS_ALARM_CRITICAL);
			updateMultirotorDesiredStabilization(flightMode,
							     fixedwingpathfollowerSettings);
			break;
		case HELICOPTER:
			// Helicopter mode is very far from being ready
			AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,
				  SYSTEMALARMS_ALARM_CRITICAL);
//          updateHelicopterDesiredStabilization(fixedwingpathfollowerSettings);
		case HOLONOMIC:
			// Holomomic mode is very far from being ready, and might never
			// even be used
			AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,
				  SYSTEMALARMS_ALARM_CRITICAL);
			break;
		case DUBINSCART:
			updateDubinsCartDesiredStabilization(flightMode,
							     fixedwingpathfollowerSettings);
			break;
		default:
			//Something has gone wrong, we shouldn't be able to get to this point
			AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,
				  SYSTEMALARMS_ALARM_CRITICAL);
			break;
		}
	}
}

//Triggered by changes in FlightStatus
static void FlightStatusUpdatedCb(UAVObjEvent * ev)
{
	flightStatusUpdate = true;
}
