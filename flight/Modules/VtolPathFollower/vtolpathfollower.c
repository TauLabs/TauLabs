/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup VtolPathFollower VTOL path follower module
 * @{
 *
 * @file       vtolpathfollower.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Compute attitude to achieve a path for VTOL aircrafts
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
 * Input object: @ref FlightStatus
 * Input object: @ref PathDesired
 * Input object: @ref PositionActual
 * Input object: @ref VelocityActual
 * Output object: @ref StabilizationDesired
 *
 * This module will periodically update the value of the @ref StabilizationDesired object based on 
 * @ref PathDesired and @PositionActual when the Flight Mode selected in @FlightStatus is supported
 * by this module.  Otherwise another module (e.g. @ref ManualControlCommand) is expected to be
 * writing to @ref StabilizationDesired.
 */

#include "openpilot.h"
#include "physical_constants.h"
#include "misc_math.h"
#include "paths.h"
#include "pid.h"

#include "vtol_follower_priv.h"

#include "accels.h"
#include "attitudeactual.h"
#include "modulesettings.h"
#include "pathdesired.h"        // object that will be updated by the module
#include "positionactual.h"
#include "manualcontrolcommand.h"
#include "flightstatus.h"
#include "gpsvelocity.h"
#include "gpsposition.h"
#include "nedaccel.h"
#include "nedposition.h"
#include "pathstatus.h"
#include "stabilizationdesired.h"
#include "stabilizationsettings.h"
#include "systemsettings.h"
#include "velocitydesired.h"
#include "velocityactual.h"
#include "vtolpathfollowersettings.h"
#include "coordinate_conversions.h"

// Private constants
#define MAX_QUEUE_SIZE 4
#define STACK_SIZE_BYTES 1548
#define TASK_PRIORITY (tskIDLE_PRIORITY+2)

// Private types

// Private variables
static xTaskHandle pathfollowerTaskHandle;
static PathDesiredData pathDesired;
static VtolPathFollowerSettingsData guidanceSettings;

// Private functions
static void vtolPathFollowerTask(void *parameters);
static void updateNedAccel();
static void updatePathVelocity();
static void updateEndpointVelocity();
static void updateVtolDesiredAttitude();
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

	VtolPathFollowerSettingsInitialize();
	PathStatusInitialize();
	NedAccelInitialize();
	PathDesiredInitialize();
	VelocityDesiredInitialize();
	
	return 0;
}

MODULE_INITCALL(VtolPathFollowerInitialize, VtolPathFollowerStart)


static float throttleOffset = 0;
/**
 * Module thread, should not return.
 */
static void vtolPathFollowerTask(void *parameters)
{
	SystemSettingsData systemSettings;
	FlightStatusData flightStatus;

	portTickType lastUpdateTime;
	
	VtolPathFollowerSettingsConnectCallback(vtol_follower_control_settings_updated);
	PathDesiredConnectCallback(vtol_follower_control_settings_updated);
	
	VtolPathFollowerSettingsGet(&guidanceSettings);
	PathDesiredGet(&pathDesired);
	
	// Main task loop
	lastUpdateTime = xTaskGetTickCount();
	while (1) {

		// Conditions when this runs:
		// 1. Must have VTOL type airframe
		// 2. Flight mode is PositionHold and PathDesired.Mode is Endpoint  OR
		//    FlightMode is PathPlanner and PathDesired.Mode is Endpoint or Path

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
			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER,SYSTEMALARMS_ALARM_WARNING);
			vTaskDelay(1000);
			continue;
		}

		// Continue collecting data if not enough time
		vTaskDelayUntil(&lastUpdateTime, MS2TICKS(guidanceSettings.UpdatePeriod));

		// Convert the accels into the NED frame
		updateNedAccel();
		
		FlightStatusGet(&flightStatus);

		// Check the combinations of flightmode and pathdesired mode
		switch(flightStatus.FlightMode) {
			/* This combination of RETURNTOHOME and HOLDPOSITION looks strange but
			 * is correct.  RETURNTOHOME mode uses HOLDPOSITION with the position
			 * set to home */
			case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:
				if (pathDesired.Mode == PATHDESIRED_MODE_HOLDPOSITION) {
					updateEndpointVelocity();
					updateVtolDesiredAttitude();
				} else {
					AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER,SYSTEMALARMS_ALARM_ERROR);
				}
				break;
			case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
				if (pathDesired.Mode == PATHDESIRED_MODE_HOLDPOSITION) {
					updateEndpointVelocity();
					updateVtolDesiredAttitude();
				} else {
					AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER,SYSTEMALARMS_ALARM_ERROR);
				}
				break;
			case FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER:
				if (pathDesired.Mode == PATHDESIRED_MODE_FLYENDPOINT ||
					pathDesired.Mode == PATHDESIRED_MODE_HOLDPOSITION) {
					updateEndpointVelocity();
					updateVtolDesiredAttitude();
				} else if (pathDesired.Mode == PATHDESIRED_MODE_FLYVECTOR ||
					pathDesired.Mode == PATHDESIRED_MODE_FLYCIRCLELEFT ||
					pathDesired.Mode == PATHDESIRED_MODE_FLYCIRCLERIGHT) {
					updatePathVelocity();
					updateVtolDesiredAttitude();
				} else {
					AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER,SYSTEMALARMS_ALARM_ERROR);
				}
				break;
			default:

				for (uint32_t i = 0; i < VTOL_PID_NUM; i++)
					pid_zero(&vtol_pids[i]);

				// Track throttle before engaging this mode.  Cheap system ident
				StabilizationDesiredData stabDesired;
				StabilizationDesiredGet(&stabDesired);
				throttleOffset = stabDesired.Throttle;

				break;
		}

		AlarmsClear(SYSTEMALARMS_ALARM_PATHFOLLOWER);

	}
}

/**
 * Keep a running filtered version of the acceleration in the NED frame
 */
static void updateNedAccel()
{
	float accel[3];
	float q[4];
	float Rbe[3][3];
	float accel_ned[3];

	// Collect downsampled attitude data
	AccelsData accels;
	AccelsGet(&accels);		
	accel[0] = accels.x;
	accel[1] = accels.y;
	accel[2] = accels.z;
	
	//rotate avg accels into earth frame and store it
	AttitudeActualData attitudeActual;
	AttitudeActualGet(&attitudeActual);
	q[0]=attitudeActual.q1;
	q[1]=attitudeActual.q2;
	q[2]=attitudeActual.q3;
	q[3]=attitudeActual.q4;
	Quaternion2R(q, Rbe);
	for (uint8_t i=0; i<3; i++){
		accel_ned[i]=0;
		for (uint8_t j=0; j<3; j++)
			accel_ned[i] += Rbe[j][i]*accel[j];
	}
	accel_ned[2] += GRAVITY;
	
	NedAccelData accelData;
	NedAccelGet(&accelData);
	accelData.North = accel_ned[0];
	accelData.East = accel_ned[1];
	accelData.Down = accel_ned[2];
	NedAccelSet(&accelData);
}

/**
 * @}
 * @}
 */

