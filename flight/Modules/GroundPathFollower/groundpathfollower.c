/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup GroundPathFollower Path follower for ground based vehicles
 * @{
 *
 * @file       groundpathfollower.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      Perform the path segment requested by @ref PathDesired

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
 * Input object: FlightStatus
 * Input object: PathDesired
 * Input object: PositionActual
 * Output object: StabilizationDesired
 *
 * This module will periodically update the value of the @ref StabilizationDesired object based on
 * @ref PathDesired and @ref PositionActual when the Flight Mode selected in @ref FlightStatus is supported
 * by this module.  Otherwise another module (e.g. @ref ManualControlCommand) is expected to be
 * writing to @ref StabilizationDesired.
 */

#include "openpilot.h"
#include "physical_constants.h"
#include "misc_math.h"
#include "paths.h"
#include "pid.h"

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
#include "groundpathfollowersettings.h"
#include "coordinate_conversions.h"
#include "pios_thread.h"

// Private constants
#define MAX_QUEUE_SIZE 4
#define STACK_SIZE_BYTES 1548
#define TASK_PRIORITY PIOS_THREAD_PRIO_NORMAL

// Private types

// Private variables
static struct pios_thread *pathfollowerTaskHandle;
static PathDesiredData pathDesired;
static GroundPathFollowerSettingsData guidanceSettings;

// Private functions
static void groundPathFollowerTask(void *parameters);
static void SettingsUpdatedCb(UAVObjEvent * ev);
static void updateNedAccel();
static void updatePathVelocity();
static void updateEndpointVelocity();
static void updateGroundDesiredAttitude();
static bool module_enabled = false;

enum ground_pid {VELOCITY, NORTH_POSITION, EAST_POSITION, GROUND_PID_NUM};
static struct pid ground_pids[GROUND_PID_NUM];

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t GroundPathFollowerStart()
{
	if (module_enabled) {
		// Start main task
		pathfollowerTaskHandle = PIOS_Thread_Create(groundPathFollowerTask, "GroundPathFollower", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
		TaskMonitorAdd(TASKINFO_RUNNING_PATHFOLLOWER, pathfollowerTaskHandle);
	}

	return 0;
}

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t GroundPathFollowerInitialize()
{
#ifdef MODULE_GroundPathFollower_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_GROUNDPATHFOLLOWER] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (!module_enabled) {
		return -1;
	}

	GroundPathFollowerSettingsInitialize();
	PathStatusInitialize();
	NedAccelInitialize();
	PathDesiredInitialize();
	VelocityDesiredInitialize();

	return 0;
}

MODULE_INITCALL(GroundPathFollowerInitialize, GroundPathFollowerStart)

static float northVelIntegral = 0;
static float eastVelIntegral = 0;

static float northPosIntegral = 0;
static float eastPosIntegral = 0;

static float throttleOffset = 0;
/**
 * Module thread, should not return.
 */
static void groundPathFollowerTask(void *parameters)
{
	SystemSettingsData systemSettings;
	FlightStatusData flightStatus;

	uint32_t lastUpdateTime;

	GroundPathFollowerSettingsConnectCallback(SettingsUpdatedCb);
	PathDesiredConnectCallback(SettingsUpdatedCb);

	GroundPathFollowerSettingsGet(&guidanceSettings);
	PathDesiredGet(&pathDesired);

	// Main task loop
	lastUpdateTime = PIOS_Thread_Systime();
	while (1) {

		// Conditions when this runs:
		// 1. Must have GROUND type airframe
		// 2. Flight mode is PositionHold and PathDesired.Mode is Endpoint  OR
		//    FlightMode is PathPlanner and PathDesired.Mode is Endpoint or Path

		SystemSettingsGet(&systemSettings);
		if ( (systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_GROUNDVEHICLECAR) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_GROUNDVEHICLEDIFFERENTIAL) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_GROUNDVEHICLEMOTORCYCLE) )
		{
			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER,SYSTEMALARMS_ALARM_WARNING);
			PIOS_Thread_Sleep(1000);
			continue;
		}

		// Continue collecting data if not enough time
		PIOS_Thread_Sleep_Until(&lastUpdateTime, guidanceSettings.UpdatePeriod);

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
					updateGroundDesiredAttitude();
				} else {
					AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER,SYSTEMALARMS_ALARM_ERROR);
				}
				break;
			case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
				if (pathDesired.Mode == PATHDESIRED_MODE_HOLDPOSITION) {
					updateEndpointVelocity();
					updateGroundDesiredAttitude();
				} else {
					AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER,SYSTEMALARMS_ALARM_ERROR);
				}
				break;
			case FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER:
				if (pathDesired.Mode == PATHDESIRED_MODE_FLYENDPOINT ||
					pathDesired.Mode == PATHDESIRED_MODE_HOLDPOSITION) {
					updateEndpointVelocity();
					updateGroundDesiredAttitude();
				} else if (pathDesired.Mode == PATHDESIRED_MODE_FLYVECTOR ||
					pathDesired.Mode == PATHDESIRED_MODE_FLYCIRCLELEFT ||
					pathDesired.Mode == PATHDESIRED_MODE_FLYCIRCLERIGHT) {
					updatePathVelocity();
					updateGroundDesiredAttitude();
				} else {
					AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER,SYSTEMALARMS_ALARM_ERROR);
				}
				break;
			default:
				// Be cleaner and get rid of global variables
				northVelIntegral = 0;
				eastVelIntegral = 0;
				northPosIntegral = 0;
				eastPosIntegral = 0;

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
 * Compute desired velocity from the current position and path
 *
 * Takes in @ref PositionActual and compares it to @ref PathDesired
 * and computes @ref VelocityDesired
 */
static void updatePathVelocity()
{
	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	float cur[3] = {positionActual.North, positionActual.East, positionActual.Down};
	struct path_status progress;

	path_progress(&pathDesired, cur, &progress);

	// Update the path status UAVO
	PathStatusData pathStatus;
	PathStatusGet(&pathStatus);
	pathStatus.fractional_progress = progress.fractional_progress;
	if (pathStatus.fractional_progress < 1)
		pathStatus.Status = PATHSTATUS_STATUS_INPROGRESS;
	else
		pathStatus.Status = PATHSTATUS_STATUS_COMPLETED;

	pathStatus.Waypoint = pathDesired.Waypoint;

	PathStatusSet(&pathStatus);

	float groundspeed = pathDesired.StartingVelocity +
	    (pathDesired.EndingVelocity - pathDesired.StartingVelocity) * progress.fractional_progress;
	if(progress.fractional_progress > 1)
		groundspeed = 0;

	VelocityDesiredData velocityDesired;
	velocityDesired.North = progress.path_direction[0] * groundspeed;
	velocityDesired.East = progress.path_direction[1] * groundspeed;

	float error_speed = progress.error * guidanceSettings.HorizontalPosPI[GROUNDPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KP];
	float correction_velocity[2] = {progress.correction_direction[0] * error_speed,
	    progress.correction_direction[1] * error_speed};

	float total_vel = sqrtf(powf(correction_velocity[0],2) + powf(correction_velocity[1],2));
	float scale = 1;
	if(total_vel > guidanceSettings.HorizontalVelMax)
		scale = guidanceSettings.HorizontalVelMax / total_vel;

	// Currently not apply a PID loop for horizontal corrections
	velocityDesired.North += progress.correction_direction[0] * error_speed * scale;
	velocityDesired.East += progress.correction_direction[1] * error_speed * scale;

	// No altitude control on a ground vehicle
	velocityDesired.Down = 0;

	VelocityDesiredSet(&velocityDesired);
}

/**
 * Compute desired velocity from the current position
 *
 * Takes in @ref PositionActual and compares it to @ref PositionDesired
 * and computes @ref VelocityDesired
 */
void updateEndpointVelocity()
{
	float dT = guidanceSettings.UpdatePeriod / 1000.0f;

	PositionActualData positionActual;
	VelocityDesiredData velocityDesired;

	PositionActualGet(&positionActual);
	VelocityDesiredGet(&velocityDesired);

	float northError;
	float eastError;
	float northCommand;
	float eastCommand;

	float northPos = positionActual.North;
	float eastPos = positionActual.East;

	// Compute desired north command velocity from position error
	northError = pathDesired.End[PATHDESIRED_END_NORTH] - northPos;
	northCommand = pid_apply(&ground_pids[NORTH_POSITION], northError, dT);

	// Compute desired east command velocity from position error
	eastError = pathDesired.End[PATHDESIRED_END_EAST] - eastPos;
	eastCommand = pid_apply(&ground_pids[EAST_POSITION], eastError, dT);

	// Limit the maximum velocity any direction (not north and east separately)
	float total_vel = sqrtf(powf(northCommand,2) + powf(eastCommand,2));
	float scale = 1;
	if(total_vel > guidanceSettings.HorizontalVelMax)
		scale = guidanceSettings.HorizontalVelMax / total_vel;

	velocityDesired.North = northCommand * scale;
	velocityDesired.East = eastCommand * scale;

	// No altitude control on a ground vehicle
	velocityDesired.Down = 0;

	VelocityDesiredSet(&velocityDesired);

	// Indicate whether we are in radius of this endpoint
	uint8_t path_status = PATHSTATUS_STATUS_INPROGRESS;
	float distance2 = powf(northError, 2) + powf(eastError, 2);
	if (distance2 < (guidanceSettings.EndpointRadius * guidanceSettings.EndpointRadius)) {
		path_status = PATHSTATUS_STATUS_COMPLETED;
	}
	PathStatusStatusSet(&path_status);
}

/**
 * Compute desired attitude from the desired velocity
 *
 * Takes in @ref NedActual which has the acceleration in the
 * NED frame as the feedback term and then compares the
 * @ref VelocityActual against the @ref VelocityDesired
 */
static void updateGroundDesiredAttitude()
{
	float dT = guidanceSettings.UpdatePeriod / 1000.0f;

	VelocityDesiredData velocityDesired;
	VelocityActualData velocityActual;
	StabilizationDesiredData stabDesired;
	AttitudeActualData attitudeActual;
	NedAccelData nedAccel;
	GroundPathFollowerSettingsData guidanceSettings;
	StabilizationSettingsData stabSettings;
	SystemSettingsData systemSettings;

	SystemSettingsGet(&systemSettings);
	GroundPathFollowerSettingsGet(&guidanceSettings);
	VelocityActualGet(&velocityActual);
	VelocityDesiredGet(&velocityDesired);
	StabilizationDesiredGet(&stabDesired);
	VelocityDesiredGet(&velocityDesired);
	AttitudeActualGet(&attitudeActual);
	StabilizationSettingsGet(&stabSettings);
	NedAccelGet(&nedAccel);

	float northVel = velocityActual.North;
	float eastVel = velocityActual.East;

	// Calculate direction from velocityDesired and set stabDesired.Yaw
	stabDesired.Yaw = atan2f( velocityDesired.East, velocityDesired.North ) * RAD2DEG;

	// Calculate throttle and set stabDesired.Throttle
	float velDesired = sqrtf(powf(velocityDesired.East,2) + powf(velocityDesired.North,2));
	float velActual = sqrtf(powf(eastVel,2) + powf(northVel,2));
	ManualControlCommandData manualControlData;
	ManualControlCommandGet(&manualControlData);
	switch (guidanceSettings.ThrottleControl) {
		case GROUNDPATHFOLLOWERSETTINGS_THROTTLECONTROL_MANUAL:
		{
			stabDesired.Throttle = manualControlData.Throttle;
			break;
		}
		case GROUNDPATHFOLLOWERSETTINGS_THROTTLECONTROL_PROPORTIONAL:
		{
			float velRatio = velDesired / guidanceSettings.HorizontalVelMax;
			stabDesired.Throttle = guidanceSettings.MaxThrottle * velRatio;
			if (guidanceSettings.ManualOverride == GROUNDPATHFOLLOWERSETTINGS_MANUALOVERRIDE_TRUE) {
				stabDesired.Throttle = stabDesired.Throttle * manualControlData.Throttle;
			}
			break;
		}
		case GROUNDPATHFOLLOWERSETTINGS_THROTTLECONTROL_AUTO:
		{
			float velError = velDesired - velActual;
			stabDesired.Throttle = pid_apply(&ground_pids[VELOCITY], velError, dT) + velDesired * guidanceSettings.VelocityFeedforward;
			if (guidanceSettings.ManualOverride == GROUNDPATHFOLLOWERSETTINGS_MANUALOVERRIDE_TRUE) {
				stabDesired.Throttle = stabDesired.Throttle * manualControlData.Throttle;
			}
			break;
		}
		default:
		{
			PIOS_Assert(0);
			break;
		}
	}

	// Limit throttle as per settings
	stabDesired.Throttle = bound_min_max(stabDesired.Throttle, 0, guidanceSettings.MaxThrottle);

	// Set StabilizationDesired object
	stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	StabilizationDesiredSet(&stabDesired);
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

static void SettingsUpdatedCb(UAVObjEvent * ev)
{
	GroundPathFollowerSettingsGet(&guidanceSettings);

	// Configure the velocity control PID loops
	pid_configure(&ground_pids[VELOCITY],
		guidanceSettings.HorizontalVelPID[GROUNDPATHFOLLOWERSETTINGS_HORIZONTALVELPID_KP], // Kp
		guidanceSettings.HorizontalVelPID[GROUNDPATHFOLLOWERSETTINGS_HORIZONTALVELPID_KI], // Ki
		0, // Kd
		guidanceSettings.HorizontalVelPID[GROUNDPATHFOLLOWERSETTINGS_HORIZONTALVELPID_ILIMIT]);

	// Configure the position control (velocity output) PID loops
	pid_configure(&ground_pids[NORTH_POSITION],
		guidanceSettings.HorizontalPosPI[GROUNDPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KP],
		guidanceSettings.HorizontalPosPI[GROUNDPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KI],
		0,
		guidanceSettings.HorizontalPosPI[GROUNDPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_ILIMIT]);
	pid_configure(&ground_pids[EAST_POSITION],
		guidanceSettings.HorizontalPosPI[GROUNDPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KP],
		guidanceSettings.HorizontalPosPI[GROUNDPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KI],
		0,
		guidanceSettings.HorizontalPosPI[GROUNDPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_ILIMIT]);

	PathDesiredGet(&pathDesired);
}

/**
 * @}
 * @}
 */

