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
static void SettingsUpdatedCb(UAVObjEvent * ev);
static void updateNedAccel();
static void updatePathVelocity();
static void updateEndpointVelocity();
static void updateVtolDesiredAttitude();
static bool module_enabled = false;

enum vtol_pid {NORTH_VELOCITY, EAST_VELOCITY, DOWN_VELOCITY, NORTH_POSITION, EAST_POSITION, DOWN_POSITION, VTOL_PID_NUM};
static struct pid vtol_pids[VTOL_PID_NUM];

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
	
	VtolPathFollowerSettingsConnectCallback(SettingsUpdatedCb);
	PathDesiredConnectCallback(SettingsUpdatedCb);
	
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
 * Compute desired velocity from the current position and path
 *
 * Takes in @ref PositionActual and compares it to @ref PathDesired 
 * and computes @ref VelocityDesired
 */
static void updatePathVelocity()
{
	float dT = guidanceSettings.UpdatePeriod / 1000.0f;

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
	PathStatusSet(&pathStatus);

	float groundspeed = pathDesired.StartingVelocity + 
	    (pathDesired.EndingVelocity - pathDesired.StartingVelocity) * progress.fractional_progress;
	if(progress.fractional_progress > 1)
		groundspeed = 0;
	
	VelocityDesiredData velocityDesired;
	velocityDesired.North = progress.path_direction[0] * groundspeed;
	velocityDesired.East = progress.path_direction[1] * groundspeed;
	
	float error_speed = progress.error * guidanceSettings.HorizontalPosPI[VTOLPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KP];
	float correction_velocity[2] = {progress.correction_direction[0] * error_speed, 
	    progress.correction_direction[1] * error_speed};
	
	float total_vel = sqrtf(powf(correction_velocity[0],2) + powf(correction_velocity[1],2));
	float scale = 1;
	if(total_vel > guidanceSettings.HorizontalVelMax)
		scale = guidanceSettings.HorizontalVelMax / total_vel;

	// Currently not apply a PID loop for horizontal corrections
	velocityDesired.North += progress.correction_direction[0] * error_speed * scale;
	velocityDesired.East += progress.correction_direction[1] * error_speed * scale;
	
	// Interpolate desired velocity along the path
	float altitudeSetpoint = pathDesired.Start[2] + (pathDesired.End[2] - pathDesired.Start[2]) *
	    bound_min_max(progress.fractional_progress,0,1);

	float downError = altitudeSetpoint - positionActual.Down;
	velocityDesired.Down = pid_apply_antiwindup(&vtol_pids[DOWN_POSITION], downError,
		-guidanceSettings.VerticalVelMax, guidanceSettings.VerticalVelMax, dT);

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
	float downError;
	float northCommand;
	float eastCommand;
	
	float northPos = positionActual.North;
	float eastPos = positionActual.East;
	float downPos = positionActual.Down;

	// Compute desired north command velocity from position error
	northError = pathDesired.End[PATHDESIRED_END_NORTH] - northPos;
	northCommand = pid_apply_antiwindup(&vtol_pids[NORTH_POSITION], northError,
	    -guidanceSettings.HorizontalVelMax, guidanceSettings.HorizontalVelMax, dT);

	// Compute desired east command velocity from position error
	eastError = pathDesired.End[PATHDESIRED_END_EAST] - eastPos;
	eastCommand = pid_apply_antiwindup(&vtol_pids[EAST_POSITION], eastError,
	    -guidanceSettings.HorizontalVelMax, guidanceSettings.HorizontalVelMax, dT);

	// Limit the maximum velocity any direction (not north and east separately)
	float total_vel = sqrtf(powf(northCommand,2) + powf(eastCommand,2));
	float scale = 1;
	if(total_vel > guidanceSettings.HorizontalVelMax)
		scale = guidanceSettings.HorizontalVelMax / total_vel;

	velocityDesired.North = northCommand * scale;
	velocityDesired.East = eastCommand * scale;

	// Compute the desired velocity from the position difference
	downError = pathDesired.End[PATHDESIRED_END_DOWN] - downPos;
	velocityDesired.Down = pid_apply_antiwindup(&vtol_pids[DOWN_POSITION], downError,
	    -guidanceSettings.VerticalVelMax, guidanceSettings.VerticalVelMax, dT);
	
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
static void updateVtolDesiredAttitude()
{
	float dT = guidanceSettings.UpdatePeriod / 1000.0f;

	VelocityDesiredData velocityDesired;
	VelocityActualData velocityActual;
	StabilizationDesiredData stabDesired;
	AttitudeActualData attitudeActual;
	NedAccelData nedAccel;
	StabilizationSettingsData stabSettings;

	float northError;
	float northCommand;
	
	float eastError;
	float eastCommand;

	float downError;
	float downCommand;
		
	VtolPathFollowerSettingsGet(&guidanceSettings);
	
	VelocityActualGet(&velocityActual);
	VelocityDesiredGet(&velocityDesired);
	StabilizationDesiredGet(&stabDesired);
	VelocityDesiredGet(&velocityDesired);
	AttitudeActualGet(&attitudeActual);
	StabilizationSettingsGet(&stabSettings);
	NedAccelGet(&nedAccel);
	
	float northVel = velocityActual.North;
	float eastVel = velocityActual.East;
	float downVel = velocityActual.Down;

	// Compute desired north command from velocity error
	northError = velocityDesired.North - northVel;
	northCommand = pid_apply_antiwindup(&vtol_pids[NORTH_VELOCITY], northError, 
	    -guidanceSettings.MaxRollPitch, guidanceSettings.MaxRollPitch, dT) + velocityDesired.North * guidanceSettings.VelocityFeedforward;
	
	// Compute desired east command from velocity error
	eastError = velocityDesired.East - eastVel;
	eastCommand = pid_apply_antiwindup(&vtol_pids[NORTH_VELOCITY], eastError,
	    -guidanceSettings.MaxRollPitch, guidanceSettings.MaxRollPitch, dT) + velocityDesired.East * guidanceSettings.VelocityFeedforward;
	
	// Compute desired down command.  Using NED accel as the damping term
	downError = velocityDesired.Down - downVel;
	// Negative is critical here since throttle is negative with down
	downCommand = -pid_apply_antiwindup(&vtol_pids[DOWN_VELOCITY], downError, -1, 1, dT) +
	    nedAccel.Down * guidanceSettings.VerticalVelPID[VTOLPATHFOLLOWERSETTINGS_VERTICALVELPID_KD];

	// If this setting is zero then the throttle level available when enabled is used for hover:wq
	float used_throttle_offset = (guidanceSettings.HoverThrottle == 0) ? throttleOffset : guidanceSettings.HoverThrottle;
	stabDesired.Throttle = bound_min_max(downCommand + used_throttle_offset, 0, 1);
	
	// Project the north and east command signals into the pitch and roll based on yaw.
	// For this to behave well the craft should move similarly for 5 deg roll versus 5 deg pitch.
	// Notice the inputs are crudely bounded by the anti-winded but if both N and E were
	// saturated and the craft were at 45 degrees that would result in a value greater than
	// the limit, so apply limit again here.
	stabDesired.Pitch = bound_min_max(-northCommand * cosf(attitudeActual.Yaw * DEG2RAD) + 
				      -eastCommand * sinf(attitudeActual.Yaw * DEG2RAD),
				      -guidanceSettings.MaxRollPitch, guidanceSettings.MaxRollPitch);
	stabDesired.Roll = bound_min_max(-northCommand * sinf(attitudeActual.Yaw * DEG2RAD) + 
				     eastCommand * cosf(attitudeActual.Yaw * DEG2RAD),
				     -guidanceSettings.MaxRollPitch, guidanceSettings.MaxRollPitch);
	
	if(guidanceSettings.ThrottleControl == VTOLPATHFOLLOWERSETTINGS_THROTTLECONTROL_FALSE) {
		// For now override throttle with manual control.  Disable at your risk, quad goes to China.
		ManualControlCommandData manualControl;
		ManualControlCommandGet(&manualControl);
		stabDesired.Throttle = manualControl.Throttle;
	}
	
	stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;

	float yaw;
	switch(guidanceSettings.YawMode) {
	case VTOLPATHFOLLOWERSETTINGS_YAWMODE_RATE:
		/* This is awkward.  This allows the transmitter to control the yaw while flying navigation */
		ManualControlCommandYawGet(&yaw);
		stabDesired.Yaw = stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_YAW] * yaw;      
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_RATE;
		break;
	case VTOLPATHFOLLOWERSETTINGS_YAWMODE_AXISLOCK:
		ManualControlCommandYawGet(&yaw);
		stabDesired.Yaw = stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_YAW] * yaw;      
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK;
		break;
	case VTOLPATHFOLLOWERSETTINGS_YAWMODE_ATTITUDE:
		ManualControlCommandYawGet(&yaw);
		stabDesired.Yaw = stabSettings.YawMax * yaw;      
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
		break;
	case VTOLPATHFOLLOWERSETTINGS_YAWMODE_POI:
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_POI;
		break;
	}
	
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
	VtolPathFollowerSettingsGet(&guidanceSettings);

	// Configure the velocity control PID loops
	pid_configure(&vtol_pids[NORTH_VELOCITY], 
		guidanceSettings.HorizontalVelPID[VTOLPATHFOLLOWERSETTINGS_HORIZONTALVELPID_KP], // Kp
		guidanceSettings.HorizontalVelPID[VTOLPATHFOLLOWERSETTINGS_HORIZONTALVELPID_KI], // Ki
		0, // Kd
		guidanceSettings.HorizontalVelPID[VTOLPATHFOLLOWERSETTINGS_HORIZONTALVELPID_ILIMIT]);
	pid_configure(&vtol_pids[EAST_VELOCITY], 
		guidanceSettings.HorizontalVelPID[VTOLPATHFOLLOWERSETTINGS_HORIZONTALVELPID_KP], // Kp
		guidanceSettings.HorizontalVelPID[VTOLPATHFOLLOWERSETTINGS_HORIZONTALVELPID_KI], // Ki
		0, // Kd
		guidanceSettings.HorizontalVelPID[VTOLPATHFOLLOWERSETTINGS_HORIZONTALVELPID_ILIMIT]);
	pid_configure(&vtol_pids[DOWN_VELOCITY], 
		guidanceSettings.VerticalVelPID[VTOLPATHFOLLOWERSETTINGS_VERTICALVELPID_KP], // Kp
		guidanceSettings.VerticalVelPID[VTOLPATHFOLLOWERSETTINGS_VERTICALVELPID_KI], // Ki
		0, // Kd
		guidanceSettings.VerticalVelPID[VTOLPATHFOLLOWERSETTINGS_VERTICALVELPID_ILIMIT]);

	// Configure the position control (velocity output) PID loops
	pid_configure(&vtol_pids[NORTH_POSITION],
		guidanceSettings.HorizontalPosPI[VTOLPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KP],
		guidanceSettings.HorizontalPosPI[VTOLPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KI],
		0,
		guidanceSettings.HorizontalPosPI[VTOLPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_ILIMIT]);
	pid_configure(&vtol_pids[EAST_POSITION],
		guidanceSettings.HorizontalPosPI[VTOLPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KP],
		guidanceSettings.HorizontalPosPI[VTOLPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KI],
		0,
		guidanceSettings.HorizontalPosPI[VTOLPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_ILIMIT]);
	pid_configure(&vtol_pids[DOWN_POSITION],
		guidanceSettings.VerticalPosPI[VTOLPATHFOLLOWERSETTINGS_VERTICALPOSPI_KP],
		guidanceSettings.VerticalPosPI[VTOLPATHFOLLOWERSETTINGS_VERTICALPOSPI_KI],
		0,
		guidanceSettings.VerticalPosPI[VTOLPATHFOLLOWERSETTINGS_VERTICALPOSPI_ILIMIT]);


	PathDesiredGet(&pathDesired);
}

/**
 * @}
 * @}
 */

