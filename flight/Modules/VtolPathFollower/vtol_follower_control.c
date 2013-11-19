/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup VtolPathFollower VTOL path follower module
 * @{
 *
 * @file       vtol_follower_control.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Control algorithms for the vtol follower
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
#include "attitudeactual.h"
#include "pathdesired.h"        // object that will be updated by the module
#include "positionactual.h"
#include "manualcontrolcommand.h"
#include "flightstatus.h"
#include "gpsvelocity.h"
#include "gpsposition.h"
#include "nedaccel.h"
#include "pathstatus.h"
#include "stabilizationdesired.h"
#include "stabilizationsettings.h"
#include "systemsettings.h"
#include "velocitydesired.h"
#include "velocityactual.h"
#include "vtolpathfollowersettings.h"
#include "coordinate_conversions.h"

// Private types

// Private variables
static VtolPathFollowerSettingsData guidanceSettings;
float throttle_offset;
struct pid vtol_pids[VTOL_PID_NUM];

// Private functions

/**
 * Compute desired velocity to follow the desired path from the current location.
 * @param[in] dT the time since last evaluation
 * @param [in] pathDesired the desired path to follow
 * @param [out] progress the current progress information along that path
 * @returns 0 if successful, <0 if an error occurred
 *
 * The calculated velocity to attempt is stored in @ref VelocityDesired
 */
int32_t vtol_follower_control_path(const float dT, const PathDesiredData *pathDesired,
	struct path_status *progress)
{
	PositionActualData positionActual;
	PositionActualGet(&positionActual);
	
	const float cur[3] = {positionActual.North, positionActual.East, positionActual.Down};
	
	path_progress(pathDesired, cur, progress);
	
	// Update the path status UAVO
	PathStatusData pathStatus;
	PathStatusGet(&pathStatus);
	pathStatus.fractional_progress = progress->fractional_progress;
	if (pathStatus.fractional_progress < 1)
		pathStatus.Status = PATHSTATUS_STATUS_INPROGRESS;
	else
		pathStatus.Status = PATHSTATUS_STATUS_COMPLETED;
	PathStatusSet(&pathStatus);

	float groundspeed = pathDesired->StartingVelocity + 
	    (pathDesired->EndingVelocity - pathDesired->StartingVelocity) * progress->fractional_progress;
	if(progress->fractional_progress > 1)
		groundspeed = 0;
	
	VelocityDesiredData velocityDesired;
	velocityDesired.North = progress->path_direction[0] * groundspeed;
	velocityDesired.East = progress->path_direction[1] * groundspeed;
	
	float error_speed = progress->error * guidanceSettings.HorizontalPosPI[VTOLPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KP];
	float correction_velocity[2] = {progress->correction_direction[0] * error_speed, 
	    progress->correction_direction[1] * error_speed};
	
	float total_vel = sqrtf(powf(correction_velocity[0],2) + powf(correction_velocity[1],2));
	float scale = 1;
	if(total_vel > guidanceSettings.HorizontalVelMax)
		scale = guidanceSettings.HorizontalVelMax / total_vel;

	// Currently not apply a PID loop for horizontal corrections
	velocityDesired.North += progress->correction_direction[0] * error_speed * scale;
	velocityDesired.East += progress->correction_direction[1] * error_speed * scale;
	
	// Interpolate desired velocity along the path
	float altitudeSetpoint = pathDesired->Start[2] + (pathDesired->End[2] - pathDesired->Start[2]) *
	    bound_min_max(progress->fractional_progress,0,1);

	float downError = altitudeSetpoint - positionActual.Down;
	velocityDesired.Down = pid_apply_antiwindup(&vtol_pids[DOWN_POSITION], downError,
		-guidanceSettings.VerticalVelMax, guidanceSettings.VerticalVelMax, dT);

	VelocityDesiredSet(&velocityDesired);

	return 0;
}

/**
 * Control algorithm to stay or approach at a fixed location.
 * @param[in] dT the time since last evaluation
 * @param[in] ned The position to attempt to hold
 * This method does not attempt any particular path, simply a straight line
 * approach. The calculated velocity to attempt is stored in @ref 
 * VelocityDesired. 
 *
 * Takes in @ref PositionActual and compares it to @ref PositionDesired 
 * and computes @ref VelocityDesired
 */
int32_t vtol_follower_control_endpoint(const float dT, const float *hold_pos_ned)
{
	PositionActualData positionActual;
	VelocityDesiredData velocityDesired;
	
	PositionActualGet(&positionActual);
	VelocityDesiredGet(&velocityDesired);
	
	float northError;
	float eastError;
	float downError;
	float northCommand;
	float eastCommand;
	
	const float cur_pos_ned[3] = {positionActual.North, positionActual.East, positionActual.Down};

	// Compute desired north command velocity from position error
	northError = hold_pos_ned[0] - cur_pos_ned[0];
	northCommand = pid_apply_antiwindup(&vtol_pids[NORTH_POSITION], northError,
	    -guidanceSettings.HorizontalVelMax, guidanceSettings.HorizontalVelMax, dT);

	// Compute desired east command velocity from position error
	eastError = hold_pos_ned[1] - cur_pos_ned[1];
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
	downError = hold_pos_ned[2] - cur_pos_ned[2];
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

	return 0;
}

/**
 * Control algorithm to land at a fixed location
 * @param[in] dT the time since last evaluation
 * @param[in] ned The position to attempt to land over (down ignored)
 * @param[in] land_velocity The speed to descend
 * @param[out] landed True once throttle low and velocity at zero
 *
 * Takes in @ref PositionActual and compares it to the hold position
 * and computes @ref VelocityDesired
 */
int32_t vtol_follower_control_land(const float dT, const float *hold_pos_ned,
	bool *landed)
{
	PositionActualData positionActual;
	VelocityDesiredData velocityDesired;
	
	PositionActualGet(&positionActual);
	VelocityDesiredGet(&velocityDesired);
	
	const float land_velocity = guidanceSettings.LandingRate;

	float northError;
	float eastError;
	float northCommand;
	float eastCommand;
	
	const float cur_pos_ned[3] = {positionActual.North, positionActual.East, positionActual.Down};

	// Compute desired north command velocity from position error
	northError = hold_pos_ned[0] - cur_pos_ned[0];
	northCommand = pid_apply_antiwindup(&vtol_pids[NORTH_POSITION], northError,
	    -guidanceSettings.HorizontalVelMax, guidanceSettings.HorizontalVelMax, dT);

	// Compute desired east command velocity from position error
	eastError = hold_pos_ned[1] - cur_pos_ned[1];
	eastCommand = pid_apply_antiwindup(&vtol_pids[EAST_POSITION], eastError,
	    -guidanceSettings.HorizontalVelMax, guidanceSettings.HorizontalVelMax, dT);

	// Limit the maximum velocity any direction (not north and east separately)
	float total_vel = sqrtf(powf(northCommand,2) + powf(eastCommand,2));
	float scale = 1;
	if(total_vel > guidanceSettings.HorizontalVelMax)
		scale = guidanceSettings.HorizontalVelMax / total_vel;

	velocityDesired.North = northCommand * scale;
	velocityDesired.East = eastCommand * scale;
	velocityDesired.Down = land_velocity;
	
	VelocityDesiredSet(&velocityDesired);	

	// Indicate whether we are in radius of this endpoint
	uint8_t path_status = PATHSTATUS_STATUS_INPROGRESS;
	float distance2 = powf(northError, 2) + powf(eastError, 2);
	if (distance2 < (guidanceSettings.EndpointRadius * guidanceSettings.EndpointRadius)) {
		path_status = PATHSTATUS_STATUS_COMPLETED;
	}
	PathStatusStatusSet(&path_status);

	return 0;
}

/**
 * Compute the desired acceleration based on the desired
 * velocity and actual velocity
 */
static int32_t vtol_follower_control_accel(float dT)
{
	VelocityDesiredData velocityDesired;
	VelocityActualData velocityActual;

	float northError;
	float northCommand;
	
	float eastError;
	float eastCommand;

	float downError;
	float downCommand;

	VelocityActualGet(&velocityActual);
	VelocityDesiredGet(&velocityDesired);
	
	float northVel = velocityActual.North;
	float eastVel = velocityActual.East;
	float downVel = velocityActual.Down;

	// TODO: update the PID wind up limits

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
	float down_accel;
	NedAccelDownGet(&down_accel);
	downCommand = -pid_apply_antiwindup(&vtol_pids[DOWN_VELOCITY], downError, -1, 1, dT) +
	    down_accel * guidanceSettings.VerticalVelPID[VTOLPATHFOLLOWERSETTINGS_VERTICALVELPID_KD];

	// Store the desired acceleration
	AccelDesiredData accelDesired;
	accelDesired.North = northCommand;
	accelDesired.East = eastCommand;
	accelDesired.Down = downCommand;
	AccelDesiredSet(&accelDesired);

	return 0;
}

/**
 * Compute desired attitude from the desired velocity
 * @param[in] dT the time since last evaluation
 *
 * Takes in @ref NedActual which has the acceleration in the 
 * NED frame as the feedback term and then compares the 
 * @ref VelocityActual against the @ref VelocityDesired
 */
int32_t vtol_follower_control_attitude(float dT)
{
	vtol_follower_control_accel(dT);

	AccelDesiredData accelDesired;
	AccelDesiredGet(&accelDesired);

	StabilizationDesiredData stabDesired;

	float northCommand = accelDesired.North;
	float eastCommand = accelDesired.East;
	float downCommand = accelDesired.Down;

	// If this setting is zero then the throttle level available when enabled is used for hover:wq
	float used_throttle_offset = (guidanceSettings.HoverThrottle == 0) ? throttle_offset : guidanceSettings.HoverThrottle;
	stabDesired.Throttle = bound_min_max(downCommand + used_throttle_offset, 0, 1);
	
	// Project the north and east acceleration signals into body frame
	float yaw;
	AttitudeActualYawGet(&yaw);
	float forward_accel_desired = -northCommand * cosf(yaw * DEG2RAD) + -eastCommand * sinf(yaw * DEG2RAD);
	float right_accel_desired = -northCommand * sinf(yaw * DEG2RAD) + eastCommand * cosf(yaw * DEG2RAD);

	// Set the angle that would achieve the desired acceleration given the thrust is enough for a hover
	stabDesired.Pitch = bound_min_max(RAD2DEG * atanf(forward_accel_desired / GRAVITY),
	                   -guidanceSettings.MaxRollPitch, guidanceSettings.MaxRollPitch);
	stabDesired.Roll = bound_min_max(RAD2DEG * atanf(right_accel_desired / GRAVITY),
	                   -guidanceSettings.MaxRollPitch, guidanceSettings.MaxRollPitch);
	
	stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;

	// Optionally allow transmitter to control throttle for safety
	if(guidanceSettings.ThrottleControl == VTOLPATHFOLLOWERSETTINGS_THROTTLECONTROL_FALSE) {
		ManualControlCommandThrottleGet(&stabDesired.Throttle);
	}
	
	// Various ways to control the yaw that are essentially manual passthrough. However, because we do not have a fine
	// grained mechanism of manual setting the yaw as it normally would we need to duplicate that code here
	float manual_rate[STABILIZATIONSETTINGS_MANUALRATE_NUMELEM];
	switch(guidanceSettings.YawMode) {
	case VTOLPATHFOLLOWERSETTINGS_YAWMODE_RATE:
		/* This is awkward.  This allows the transmitter to control the yaw while flying navigation */
		ManualControlCommandYawGet(&yaw);
		StabilizationSettingsManualRateGet(manual_rate);
		stabDesired.Yaw = manual_rate[STABILIZATIONSETTINGS_MANUALRATE_YAW] * yaw;      
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_RATE;
		break;
	case VTOLPATHFOLLOWERSETTINGS_YAWMODE_AXISLOCK:
		ManualControlCommandYawGet(&yaw);
		StabilizationSettingsManualRateGet(manual_rate);
		stabDesired.Yaw = manual_rate[STABILIZATIONSETTINGS_MANUALRATE_YAW] * yaw;      
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK;
		break;
	case VTOLPATHFOLLOWERSETTINGS_YAWMODE_ATTITUDE:
	{
		uint8_t yaw_max;
		StabilizationSettingsYawMaxGet(&yaw_max);
		ManualControlCommandYawGet(&yaw);
		stabDesired.Yaw = yaw_max * yaw;      
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	}
		break;
	case VTOLPATHFOLLOWERSETTINGS_YAWMODE_POI:
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_POI;
		break;
	}
	
	StabilizationDesiredSet(&stabDesired);

	return 0;
}

void vtol_follower_control_settings_updated(UAVObjEvent * ev)
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
}

/**
 * @}
 * @}
 */

