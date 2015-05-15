/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup VtolPathFollower VTOL path follower module
 * @{
 *
 * @file       vtol_follower_control.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
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

#include "coordinate_conversions.h"
#include "physical_constants.h"
#include "misc_math.h"
#include "paths.h"
#include "pid.h"

#include "vtol_follower_priv.h"

#include "acceldesired.h"
#include "altitudeholdsettings.h"
#include "altitudeholdstate.h" 
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

// Private variables
static VtolPathFollowerSettingsData guidanceSettings;
static AltitudeHoldSettingsData altitudeHoldSettings;
struct pid vtol_pids[VTOL_PID_NUM];

// Constants used in deadband calculation
static float vtol_path_m=0, vtol_path_r=0, vtol_end_m=0, vtol_end_r=0;

static int32_t vtol_follower_control_simple(const float dT,
	const float *hold_pos_ned, bool landing, bool update_status);

/**
 * Interpolate values (groundspeeds, altitudes) over flight legs
 * @param[in] fraction how far we are through the leg
 * @param[in] beginVal the configured value for the beginning of the leg
 * @param[in] endVal the configured value for the end of the leg
 * @returns the interpolated value
 *
 * Simple linear interpolation with clipping to ends (fraction>=0, <=1).
 */
static float vtol_interpolate(const float fraction, const float beginVal,
	const float endVal) {
	return beginVal + (endVal - beginVal) * bound_min_max(fraction, 0, 1);
}

/**
 * Calculate pythagorean magnitude of a vector.
 * @param[in] v pointer to a float array
 * @param[in] n length of the amount to take the magnitude of
 * @returns Root of sum of squares of the vector
 *
 * Note that sometimes we only take the magnitude of the first 2 elements
 * of a 3-vector to get the horizontal component.
 */
static float vtol_magnitude(const float *v, int n)
{
	float sum=0;

	for (int i=0; i<n; i++) {
		sum += powf(v[i], 2);
	}

	return sqrtf(sum);
}

/**
 * Subtract two 3-vectors, and optionally normalize to return an error value.
 * @param[in] actual the measured process value
 * @param[in] desired the setpoint of the system
 * @param[out] the resultant error vector desired-actual
 * @param[in] normalize True if it's desired to normalize the output vector
 * @returns the norm, for fun.
 */
static float vtol_calculate_distances(const float *actual,
	const float *desired, float *out, bool normalize)
{
	out[0] = desired[0] - actual[0];
	out[1] = desired[1] - actual[1];
	out[2] = desired[2] - actual[2];

	if (normalize) {
		float mag=vtol_magnitude(out, 3);

		if (mag < 0.00001f) {
			/* Just pick a direction. */
			out[0] = 1.0; out[1] = 0.0; out[2] = 0.0;
		} else {
			out[0] /= mag;  out[1] /= mag;  out[2] /= mag;
		}

		return mag;
	}

	return 0.0;
}

/**
 * Clip a velocity 2-vector  while maintaining vector direction.
 * @param[in,out] vels velocity to clip
 * @param[in] limit desired limit magnitude.
 *
 * if mag(vels) > limit, vels=vels / mag(vels) * limit
 */
static void vtol_limit_velocity(float *vels, float limit)
{
	float mag = vtol_magnitude(vels, 2);	// only horiz component
	float scale = mag / limit;

	if (scale > 1) {
		vels[0] /= mag;
		vels[1] /= mag;
	}
}

/**
 * Apply a "cubic deadband" to the input.
 * @param[in] in the value to deadband
 * @param[in] w deadband width
 * @param[in] b slope of deadband at in=0
 * @param[in] m cubic weighting calculated by vtol_deadband_setup
 * @param[in] r integrated response at in=w, calculated by vtol_deadband_setup
 *
 * "Real" deadbands are evil.  Control systems end up fighting the edge.
 * You don't teach your integrator about emerging drift.  Discontinuities
 * in your control inputs cause all kinds of neat stuff.  As a result this
 * calculates a cubic function within the deadband which has a low slope
 * within the middle, but unity slope at the edge.
 */
static float vtol_deadband(float in, float w, float b, float m, float r)
{
	// First get the nice linear bits -- outside the deadband-- out of
	// the way.
	if (in <= -w) {
		return in+w-r;
	} else if (in >= w) {
		return in-w+r;
	}


	return powf(m*in, 3)+b*in;
}

/**
 * Calculate the "cubic deadband" system parameters.
 * @param[in] The width of the deadband
 * @param[in] Slope of deadband at in=0, sane values between 0 and 1.
 * @param[out] m cubic weighting of function
 * @param[out] integrated response at in=w
 */
static void vtol_deadband_setup(float w, float b, float *m, float *r)
{
	/* So basically.. we want the function to be tangent to the
	** linear sections-- have a slope of 1-- at -w and w.  In the
	** middle we want a slope of b.   So the cube here does all the
	** work b isn't doing. */
	*m = cbrtf((1-b)/(3*powf(w,2)));

	*r = powf(*m*w, 3)+b*w;
}

/**
 * Compute desired velocity to follow the desired path from the current location.
 * @param[in] dT the time since last evaluation
 * @param[in] pathDesired the desired path to follow
 * @param[out] progress the current progress information along that path
 * @returns 0 if successful, <0 if an error occurred
 *
 * The calculated velocity to attempt is stored in @ref VelocityDesired
 */
int32_t vtol_follower_control_path(const float dT, const PathDesiredData *pathDesired,
	struct path_status *progress)
{
	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	VelocityActualData velocityActual;
	VelocityActualGet(&velocityActual);

	PathStatusData pathStatus;
	PathStatusGet(&pathStatus);

	const float cur_pos_ned[3] = {
		positionActual.North +
		    velocityActual.North * guidanceSettings.PositionFeedforward,
		positionActual.East +
		    velocityActual.East * guidanceSettings.PositionFeedforward,
		positionActual.Down };

	path_progress(pathDesired, cur_pos_ned, progress);

	// Check if we have already completed this leg
	bool current_leg_completed = 
		(pathStatus.Status == PATHSTATUS_STATUS_COMPLETED) &&
		(pathStatus.Waypoint == pathDesired->Waypoint);

	pathStatus.fractional_progress = progress->fractional_progress;
	pathStatus.error = progress->error;
	pathStatus.Waypoint = pathDesired->Waypoint;

	// Figure out how low (high) we should be and the error
	const float altitudeSetpoint = vtol_interpolate(progress->fractional_progress,
	    pathDesired->Start[2], pathDesired->End[2]);

	const float downError = altitudeSetpoint - positionActual.Down;

	// If leg is completed signal this
	if (current_leg_completed || pathStatus.fractional_progress > 1.0f) {
		const bool criterion_altitude =
			(downError > -guidanceSettings.WaypointAltitudeTol) ||
			(!guidanceSettings.ThrottleControl);

		// Once we complete leg and hit altitude criterion signal this
		// waypoint is done.  Or if we're not controlling throttle,
		// ignore height for completion.

		// Waypoint heights are thus treated as crossing restrictions-
		// cross this point at or above...
		if (criterion_altitude || current_leg_completed) {
			pathStatus.Status = PATHSTATUS_STATUS_COMPLETED;
			PathStatusSet(&pathStatus);
		} else {
			pathStatus.Status = PATHSTATUS_STATUS_INPROGRESS;
			PathStatusSet(&pathStatus);
		}

		// Wait here for new path segment
		return vtol_follower_control_simple(dT, pathDesired->End, false, false);
	}
	
	// Interpolate desired velocity and altitude along the path
	float groundspeed = vtol_interpolate(progress->fractional_progress,
	    pathDesired->StartingVelocity, pathDesired->EndingVelocity);

	float error_speed = vtol_deadband(progress->error,
		guidanceSettings.PathDeadbandWidth,
		guidanceSettings.PathDeadbandCenterGain,
		vtol_path_m, vtol_path_r) *
	    guidanceSettings.HorizontalPosPI[VTOLPATHFOLLOWERSETTINGS_HORIZONTALPOSPI_KP];

	/* Sum the desired path movement vector with the correction vector */
	float commands_ned[3];
	commands_ned[0] = progress->path_direction[0] * groundspeed +
	    progress->correction_direction[0] * error_speed;
	
	commands_ned[1] = progress->path_direction[1] * groundspeed +
	    progress->correction_direction[1] * error_speed;

	/* Limit the total velocity based on the configured value. */
	vtol_limit_velocity(commands_ned, guidanceSettings.HorizontalVelMax);

	commands_ned[2] = pid_apply_antiwindup(&vtol_pids[DOWN_POSITION], downError,
		-guidanceSettings.VerticalVelMax, guidanceSettings.VerticalVelMax, dT);

	VelocityDesiredData velocityDesired;
	VelocityDesiredGet(&velocityDesired);
	velocityDesired.North = commands_ned[0];
	velocityDesired.East = commands_ned[1];
	velocityDesired.Down = commands_ned[2];
	VelocityDesiredSet(&velocityDesired);

	pathStatus.Status = PATHSTATUS_STATUS_INPROGRESS;
	PathStatusSet(&pathStatus);

	return 0;
}

/**
 * Controller to maintain/seek a position and optionally descend.
 * @param[in] dT time since last eval
 * @param[in] hold_pos_ned a position to hold
 * @param[in] landing whether to descend
 * @param[in] update_status does this update path_status, or does somoene else?
 */
static int32_t vtol_follower_control_simple(const float dT,
	const float *hold_pos_ned, bool landing, bool update_status) {
	PositionActualData positionActual;
	VelocityDesiredData velocityDesired;
	
	PositionActualGet(&positionActual);
	
	VelocityActualData velocityActual;

	VelocityActualGet(&velocityActual);

	/* Where would we be in ___ second at current rates? */	
	const float cur_pos_ned[3] = {
		positionActual.North +
		    velocityActual.North * guidanceSettings.PositionFeedforward,
		positionActual.East +
		    velocityActual.East * guidanceSettings.PositionFeedforward,
		positionActual.Down };

	float errors_ned[3];

	/* Calculate the difference between where we want to be and the
	 * above position */
	float horiz_error_norm = vtol_calculate_distances(cur_pos_ned,
		hold_pos_ned, errors_ned, true);

	/* Apply a cubic deadband; if we are already really close don't work
	 * as hard to fix it as if we're far away.  Prevents chasing high
	 * frequency noise in direction of correction.
	 *
	 * That is, when we're far, noise in estimated position mostly results
	 * in noise/dither in how fast we go towards the target.  When we're
	 * close, there's a large directional / hunting component.  This
	 * attenuates that.
	 */

	horiz_error_norm = vtol_deadband(horiz_error_norm,
		guidanceSettings.EndpointDeadbandWidth,
		guidanceSettings.EndpointDeadbandCenterGain,
		vtol_end_m, vtol_end_r);

	float damped_ne[2] = { errors_ned[0] * horiz_error_norm,
			errors_ned[1] * horiz_error_norm };

	float commands_ned[3];

	// Compute desired north command velocity from position error
	commands_ned[0] = pid_apply_antiwindup(&vtol_pids[NORTH_POSITION], damped_ne[0],
	    -guidanceSettings.HorizontalVelMax, guidanceSettings.HorizontalVelMax, dT);

	// Compute desired east command velocity from position error
	commands_ned[1] = pid_apply_antiwindup(&vtol_pids[EAST_POSITION], damped_ne[1],
	    -guidanceSettings.HorizontalVelMax, guidanceSettings.HorizontalVelMax, dT);

	if (!landing) {
		// Compute desired down comand velocity from the position difference
		commands_ned[2] = pid_apply_antiwindup(&vtol_pids[DOWN_POSITION], errors_ned[2],
		    -guidanceSettings.VerticalVelMax, guidanceSettings.VerticalVelMax, dT);
	} else {
		// Just use the landing rate.
		commands_ned[2] = guidanceSettings.LandingRate;
	}
	
	// Limit the maximum horizontal velocity any direction (not north and east separately)
	vtol_limit_velocity(commands_ned, guidanceSettings.HorizontalVelMax);

	velocityDesired.North = commands_ned[0];
	velocityDesired.East = commands_ned[1];
	velocityDesired.Down = commands_ned[2];

	VelocityDesiredSet(&velocityDesired);	

	if (update_status) {
		uint8_t path_status = PATHSTATUS_STATUS_INPROGRESS;

		if (!landing) {
			const bool criterion_altitude =
				(errors_ned[2]> -guidanceSettings.WaypointAltitudeTol) || (!guidanceSettings.ThrottleControl);

			// Indicate whether we are in radius of this endpoint
			// And at/above the altitude requested
			if ((vtol_magnitude(errors_ned, 2) < guidanceSettings.EndpointRadius) && criterion_altitude) {
				path_status = PATHSTATUS_STATUS_COMPLETED;
			}
		}  // landing never terminates.

		PathStatusStatusSet(&path_status);
	}

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
	return vtol_follower_control_simple(dT, hold_pos_ned, false, true);
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
	return vtol_follower_control_simple(dT, hold_pos_ned, false, true);
}

/**
 * Compute the desired acceleration based on the desired
 * velocity and actual velocity
 */
static int32_t vtol_follower_control_accel(float dT)
{
	VelocityDesiredData velocityDesired;
	VelocityActualData velocityActual;
	AccelDesiredData accelDesired;
	NedAccelData nedAccel;

	float north_error, north_acceleration;
	float east_error, east_acceleration;
	float down_error;

	static float last_north_velocity;
	static float last_east_velocity;

	NedAccelGet(&nedAccel);
	VelocityActualGet(&velocityActual);
	VelocityDesiredGet(&velocityDesired);
	
	// Optionally compute the acceleration required component from a changing velocity desired
	if (guidanceSettings.VelocityChangePrediction == VTOLPATHFOLLOWERSETTINGS_VELOCITYCHANGEPREDICTION_TRUE && dT > 0) {
		north_acceleration = (velocityDesired.North - last_north_velocity) / dT;
		east_acceleration = (velocityDesired.East - last_east_velocity) / dT;
		last_north_velocity = velocityDesired.North;
		last_east_velocity = velocityDesired.East;
	} else {
		north_acceleration = 0;
		east_acceleration = 0;
	}

	// Convert the max angles into the maximum angle that would be requested
	const float MAX_ACCELERATION = GRAVITY * sinf(guidanceSettings.MaxRollPitch * DEG2RAD);

	// Compute desired north command from velocity error
	north_error = velocityDesired.North - velocityActual.North;
	north_acceleration += pid_apply_antiwindup(&vtol_pids[NORTH_VELOCITY], north_error,
	    -MAX_ACCELERATION, MAX_ACCELERATION, dT) +
	    velocityDesired.North * guidanceSettings.VelocityFeedforward +
	    -nedAccel.North * guidanceSettings.HorizontalVelPID[VTOLPATHFOLLOWERSETTINGS_HORIZONTALVELPID_KD];
	
	// Compute desired east command from velocity error
	east_error = velocityDesired.East - velocityActual.East;
	east_acceleration += pid_apply_antiwindup(&vtol_pids[EAST_VELOCITY], east_error,
	    -MAX_ACCELERATION, MAX_ACCELERATION, dT) +
	    velocityDesired.East * guidanceSettings.VelocityFeedforward +
	    -nedAccel.East * guidanceSettings.HorizontalVelPID[VTOLPATHFOLLOWERSETTINGS_HORIZONTALVELPID_KD];

	accelDesired.North = north_acceleration;
	accelDesired.East = east_acceleration;

	// Note: vertical controller really isn't currently in units of acceleration and the anti-windup may
	// not be appropriate. However, it is fine for now since it this output is just directly used on the
	// output. To use this appropriately we need a model of throttle to acceleration.
	// Compute desired down command.  Using NED accel as the damping term
	down_error = velocityDesired.Down - velocityActual.Down;
	// Negative is critical here since throttle is negative with down
	accelDesired.Down = -pid_apply_antiwindup(&vtol_pids[DOWN_VELOCITY], down_error, -1, 0, dT);

	// Store the desired acceleration
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


	// Calculate the throttle setting or use pass through from transmitter
	if (guidanceSettings.ThrottleControl == VTOLPATHFOLLOWERSETTINGS_THROTTLECONTROL_FALSE) {
		ManualControlCommandThrottleGet(&stabDesired.Throttle);
	} else {
		float downCommand = accelDesired.Down;

		AltitudeHoldStateData altitudeHoldState;
		altitudeHoldState.VelocityDesired = downCommand;
		altitudeHoldState.Integral = vtol_pids[DOWN_VELOCITY].iAccumulator / 1000.0f;
		altitudeHoldState.AngleGain = 1.0f;

		if (altitudeHoldSettings.AttitudeComp > 0) {
			// Throttle desired is at this point the mount desired in the up direction, we can
			// account for the attitude if desired
			AttitudeActualData attitudeActual;
			AttitudeActualGet(&attitudeActual);

			// Project a unit vector pointing up into the body frame and
			// get the z component
			float fraction = attitudeActual.q1 * attitudeActual.q1 -
			                 attitudeActual.q2 * attitudeActual.q2 -
			                 attitudeActual.q3 * attitudeActual.q3 +
			                 attitudeActual.q4 * attitudeActual.q4;

			// Add ability to scale up the amount of compensation to achieve
			// level forward flight
			fraction = powf(fraction, (float) altitudeHoldSettings.AttitudeComp / 100.0f);

			// Dividing by the fraction remaining in the vertical projection will
			// attempt to compensate for tilt. This acts like the thrust is linear
			// with the output which isn't really true. If the fraction is starting
			// to go negative we are inverted and should shut off throttle
			downCommand = (fraction > 0.1f) ? (downCommand / fraction) : 0.0f;

			altitudeHoldState.AngleGain = 1.0f / fraction;
		}

		altitudeHoldState.Throttle = downCommand;
		AltitudeHoldStateSet(&altitudeHoldState);

		stabDesired.Throttle = bound_min_max(downCommand, 0, 1);
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
	case VTOLPATHFOLLOWERSETTINGS_YAWMODE_PATH:
	{
		// Face forward on the path
		VelocityDesiredData velocityDesired;
		VelocityDesiredGet(&velocityDesired);
		float total_vel2 = velocityDesired.East*velocityDesired.East + velocityDesired.North*velocityDesired.North;
		float path_direction = atan2f(velocityDesired.East, velocityDesired.North) * RAD2DEG;
		if (total_vel2 > 1) {
			stabDesired.Yaw = path_direction;
			stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
		} else {
			stabDesired.Yaw = 0;
			stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_RATE;
		}
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

	// The parameters for vertical control are shared with Altitude Hold
	AltitudeHoldSettingsGet(&altitudeHoldSettings);
	pid_configure(&vtol_pids[DOWN_POSITION], altitudeHoldSettings.PositionKp, 0, 0, 0);
	pid_configure(&vtol_pids[DOWN_VELOCITY],
	              altitudeHoldSettings.VelocityKp, altitudeHoldSettings.VelocityKi,
	              0, 1);  // Note the ILimit here is 1 because we use this offset to set the throttle offset

	// Calculate the constants used in the deadband calculation
	vtol_deadband_setup(guidanceSettings.EndpointDeadbandWidth,
	    guidanceSettings.EndpointDeadbandCenterGain,
	    &vtol_end_m, &vtol_end_r);

	vtol_deadband_setup(guidanceSettings.PathDeadbandWidth,
	    guidanceSettings.PathDeadbandCenterGain,
	    &vtol_path_m, &vtol_path_r);

}

/**
 * @}
 * @}
 */

