/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup StabilizationModule Stabilization Module
 * @{
 *
 * @file       virtualflybar.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Virtual flybar control mode
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
#include "physical_constants.h"
#include "pid.h"
#include "stabilization.h"
#include "stabilizationsettings.h"

//! Private variables
static float vbar_integral[MAX_AXES];
static float vbar_decay = 0.991f;

//! Private methods
static float bound(float val, float range);

int stabilization_virtual_flybar(float gyro, float command, float *output, float dT, bool reinit, uint32_t axis, struct pid *pid, StabilizationSettingsData *settings)
{
	float gyro_gain = 1.0f;

	if(reinit)
		vbar_integral[axis] = 0;

	// Track the angle of the virtual flybar which includes a slow decay
	vbar_integral[axis] = vbar_integral[axis] * vbar_decay + gyro * dT;
	vbar_integral[axis] = bound(vbar_integral[axis], settings->VbarMaxAngle);

	// Compute the normal PID controller output
	float pid_out = pid_apply_setpoint(pid,  0,  gyro, dT);

	// Command signal can indicate how much to disregard the gyro feedback (fast flips)
	if (settings->VbarGyroSuppress > 0.0f) {
		gyro_gain = (1.0f - fabsf(command) * settings->VbarGyroSuppress / 100.0f);
		gyro_gain = (gyro_gain < 0.0f) ? 0.0f : gyro_gain;
	}

	// Command signal is composed of stick input added to the gyro and virtual flybar
	// Note the PID output has a positive sign (consistent with its use in other places)
	// but the integral is negative since it is the angle of the virtual flybar which is
	// the negative of the accumulated error.
	*output = command * settings->VbarSensitivity[axis] + 
	    gyro_gain * (pid_out - vbar_integral[axis] * pid->i);

	return 0;
}

/**
 * Want to keep the virtual flybar fixed in world coordinates as we pirouette
 * @param[in] z_gyro The deg/s of rotation along the z axis
 * @param[in] dT The time since last sample
 */
int stabilization_virtual_flybar_pirocomp(float z_gyro, float dT)
{
	float cy = cosf(z_gyro * DEG2RAD * dT);
	float sy = sinf(z_gyro * DEG2RAD * dT);

	float vbar_pitch = cy * vbar_integral[1] - sy * vbar_integral[0];
	float vbar_roll = sy * vbar_integral[1] + cy * vbar_integral[0];

	vbar_integral[1] = vbar_pitch;
	vbar_integral[0] = vbar_roll;

	return 0;
}

/**
 * Bound input value between limits
 */
static float bound(float val, float range)
{
	if(val < -range) {
		val = -range;
	} else if(val > range) {
		val = range;
	}
	return val;
}

/**
 * @}
 * @}
 */
