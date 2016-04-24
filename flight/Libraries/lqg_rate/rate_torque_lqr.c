/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       rate_torque_lqr.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      LQR controller based on rate and torque estimate
 *
 * @see        The GNU Public License (GPL) Version 3
 *
 ******************************************************************************/
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

#include "rate_torque_kf.h"
#include "rate_torque_lqr.h"

// This is the LQR tuning matrix. This is currently calculated statically
// based on the system dynamics (measured with system identification) and
// the LQR cost matricies. Note all the zeros because there is no cross
// axis cost (at least in this implementation).
static float Lr[3][9] = {
  {0.037116f,0.000000f,0.000000f,6.995101f,0.000000f,0.000000f,0.783025f,0.000000f,0.000000f},
  {0.000000f,0.035699f,0.000000f,0.000000f,7.518292f,0.000000f,0.000000f,0.768337f,0.000000f},
  {0.000000f,0.000000f,0.029410f,0.000000f,0.000000f,0.045482f,0.000000f,0.000000f,0.266737f},
};

static float La[3][9] = {
  {0.080852f,0.000000f,0.000000f,0.026854f,0.000000f,0.000000f,6.070598f,0.000000f,0.000000f},
  {0.000000f,0.079358f,0.000000f,0.000000f,0.026281f,0.000000f,0.000000f,6.586221f,0.000000f},
  {0.000000f,0.000000f,0.029929f,0.000000f,0.000000f,0.010262f,0.000000f,0.000000f,0.045315f},
};


static float rtlqr_integral[3];

/**
 * @brief LQR based attitude controller using
 * @param[in] rtkf_X the state estimate (from kalman filter)
 * @param[in] angle_desired the desired rate
 * @param[in] axis which axis to control
 * @returns the control signal for this axis
 */
float rtlqr_angle_calculate_axis(uintptr_t rtkf_handle, float angle_error, uint32_t axis, float dT)
{
	const float * axis_L = La[axis];

	float rates[3];
	float torques[3];
	float bias[3];
	rtkf_get_rate(rtkf_handle, rates);
	rtkf_get_torque(rtkf_handle, torques);
	rtkf_get_bias(rtkf_handle, bias);

	// calculate the desired control signal. Note that there is a negative
	// sign on the state through the rate_error calculation, but this is
	// added explicitly for the torque component (analogous to normal
	// derivative).
	float desired = axis_L[axis] * angle_error                             // "Proportional"
	              - axis_L[axis + 3] * rates[axis]                         // "Rate"
	              - axis_L[axis + 6] * torques[axis]                       // "Derivative"
	              + bias[axis];     // Add estimated bias so calculated output has desired influence

	return desired;
}

/**
 * @brief LQR based controller using
 * @param[in] rtkf_X the state estimate (from kalman filter)
 * @param[in] rate_desired the desired rate
 * @param[in] axis which axis to control
 * @returns the control signal for this axis
 */
float rtlqr_rate_calculate_axis(uintptr_t rtkf_handle, float rate_desired, uint32_t axis, float dT)
{
	const float * axis_L = Lr[axis];

	float rates[3];
	float torques[3];
	float bias[3];
	rtkf_get_rate(rtkf_handle, rates);
	rtkf_get_torque(rtkf_handle, torques);
	rtkf_get_bias(rtkf_handle, bias);

	float rate_error = rate_desired - rates[axis];

	// Update the integral
	rtlqr_integral[axis] = rtlqr_integral[axis] + rate_error * dT;

	// calculate the desired control signal. Note that there is a negative
	// sign on the state through the rate_error calculation, but this is
	// added explicitly for the torque component (analogous to normal
	// derivative).
	float desired = axis_L[axis] * rate_error                                 // "Proportional"
	              - axis_L[axis + 3] * torques[axis]                          // "Derivative"
	              + axis_L[axis + 6] * rtlqr_integral[axis]                   // "Integral"
	              + bias[axis];      // Add estimated bias so calculated output has desired influence

	return desired;
}

void rtlqr_get_integral(float *integral)
{
	integral[0] = rtlqr_integral[0];
	integral[1] = rtlqr_integral[1];
	integral[2] = rtlqr_integral[2];
}

void rtlqr_init()
{
	rtlqr_integral[0] = 0.0f;
	rtlqr_integral[1] = 0.0f;
	rtlqr_integral[2] = 0.0f;
}

void rtlqr_rate_set_roll_gains(const float gains[3])
{
	Lr[0][0] = gains[0];
	Lr[0][3] = gains[1];
	Lr[0][6] = gains[2];
}

void rtlqr_rate_set_pitch_gains(const float gains[3])
{
	Lr[1][1] = gains[0];
	Lr[1][4] = gains[1];
	Lr[1][7] = gains[2];
}

void rtlqr_rate_set_yaw_gains(const float gains[3])
{
	Lr[2][2] = gains[0];
	Lr[2][5] = gains[1];
	Lr[2][8] = gains[2];
}

void rtlqr_angle_set_roll_gains(const float gains[3])
{
	La[0][0] = gains[0];
	La[0][3] = gains[1];
	La[0][6] = gains[2];
}

void rtlqr_angle_set_pitch_gains(const float gains[3])
{
	La[1][1] = gains[0];
	La[1][4] = gains[1];
	La[1][7] = gains[2];
}

void rtlqr_angle_set_yaw_gains(const float gains[3])
{
	La[2][2] = gains[0];
	La[2][5] = gains[1];
	La[2][8] = gains[2];
}

/**
 * @}
 * @}
 */
