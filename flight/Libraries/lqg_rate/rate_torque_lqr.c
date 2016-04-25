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
static float Lr[3][2] = {
  {0.03f, 7.0f},
  {0.03f, 7.0f},
  {0.03f, 0.05f},
};

static float La[3][3] = {
  {0.08f, 0.03f, 7.0f},
  {0.08f, 0.03f, 7.0f},
  {0.03f, 0.01f, 0.05f}
};

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
	float desired = axis_L[0] * angle_error                             // "Proportional"
	              - axis_L[1] * rates[axis]                         // "Rate"
	              - axis_L[2] * torques[axis]                       // "Derivative"
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

	// calculate the desired control signal. Note that there is a negative
	// sign on the state through the rate_error calculation, but this is
	// added explicitly for the torque component (analogous to normal
	// derivative).
	float desired = axis_L[0] * rate_error                                 // "Proportional"
	              - axis_L[1] * torques[axis]                          // "Derivative"
	              + bias[axis];      // Add estimated bias so calculated output has desired influence

	return desired;
}

void rtlqr_init()
{
}

void rtlqr_rate_set_roll_gains(const float gains[2])
{
	Lr[0][0] = gains[0];
	Lr[0][1] = gains[1];
}

void rtlqr_rate_set_pitch_gains(const float gains[2])
{
	Lr[1][0] = gains[0];
	Lr[1][1] = gains[1];
}

void rtlqr_rate_set_yaw_gains(const float gains[2])
{
	Lr[2][0] = gains[0];
	Lr[2][1] = gains[1];
}

void rtlqr_angle_set_roll_gains(const float gains[3])
{
	La[0][0] = gains[0];
	La[0][1] = gains[1];
	La[0][2] = gains[2];
}

void rtlqr_angle_set_pitch_gains(const float gains[3])
{
	La[1][0] = gains[0];
	La[1][1] = gains[1];
	La[1][2] = gains[2];
}

void rtlqr_angle_set_yaw_gains(const float gains[3])
{
	La[2][0] = gains[0];
	La[2][1] = gains[1];
	La[2][2] = gains[2];
}

/**
 * @}
 * @}
 */
