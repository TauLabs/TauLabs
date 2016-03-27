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

#include "rate_torque_lqr.h"

// This is the LQR tuning matrix. This is currently calculated statically
// based on the system dynamics (measured with system identification) and
// the LQR cost matricies. Note all the zeros because there is no cross
// axis cost (at least in this implementation).
static const float L[3][9] = {
  {0.030625f,0.000000f,0.000000f,7.666974f,0.000000f,0.000000f,0.260407f,0.000000f,0.000000f},
  {0.000000f,0.030992f,0.000000f,0.000000f,7.388925f,0.000000f,0.000000f,0.262287f,0.000000f},
  {0.000000f,0.000000f,0.077515f,0.000000f,0.000000f,0.963848f,0.000000f,0.000000f,0.308677f},
};

static float rtlqr_integral[3];

/**
 * @brief LQR based controller using
 * @param[in] rtkf_X the state estimate (from kalman filter)
 * @param[in] rate_desired the desired rate
 * @param[in] axis which axis to control
 * @returns the control signal for this axis
 */
float rtlqr_calculate_axis(const float *rtkf_X, float rate_desired, uint32_t axis, float dT)
{
	const float * axis_L = L[axis];
	float rate_error = rate_desired - rtkf_X[axis];

	// Update the integral
	rtlqr_integral[axis] = rtlqr_integral[axis] + rate_error * dT;

	// calculate the desired control signal. Note that there is a negative
	// sign on the state through the rate_error calculation, but this is
	// added explicitly for the torque component (analogous to normal
	// derivative).
	float desired = axis_L[axis] * rate_error                                 // "Proportional"
	              - axis_L[axis + 3] * (rtkf_X[axis + 3] - rtkf_X[axis + 6])  // "Derivative"
	              + axis_L[axis + 6] * rtlqr_integral[axis];                  // "Integral"

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
/**
 * @}
 * @}
 */
