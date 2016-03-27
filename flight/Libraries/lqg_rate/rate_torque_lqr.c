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
static const float L[3][6] = {
  {0.012523f,0.000000f,0.000000f,4.827413f,0.000000f,0.000000f},
  {0.000000f,0.012583f,0.000000f,0.000000f,4.639445f,0.000000f},
  {0.000000f,0.000000f,0.013997f,0.000000f,0.000000f,0.410062f},
};

/**
 * @brief LQR based controller using
 * @param[in] rtkf_X the state estimate (from kalman filter)
 * @param[in] rate_desired the desired rate
 * @param[in] axis which axis to control
 * @returns the control signal for this axis
 */
float rtlqr_calculate_axis(const float *rtkf_X, float rate_desired, uint32_t axis)
{
	const float * axis_L = L[axis];
	float rate_error = rate_desired - rtkf_X[axis];

	// calculate the desired control signal. Note that there is a negative
	// sign on the state through the rate_error calculation, but this is
	// added explicitly for the torque component (analogous to normal
	// derivative).
	float desired = axis_L[axis] * rate_error - axis_L[axis + 3] * (rtkf_X[axis + 3] - rtkf_X[axis + 6]);

	return desired;
}

/**
 * @}
 * @}
 */
