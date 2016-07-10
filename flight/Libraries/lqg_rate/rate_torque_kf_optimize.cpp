/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       rate_torque_kf_optimize.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      Calculate static kalman gains using system properties measured
 *             by system identification
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

#include "dare.h"

using Eigen::Matrix;

// one gain for roll and pitch and two for yaw
static float gains[4];

// timing parameters
static float Ts;
static float tau;
static float ets;

// params for rate controller
static float process_noise[3];
static float gyro_noise;

// Results
static struct computed_gains {
	float roll[3];
	float pitch[3];
	float yaw[3];
} computed_gains;

/**
 * @brief rtlqr_init prepare the solver
 * this still reqires the @ref set_tau and @ref set_gains
 * methods to be set at a minimum
 * @param new_Ts the time step this is called at (in seconds)
 */
extern "C" void rtkfo_init(float new_Ts)
{
	Ts = new_Ts;
}

/**
 * @brief rtlqr_set_tau set the time constant from system
 * identification
 * @param[in] tau the time constant ln(seconds)
 */
extern "C" void rtkfo_set_tau(float tau_new)
{
	tau = expf(tau_new);
	ets = expf(-Ts/tau);
}

/**
 * @brief rtlqr_set_gains the channel gains including two
 * for yaw
 * @param[in] gains to be used
 */
extern "C" void rtkfo_set_gains(const float new_gains[4])
{
	for (uint32_t i = 0; i < 4; i++)
		gains[i] = expf(new_gains[i]);
}

/**
 * @brief rtkfo_set_noise set the noise parameters for the
 * kalman filter
 * @param[in] q the process noise
 * @param[in] g the gyro measurement noise
 */
extern "C" void rtkfo_set_noise(float *q, float g)
{
	process_noise[0] = q[0];
	process_noise[1] = q[1];
	process_noise[2] = q[2];
	gyro_noise = g;
}

static MXX rtlqro_construct_A(float b1, float b2)
{
	MXX A = MXX::Identity();
	A(0,1) = (b1 - b2) * (tau - tau*ets);
	A(0,2) = -Ts*b1 + A(0,1);
	A(1,1) = ets;
	A(1,2) = ets-1;

	return A;
}

static MXU rtlqro_construct_B(float b1, float b2)
{
	MXU B = MXU::Constant(0.0f);
	B(0,0) = Ts*b1 - (b1 - b2) * (tau - tau*ets);
	B(1,0) = 1 - ets;
	B(2,0) = 0;

	return B;
}

extern "C" void rtkfo_solver()
{
	MXX A;
	MXU B;
	MXX Q = MXX::Constant(0.0f);
	MUU R = MUU::Constant(gyro_noise);

	MXU gain;

	Q(0,0) = process_noise[0];
	Q(1,1) = process_noise[1];
	Q(2,2) = process_noise[2];

	// Solve for roll
	A = rtlqro_construct_A(gains[0], 0);
	B = rtlqro_construct_B(gains[0], 0);
	gain = kalman_gain_solve(A, B, Q, R);
	computed_gains.roll[0] = gain(0,0);
	computed_gains.roll[1] = gain(1,0);
	computed_gains.roll[2] = gain(2,0);

	// Solve for pitch
	A = rtlqro_construct_A(gains[1], 0);
	B = rtlqro_construct_B(gains[1], 0);
	gain = kalman_gain_solve(A, B, Q, R);
	computed_gains.pitch[0] = gain(0,0);
	computed_gains.pitch[1] = gain(1,0);
	computed_gains.pitch[2] = gain(2,0);

	// Solve for yaw
	A = rtlqro_construct_A(gains[2], gains[3]);
	B = rtlqro_construct_B(gains[2], gains[3]);
	gain = kalman_gain_solve(A, B, Q, R);
	computed_gains.yaw[0] = gain(0,0);
	computed_gains.yaw[1] = gain(1,0);
	computed_gains.yaw[2] = gain(2,0);
}

extern "C" void rtkfo_get_roll_gain(float g[3])
{
	g[0] = computed_gains.roll[0];
	g[1] = computed_gains.roll[1];
	g[2] = computed_gains.roll[2];
}

extern "C" void rtkfo_get_pitch_gain(float g[3])
{
	g[0] = computed_gains.pitch[0];
	g[1] = computed_gains.pitch[1];
	g[2] = computed_gains.pitch[2];
}

extern "C" void rtkfo_get_yaw_gain(float g[3])
{
	g[0] = computed_gains.yaw[0];
	g[1] = computed_gains.yaw[1];
	g[2] = computed_gains.yaw[2];
}

/**
 * @}
 * @}
 */