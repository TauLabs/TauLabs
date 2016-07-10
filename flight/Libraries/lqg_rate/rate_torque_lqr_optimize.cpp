/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       rate_torque_lqr_optimize.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      Optimize discrete time LQR controller gain matrix using
 *             system properties measured by system identification
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
static float roll_pitch_cost;
static float yaw_cost;

// params for rate controller
static float rate_cost;
static float torque_cost;
static float yaw_rate_cost;
static float yaw_torque_cost;

// this is for attitude controller
static float attitude_cost;
static float attitude_rate_cost;
static float attitude_torque_cost;

static struct computed_gains {
	float roll_attitude_gains[3];
	float roll_rate_gains[2];
	float pitch_attitude_gains[3];
	float pitch_rate_gains[2];
	float yaw_attitude_gains[3];
	float yaw_rate_gains[2];
} computed_gains;

// timing parameters
static float Ts;
static float tau;
static float ets;

/**
 * @brief rtlqr_init prepare the solver
 * this still reqires the @ref set_tau and @ref set_gains
 * methods to be set at a minimum
 * @param new_Ts the time step this is called at (in seconds)
 */
extern "C" void rtlqro_init(float new_Ts)
{
	Ts = new_Ts;
}

/**
 * @brief rtlqr_set_tau set the time constant from system
 * identification
 * @param[in] tau the time constant ln(seconds)
 */
extern "C" void rtlqro_set_tau(float tau_new)
{
	tau = expf(tau_new);
	ets = expf(-Ts/tau);
}

/**
 * @brief rtlqr_set_gains the channel gains including two
 * for yaw
 * @param[in] gains to be used
 */
extern "C" void rtlqro_set_gains(const float new_gains[4])
{
	for (uint32_t i = 0; i < 4; i++)
		gains[i] = expf(new_gains[i]);
}

/**
 * @brief rtlqr_set_costs set the state and output costs for optimized LQR
 * @param[in] rate_error cost for static rate error
 * @param[in] torque_error cost for having static torque error
 * @param[in] integral_error cost for having accumulated error
 * @param[in] roll_pitch_input cost of using roll or pitch control
 * @param[in] yaw_input cost of using yaw control
 */
extern "C" void rtlqro_set_costs(float attitude_error,
	float attitude_rate_error,
	float attitude_torque_error,
	float rate_error,
	float torque_error,
	float yaw_rate_error,
	float yaw_torque_error)
{
	// These are not user configurable as it is a degenerate set of parameters
	roll_pitch_cost = 1.0f;
	yaw_cost = 1.0f;

	attitude_cost = attitude_error;
	attitude_rate_cost = attitude_rate_error;
	attitude_torque_cost = attitude_torque_error;

	rate_cost = rate_error;
	torque_cost = torque_error;

	yaw_rate_cost = yaw_rate_error;
	yaw_torque_cost = yaw_torque_error;
}

static MXX A;
static MXU B;
static MXX Q;
static MUU R;


static MXX rtlqro_construct_A(float b1, float b2)
{
	MXX A = MXX::Identity();
	A(0,1) = Ts;
	A(0,2) = tau * (b1 - b2) * (Ts  - tau + tau*ets);
	A(1,2) = tau * (b1 - b2) * (1 - ets);
	A(2,2) = ets;

	return A;
}

static MXU rtlqro_construct_B(float b1, float b2)
{
	MXU B = MXU::Constant(0.0f);
	B(0,0) = Ts * Ts * b1 / 2 - tau * (b1 - b2) * (Ts  - tau + tau*ets);
	B(1,0) = Ts * b1 - tau * (b1 - b2) * (1 - ets);
	B(2,0) = 1 - ets;

	return B;
}

static void rtlqro_solver_roll()
{
	MUX K_dlqr;

	// Set up dynamics with roll parameters
	MXX A = rtlqro_construct_A(gains[0], 0);
	MXU B = rtlqro_construct_B(gains[0], 0);

	// Solve for the rate controller
	R(0,0) = roll_pitch_cost;
	Q(0,0) = 1e-5f;  // no integral component for rate is needed
	Q(1,1) = rate_cost;
	Q(2,2) = torque_cost;

	K_dlqr = lqr_gain_solve(A,B,Q,R);
	// K_dlqr(0,0) is integral which is unused for rate controller
	computed_gains.roll_rate_gains[0] = K_dlqr(0,1); // rate term
	computed_gains.roll_rate_gains[1] = K_dlqr(0,2); // torque term

	// Solve for the attitude controller
	Q(0,0) = attitude_cost;
	Q(1,1) = attitude_rate_cost;
	Q(2,2) = attitude_torque_cost;

	K_dlqr = lqr_gain_solve(A,B,Q,R);
	computed_gains.roll_attitude_gains[0] = K_dlqr(0,0); // attitude term
	computed_gains.roll_attitude_gains[1] = K_dlqr(0,1); // rate term
	computed_gains.roll_attitude_gains[2] = K_dlqr(0,2); // torque term
}

static void rtlqro_solver_pitch()
{
	MUX K_dlqr;

	// Set up dynamics with pitch parameters
	MXX A = rtlqro_construct_A(gains[1], 0);
	MXU B = rtlqro_construct_B(gains[1], 0);

	// Solve for the rate controller
	R(0,0) = roll_pitch_cost;
	Q(0,0) = 1e-5f;  // no integral component for rate is needed
	Q(1,1) = rate_cost;
	Q(2,2) = torque_cost;

	K_dlqr = lqr_gain_solve(A,B,Q,R);
	// K_dlqr(0,0) is integral which is unused for rate controller
	computed_gains.pitch_rate_gains[0] = K_dlqr(0,1); // rate term
	computed_gains.pitch_rate_gains[1] = K_dlqr(0,2); // torque term

	// Solve for the attitude controller
	Q(0,0) = attitude_cost;
	Q(1,1) = attitude_rate_cost;
	Q(2,2) = attitude_torque_cost;

	K_dlqr = lqr_gain_solve(A,B,Q,R);
	computed_gains.pitch_attitude_gains[0] = K_dlqr(0,0); // attitude term
	computed_gains.pitch_attitude_gains[1] = K_dlqr(0,1); // rate term
	computed_gains.pitch_attitude_gains[2] = K_dlqr(0,2); // torque term
}

static void rtlqro_solver_yaw()
{
	MUX K_dlqr;

	// Set up dynamics with yaw parameters
	MXX A = rtlqro_construct_A(gains[1], 0);
	MXU B = rtlqro_construct_B(gains[1], 0);


	// Solve for the rate controller
	R(0,0) = yaw_cost;
	Q(0,0) = 1e-5f;  // no integral component for rate is needed
	Q(1,1) = yaw_rate_cost;
	Q(2,2) = yaw_torque_cost;

	K_dlqr = lqr_gain_solve(A,B,Q,R);
	// K_dlqr(0,0) is integral which is unused for rate controller
	computed_gains.yaw_rate_gains[0] = K_dlqr(0,1); // rate term
	computed_gains.yaw_rate_gains[1] = K_dlqr(0,2); // torque term

	// Solve for the attitude controller
	Q(0,0) = attitude_cost;
	Q(1,1) = attitude_rate_cost;
	Q(2,2) = attitude_torque_cost;

	K_dlqr = lqr_gain_solve(A,B,Q,R);
	computed_gains.yaw_attitude_gains[0] = K_dlqr(0,0); // attitude term
	computed_gains.yaw_attitude_gains[1] = K_dlqr(0,1); // rate term
	computed_gains.yaw_attitude_gains[2] = K_dlqr(0,2); // torque term
}

extern "C" void rtlqro_solver()
{
	rtlqro_solver_roll();
	rtlqro_solver_pitch();
	rtlqro_solver_yaw();
}

extern "C" void rtlqro_get_roll_rate_gain(float g[2])
{
	for (uint32_t i = 0; i < 2; i++)
		g[i] = computed_gains.roll_rate_gains[i];
}

extern "C" void rtlqro_get_pitch_rate_gain(float g[2])
{
	for (uint32_t i = 0; i < 2; i++)
		g[i] = computed_gains.pitch_rate_gains[i];
}

extern "C" void rtlqro_get_yaw_rate_gain(float g[2])
{
	for (uint32_t i = 0; i < 2; i++)
		g[i] = computed_gains.yaw_rate_gains[i];
}

extern "C" void rtlqro_get_roll_attitude_gain(float g[3])
{
	for (uint32_t i = 0; i < 3; i++)
		g[i] = computed_gains.roll_attitude_gains[i];
}

extern "C" void rtlqro_get_pitch_attitude_gain(float g[3])
{
	for (uint32_t i = 0; i < 3; i++)
		g[i] = computed_gains.pitch_attitude_gains[i];
}

extern "C" void rtlqro_get_yaw_attitude_gain(float g[3])
{
	for (uint32_t i = 0; i < 3; i++)
		g[i] = computed_gains.yaw_attitude_gains[i];
}

/**
 * @}
 * @}
 */