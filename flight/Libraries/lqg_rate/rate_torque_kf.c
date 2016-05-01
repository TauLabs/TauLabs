/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       rate_torque_kf.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      Kalman filter to estimate rate and torque of motors
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

#include "pios_heap.h"
#include "math.h"
#include "stdint.h"

#include "rate_torque_kf.h"

#define AF_NUMX 3
#define AF_NUMP 6

enum rtkf_state_magic {
  RTKF_STATE_MAGIC = 0x17b0a57c, // echo "rate_torque_kf.c" | md5
};

struct rtkf_state {
	float q_w;
	float q_ud;
	float q_bias;
	float s_a;
	float tau;
	float gains[4];
	float init_bias[3];
	float roll_X[AF_NUMX];
	float roll_P[AF_NUMP];
	float pitch_X[AF_NUMX];
	float pitch_P[AF_NUMP];
	float yaw_X[AF_NUMX];
	float yaw_P[AF_NUMP];
	enum rtkf_state_magic magic;
};

bool rtkf_alloc(uintptr_t *rtkf_handle)
{
	struct rtkf_state *rtkf_state = (struct rtkf_state *) PIOS_malloc(sizeof(struct rtkf_state));
	if (rtkf_state == NULL)
		return false;

	rtkf_state->magic = RTKF_STATE_MAGIC;

	// Use reasonable defaults
	rtkf_state->q_w = 1e0f;
	rtkf_state->q_ud = 1e-5f;
	rtkf_state->q_bias = 1e-7f;
	rtkf_state->s_a = 1000.0f;
	rtkf_state->gains[0] = 5.f;
	rtkf_state->gains[1] = 5.f;
	rtkf_state->gains[2] = 5.f;
	rtkf_state->gains[3] = 5.f;
	rtkf_state->tau = -5.f;

	rtkf_init((uintptr_t) rtkf_state);

	(*rtkf_handle) = (uintptr_t) rtkf_state;

	return true;
}

bool rtkf_validate(struct rtkf_state *rtkf_state)
{
	if (rtkf_state == NULL)
		return false;

	return (rtkf_state->magic == RTKF_STATE_MAGIC);
}

/**
 * @param[in] rtkf_handle handle for estimation
 * @param[in] qw process noise in the rate estimate
 */
void rtkf_set_qw(uintptr_t rtkf_handle, const float qw_new)
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rtkf_state->q_w = qw_new;
}

/**
 * @param[in] rtkf_handle handle for estimation
 * @param[in] qu process noise in the torque estimate
 */
void rtkf_set_qu(uintptr_t rtkf_handle, const float qu_new)
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rtkf_state->q_ud = qu_new;
}

/**
 * @param[in] rtkf_handle handle for estimation
 * @param[in] qbias process noise in the bias estimate
 */
void rtkf_set_qbias(uintptr_t rtkf_handle, const float qbias_new)
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rtkf_state->q_bias = qbias_new;
}

/**
 * @param[in] rtkf_handle handle for estimation
 * @param[in] sa the gyro noise variance
 */
void rtkf_set_sa(uintptr_t rtkf_handle, const float sa_new)
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rtkf_state->s_a = sa_new;
}

/**
 * @param[in] rtkf_handle handle for estimation
 * @param[in] gains the new gains
 */
void rtkf_set_gains(uintptr_t rtkf_handle, const float gains_new[4])
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rtkf_state->gains[0] = gains_new[0];
	rtkf_state->gains[1] = gains_new[1];
	rtkf_state->gains[2] = gains_new[2];
	rtkf_state->gains[3] = gains_new[3];
}

/**
 * @param[in] rtkf_handle handle for estimation
 * @param[in] tau the new tau parameter
 */
void rtkf_set_tau(uintptr_t rtkf_handle, const float tau_new)
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rtkf_state->tau = tau_new;
}

/**
 * @param[in] rtkf_handle handle for estimation
 * @param[out] rate store the current rate here
 */
void rtkf_get_rate(uintptr_t rtkf_handle, float rate[3])
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rate[0] = rtkf_state->roll_X[0];
	rate[1] = rtkf_state->pitch_X[0];
	rate[2] = rtkf_state->yaw_X[0];
}

/**
 * @param[in] rtkf_handle handle for estimation
 * @param[out] rate store the current rate here
 */
void rtkf_get_torque(uintptr_t rtkf_handle, float torque[3])
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	torque[0] = rtkf_state->roll_X[1];
	torque[1] = rtkf_state->pitch_X[1];
	torque[2] = rtkf_state->yaw_X[1];
}

/**
 * @param[in] rtkf_handle handle for estimation
 * @param[out] rate store the current rate here
 */
void rtkf_get_bias(uintptr_t rtkf_handle, float bias[3])
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	bias[0] = rtkf_state->roll_X[2];
	bias[1] = rtkf_state->pitch_X[2];
	bias[2] = rtkf_state->yaw_X[2];
}

/**
 * @brief Run a prediction step for the rate torque KF
 * @param[in] rtkf_handle handle for estimation
 * @param[in] u_in control input
 * @param[in] gyro gyro measurement
 * @param[in] gains gains from system identification estimate
 * @param[in] tau the time constant from system identification
 * @param[in] dT_s the time step to advance at
 */
void rtkf_predict(uintptr_t rtkf_handle, float throttle, const float control_in[3], const float gyros[3], const float dT_s)
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	const float Ts = dT_s;
	const float Tsq = Ts*Ts;

	const float q_w = rtkf_state->q_w;
	const float q_ud = rtkf_state->q_ud;
	const float q_bias = rtkf_state->q_bias;
	const float s_a = rtkf_state->s_a;

	const float tau = rtkf_state->tau;
	const float e_tau = expf(tau); // time response of the motors
	const float e_tau2 = e_tau*e_tau;
	const float Ts_e_tau2 = (Ts + e_tau) * (Ts + e_tau);

	float u_in;
	float gyro;

	float b;
	float bd;

	// Update state for each axis
	for (uint32_t i = 0; i < 3; i++) {
		float *X;
		float *P;

		switch(i) {
		case 0:       // roll
			X = rtkf_state->roll_X;
			P = rtkf_state->roll_P;
			b = rtkf_state->gains[0];
			bd = -100.0f;
			u_in = control_in[0];
			gyro = gyros[0];
			break;
		case 1:       // pitch
			X = rtkf_state->pitch_X;
			P = rtkf_state->pitch_P;
			b = rtkf_state->gains[1];
			bd = -100.0f;
			u_in = control_in[1];
			gyro = gyros[1];
			break;
		case 2:       // yaw
			X = rtkf_state->yaw_X;
			P = rtkf_state->yaw_P;
			b = rtkf_state->gains[2];
			bd = rtkf_state->gains[3];
			u_in = control_in[2];
			gyro = gyros[2];
			break;
		}

		float w = X[0];
		float u = X[1];
		float bias = X[2];

		const float e_b = expf(b);
		const float e_bd = expf(bd);

		// X update
		w = X[0] = w - Ts*bias*e_bd + Ts*u*e_b + Ts*u_in*e_bd;
		u = X[1] = (Ts*u_in)/(Ts + e_tau) - (Ts*bias)/(Ts + e_tau) + (u*e_tau)/(Ts + e_tau);

		const float Q[AF_NUMX] = {q_w, q_ud, q_bias};

		float D[AF_NUMP];
		for (uint32_t i = 0; i < AF_NUMP; i++)
			D[i] = P[i];

		// Covariance calculation
		P[0] = D[0] + Q[0] + D[2]*Tsq*(e_b*e_b) + D[5]*Tsq*(e_bd*e_bd) + 2*D[1]*Ts*e_b - 2*D[3]*Ts*e_bd - 2*D[4]*Tsq*e_b*e_bd;
		P[1] = - (Ts/(Ts + e_tau) - 1)*(D[1] + D[2]*Ts*e_b - D[4]*Ts*e_bd) - (Ts*(D[3] + D[4]*Ts*e_b - D[5]*Ts*e_bd))/(Ts + e_tau);
		P[2] = (D[5]*Tsq + Q[1]*Tsq + D[2]*e_tau2 + Q[1]*e_tau2 - 2*D[4]*Ts*e_tau + 2*Q[1]*Ts*e_tau)/Ts_e_tau2;
		P[3] = D[3] + D[4]*Ts*e_b - D[5]*Ts*e_bd;
		P[4] = -(D[5]*Ts - D[4]*e_tau)/(Ts + e_tau);
		P[5] = D[5] + Q[2];
	
		/********* this is the update part of the equation ***********/
		const float S = P[0] + s_a;

		// X update
		X[0] = w + (P[0]*(gyro - w))/S;
		X[1] = u + (P[1]*(gyro - w))/S;
		X[2] = bias + (P[3]*(gyro - w))/S;

		// If throttle is low don't allow bias or torque to wind up
		if (throttle <= 0) {
			X[1] = 0;
			X[2] = rtkf_state->init_bias[i];
		}

		for (uint32_t i = 0; i < AF_NUMP; i++)
			D[i] = P[i];

		// Covariance calculation
		P[0] = -D[0]*(D[0]/S - 1);
		P[1] = -D[1]*(D[0]/S - 1);
		P[2] = D[2] - D[1]*D[1]/S;
		P[3] = -D[3]*(D[0]/S - 1);
		P[4] = D[4] - (D[1]*D[3])/S;
		P[5] = D[5] - D[3]*D[3]/S;
	}

}

void rtkf_set_init_bias(uintptr_t rtkf_handle, const float bias[3])
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rtkf_state->init_bias[0] = bias[0];
	rtkf_state->init_bias[1] = bias[1];
	rtkf_state->init_bias[2] = bias[2];
}

/**
 * Initialize the state variable and covariance matrix
 * for the rate and torque KF. This also allocates the
 * memory
 * @param[in] rtkf_handle handle for estimation
 * @returns true if successful
 */
bool rtkf_init(uintptr_t rtkf_handle)
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return false;

	const float q_init[AF_NUMX] = {1.0f, 1.0f, 0.05f};

	for (uint32_t i = 0; i < 3; i++) {
		float *X;
		float *P;

		switch(i) {
		case 0:       // roll
			X = rtkf_state->roll_X;
			P = rtkf_state->roll_P;
			break;
		case 1:       // pitch
			X = rtkf_state->pitch_X;
			P = rtkf_state->pitch_P;
			break;
		case 2:       // yaw
			X = rtkf_state->yaw_X;
			P = rtkf_state->yaw_P;
			break;
		}

		X[0] = 0.0f;  // init no rotation
		X[1] = 0.0f;  // init no torque
		X[2] = 0.0f;  // init no bias

		// P initialization
		P[0] = q_init[0];
		P[1] = 0.0f;
		P[2] = q_init[1];
		P[3] = 0.0f;
		P[4] = 0.0f;
		P[5] = q_init[2];
	}

	return true;
}


/**
 * @}
 * @}
 */

