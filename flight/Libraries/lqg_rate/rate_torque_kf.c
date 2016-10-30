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

enum rtkf_state_magic {
  RTKF_STATE_MAGIC = 0x17b0a57c, // echo "rate_torque_kf.c" | md5
};

struct rtkf_state {

	float Ts;      // The kalman filter time step
	float tau_s;

	float init_bias[3];

	float roll_X[AF_NUMX];
	float pitch_X[AF_NUMX];
	float yaw_X[AF_NUMX];

	// intermediate parameters for the kalman filter
	float ets;
	float roll_ad12;
	float roll_ad13;
	float pitch_ad12;
	float pitch_ad13;
	float yaw_ad12;
	float yaw_ad13;

	// the calculated feedback gains for the kalman filter
	float kalman_gains[3][3];

	enum rtkf_state_magic magic;
};

bool rtkf_alloc(uintptr_t *rtkf_handle)
{
	struct rtkf_state *rtkf_state = (struct rtkf_state *) PIOS_malloc(sizeof(struct rtkf_state));
	if (rtkf_state == NULL)
		return false;

	rtkf_state->magic = RTKF_STATE_MAGIC;

	rtkf_state->init_bias[0] = 0.0f;
	rtkf_state->init_bias[1] = 0.0f;
	rtkf_state->init_bias[2] = 0.0f;

	rtkf_state->tau_s = 0.03f;
	rtkf_state->Ts = 1.0f/400.0f;

	// These will be calculated either offline or prior to flight
	float K1 = 0.0439f;
	float K2 = 0.0002f;
	float K3 = -0.0001f;
	rtkf_state->kalman_gains[0][0] = K1;
	rtkf_state->kalman_gains[0][1] = K2;
	rtkf_state->kalman_gains[0][2] = K3;
	rtkf_state->kalman_gains[1][0] = K1;
	rtkf_state->kalman_gains[1][1] = K2;
	rtkf_state->kalman_gains[1][2] = K3;
	rtkf_state->kalman_gains[2][0] = K1;
	rtkf_state->kalman_gains[2][1] = K2;
	rtkf_state->kalman_gains[2][2] = K3;

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

void rtkf_set_roll_kalman_gain(uintptr_t rtkf_handle, const float kg[3])
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rtkf_state->kalman_gains[0][0] = kg[0];
	rtkf_state->kalman_gains[0][1] = kg[1];
	rtkf_state->kalman_gains[0][2] = kg[2];
}

void rtkf_set_pitch_kalman_gain(uintptr_t rtkf_handle, const float kg[3])
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rtkf_state->kalman_gains[1][0] = kg[0];
	rtkf_state->kalman_gains[1][1] = kg[1];
	rtkf_state->kalman_gains[1][2] = kg[2];
}

void rtkf_set_yaw_kalman_gain(uintptr_t rtkf_handle, const float kg[3])
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rtkf_state->kalman_gains[2][0] = kg[0];
	rtkf_state->kalman_gains[2][1] = kg[1];
	rtkf_state->kalman_gains[2][2] = kg[2];
}

/**
 * @param[in] rtkf_handle handle for estimation
 * @param[in] tau the new tau parameter
 */
void rtkf_set_tau(uintptr_t rtkf_handle, const float tau_new, const float Ts_new)
{
	struct rtkf_state * rtkf_state = (struct rtkf_state *) rtkf_handle;
	if (!rtkf_validate(rtkf_state))
		return;

	rtkf_state->Ts = Ts_new;
	rtkf_state->tau_s = expf(tau_new);
	rtkf_state->ets = expf(-rtkf_state->Ts / rtkf_state->tau_s);
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

	// Handy variables (must be set properly before calling)
	const float Ts = rtkf_state->Ts;
	const float tau = rtkf_state->tau_s;
	const float ets = rtkf_state->ets;

	rtkf_state->roll_ad12 = expf(gains_new[0]) * (tau - tau * ets);
	rtkf_state->roll_ad13 = -Ts * expf(gains_new[0]) + rtkf_state->roll_ad12;
	rtkf_state->pitch_ad12 = expf(gains_new[1]) * (tau - tau * ets);
	rtkf_state->pitch_ad13 = -Ts * expf(gains_new[1]) + rtkf_state->pitch_ad12;
	rtkf_state->yaw_ad12 = (expf(gains_new[2]) - expf(gains_new[3]))* (tau - tau * ets);
	rtkf_state->yaw_ad13 = -Ts * expf(gains_new[2]) + rtkf_state->yaw_ad12;
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

	float u_in;
	float gyro;
	const float ets = rtkf_state->ets;

	// Update state for each axis
	for (uint32_t i = 0; i < 3; i++) {
		float *X;
		float *L;
		float Ad12;
		float Ad13;

		switch(i) {
		case 0:       // roll
			X = rtkf_state->roll_X;
			Ad12 = rtkf_state->roll_ad12;
			Ad13 = rtkf_state->roll_ad13;
			L = rtkf_state->kalman_gains[0];
			u_in = control_in[0];
			gyro = gyros[0];
			break;
		case 1:       // pitch
			X = rtkf_state->pitch_X;
			Ad12 = rtkf_state->pitch_ad12;
			Ad13 = rtkf_state->pitch_ad13;
			L = rtkf_state->kalman_gains[1];
			u_in = control_in[1];
			gyro = gyros[1];
			break;
		case 2:       // yaw
			X = rtkf_state->yaw_X;
			Ad12 = rtkf_state->yaw_ad12;
			Ad13 = rtkf_state->yaw_ad13;
			L = rtkf_state->kalman_gains[2];
			u_in = control_in[2];
			gyro = gyros[2];
			break;
		}

		const float w = X[0];
		const float u = X[1];
		const float bias = X[2];

		// Advance the state based on the natural dynamics and input
		X[0] = w + Ad12 * u + Ad13 * bias - Ad13 * u_in;

		// Calculate the error and correct state. Note that we do this in one step
		// below as the error does not depend on these predictions
		const float err = (gyro-w);

		X[0] = X[0] + L[0] * err;
		X[1] = u * ets + (ets - 1) * bias + (1-ets) * u_in + L[1] * err;
		X[2] = bias + L[2] * err;
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

	for (uint32_t i = 0; i < 3; i++) {
		float *X;

		switch(i) {
		case 0:       // roll
			X = rtkf_state->roll_X;
			break;
		case 1:       // pitch
			X = rtkf_state->pitch_X;
			break;
		case 2:       // yaw
			X = rtkf_state->yaw_X;
			break;
		}

		X[0] = 0.0f;  // init no rotation
		X[1] = 0.0f;  // init no torque
		X[2] = rtkf_state->init_bias[i];  // use initiation bias
	}

	return true;
}


/**
 * @}
 * @}
 */

