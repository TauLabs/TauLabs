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

#define AF_NUMX 9
#define AF_NUMP 18

/**** filter parameters ****/
static float q_w = 1e0f;
static float q_ud = 1e-5f;
static float q_bias = 1e-10f;
static float s_a = 1000.0f;  // expected gyro noise
static float gains[4] = {5.f,5.f,5.f,5.f};
static float tau = -5.f;

void rtkf_set_qw(const float qw_new)
{
	q_w = qw_new;
}

void rtkf_set_qu(const float qu_new)
{
	q_ud = qu_new;
}

void rtkf_set_qbias(const float qbias_new)
{
	q_bias = qbias_new;
}

void rtkf_set_sa(const float sa_new)
{
	s_a = sa_new;
}

void rtkf_set_gains(const float gains_new[4])
{
	gains[0] = gains_new[0];
	gains[1] = gains_new[1];
	gains[2] = gains_new[2];
	gains[3] = gains_new[3];
}

void rtkf_set_tau(const float tau_new)
{
	tau = tau_new;
}

/**
 * @param[in] X current state estimate
 * @param[out] rate store the current rate here
 */
void rtkf_get_rate(const float X[AF_NUMX], float rate[3])
{
	rate[0] = X[0];
	rate[1] = X[1];
	rate[2] = X[2];
}

/**
 * @param[in] X current state estimate
 * @param[out] rate store the current rate here
 */
void rtkf_get_torque(const float X[AF_NUMX], float torque[3])
{
	torque[0] = X[3];
	torque[1] = X[4];
	torque[2] = X[5];
}

/**
 * @param[in] X current state estimate
 * @param[out] rate store the current rate here
 */
void rtkf_get_bias(const float X[AF_NUMX], float bias[3])
{
	bias[0] = X[6];
	bias[1] = X[7];
	bias[2] = X[8];
}

/**
 * @brief Run a prediction step for the rate torque KF
 * @param X current state estimate
 * @param P current covariance estimate (active elements)
 * @param[in] u_in control input
 * @param[in] gyro gyro measurement
 * @param[in] gains gains from system identification estimate
 * @param[in] tau the time constant from system identification
 * @param[in] dT_s the time step to advance at
 */
void rtkf_predict(float *X, float *P, const float u_in[3], const float gyro[3], const float dT_s)
{
	const float Ts = dT_s;
	const float Tsq = Ts * Ts;

	// for convenience and clarity code below uses the named versions of
	// the state variables
	float w1 = X[0];           // roll rate estimate
	float w2 = X[1];           // pitch rate estimate
	float w3 = X[2];           // yaw rate estimate
	float u1 = X[3];           // scaled roll torque 
	float u2 = X[4];           // scaled pitch torque
	float u3 = X[5];           // scaled yaw torque
	const float bias1 = X[6];       // bias in the roll torque
	const float bias2 = X[7];       // bias in the pitch torque
	const float bias3 = X[8];       // bias in the yaw torque
	const float e_b1 = expf(gains[0]);   // roll torque scale
	const float e_b2 = expf(gains[1]);   // pitch torque scale
	const float e_b3 = expf(gains[2]);   // yaw torque scale
	const float e_b3d = expf(gains[3]);   // yaw torque scale
	const float e_tau = expf(tau); // time response of the motors

	// inputs to the system (roll, pitch, yaw)
	const float u1_in = u_in[0];
	const float u2_in = u_in[1];
	const float u3_in = u_in[2];

	// measurements from gyro
	const float gyro_x = gyro[0];
	const float gyro_y = gyro[1];
	const float gyro_z = gyro[2];

	// update named variables because we want to use predicted
	// values below
	w1 = X[0] = w1 - Ts*bias1*e_b1 + Ts*u1*e_b1;
	w2 = X[1] = w2 - Ts*bias2*e_b2 + Ts*u2*e_b2;
	w3 = X[2] = w3 - Ts*bias3*e_b3 + Ts*u3*e_b3 + Ts*u3_in*e_b3d;
	u1 = X[3] = (Ts*u1_in)/(Ts + e_tau) + (u1*e_tau)/(Ts + e_tau);
	u2 = X[4] = (Ts*u2_in)/(Ts + e_tau) + (u2*e_tau)/(Ts + e_tau);
	u3 = X[5] = (Ts*u3_in)/(Ts + e_tau) + (u3*e_tau)/(Ts + e_tau);
	// X[6] to X[8] unchanged

	const float Q[AF_NUMX] = {q_w, q_w, q_w, q_ud, q_ud, q_ud, q_bias, q_bias, q_bias};

	float D[AF_NUMP];
	for (uint32_t i = 0; i < AF_NUMP; i++)
		D[i] = P[i];

	//const float e_tau2    = e_tau * e_tau;
	//const float Ts_e_tau2 = (Ts + e_tau) * (Ts + e_tau);
	//const float e_tau_y2    = e_tau_y * e_tau_y;
	//const float Ts_e_tau_y2 = (Ts + e_tau_y) * (Ts + e_tau_y);

	// covariance propagation - D is stored copy of covariance	
	P[0] = D[0] + Q[0] + D[4]*Tsq*(e_b1*e_b1) - 2*D[10]*Tsq*(e_b1*e_b1) + D[11]*Tsq*(e_b1*e_b1) + 2*D[3]*Ts*e_b1 - 2*D[9]*Ts*e_b1;
	P[1] = D[1] + Q[1] + D[6]*Tsq*(e_b2*e_b2) - 2*D[13]*Tsq*(e_b2*e_b2) + D[14]*Tsq*(e_b2*e_b2) + 2*D[5]*Ts*e_b2 - 2*D[12]*Ts*e_b2;
	P[2] = D[2] + Q[2] + D[8]*Tsq*(e_b3*e_b3) - 2*D[16]*Tsq*(e_b3*e_b3) + D[17]*Tsq*(e_b3*e_b3) + 2*D[7]*Ts*e_b3 - 2*D[15]*Ts*e_b3;
	P[3] = (e_tau*(D[3] + D[4]*Ts*e_b1 - D[10]*Ts*e_b1))/(Ts + e_tau);
	P[4] = Q[3] + D[4]*powf(Ts/(Ts + e_tau) - 1,2);
	P[5] = (e_tau*(D[5] + D[6]*Ts*e_b2 - D[13]*Ts*e_b2))/(Ts + e_tau);
	P[6] = Q[4] + D[6]*powf(Ts/(Ts + e_tau) - 1,2);
	P[7] = (e_tau*(D[7] + D[8]*Ts*e_b3 - D[16]*Ts*e_b3))/(Ts + e_tau);
	P[8] = Q[5] + D[8]*powf(Ts/(Ts + e_tau) - 1,2);
	P[9] = D[9] + D[10]*Ts*e_b1 - D[11]*Ts*e_b1;
	P[10] = (D[10]*e_tau)/(Ts + e_tau);
	P[11] = D[11] + Q[6];
	P[12] = D[12] + D[13]*Ts*e_b2 - D[14]*Ts*e_b2;
	P[13] = (D[13]*e_tau)/(Ts + e_tau);
	P[14] = D[14] + Q[7];
	P[15] = D[15] + D[16]*Ts*e_b3 - D[17]*Ts*e_b3;
	P[16] = (D[16]*e_tau)/(Ts + e_tau);
	P[17] = D[17] + Q[8];

	/********* this is the update part of the equation ***********/

	float S[3] = {P[0] + s_a, P[1] + s_a, P[2] + s_a};

	X[0] = w1 + (P[0]*(gyro_x - w1))/S[0];
	X[1] = w2 + (P[1]*(gyro_y - w2))/S[1];
	X[2] = w3 + (P[2]*(gyro_z - w3))/S[2];
	X[3] = u1 + (P[3]*(gyro_x - w1))/S[0];
	X[4] = u2 + (P[5]*(gyro_y - w2))/S[1];
	X[5] = u3 + (P[7]*(gyro_z - w3))/S[2];
	X[6] = bias1 + (P[9]*(gyro_x - w1))/S[0];
	X[7] = bias2 + (P[12]*(gyro_y - w2))/S[1];
	X[8] = bias3 + (P[15]*(gyro_z - w3))/S[2];

	// update the duplicate cache
	for (uint32_t i = 0; i < AF_NUMP; i++)
		D[i] = P[i];

	// This is an approximation that removes some cross axis uncertainty but
	// substantially reduces the number of calculations
	P[0] = -D[0]*(D[0]/S[0] - 1);
	P[1] = -D[1]*(D[1]/S[1] - 1);
	P[2] = -D[2]*(D[2]/S[2] - 1);
	P[3] = -D[3]*(D[0]/S[0] - 1);
	P[4] = D[4] - D[3]*D[3]/S[0];
	P[5] = -D[5]*(D[1]/S[1] - 1);
	P[6] = D[6] - D[5]*D[5]/S[1];
	P[7] = -D[7]*(D[2]/S[2] - 1);
	P[8] = D[8] - D[7]*D[7]/S[2];
	P[9] = -D[9]*(D[0]/S[0] - 1);
	P[10] = D[10] - (D[3]*D[9])/S[0];
	P[11] = D[11] - D[9]*D[9]/S[0];
	P[12] = -D[12]*(D[1]/S[1] - 1);
	P[13] = D[13] - (D[5]*D[12])/S[1];
	P[14] = D[14] - D[12]*D[12]/S[1];
	P[15] = -D[15]*(D[2]/S[2] - 1);
	P[16] = D[16] - (D[7]*D[15])/S[2];
	P[17] = D[17] - D[15]*D[15]/S[2];
}

/**
 * Initialize the state variable and covariance matrix
 * for the rate and torque KF. This also allocates the
 * memory
 * @param[out] X allocate and initialize state
 * @param[out] P allocate and initialize covariance
 * @returns true if successful
 */
bool rtkf_init(float **X_in, float **P_in)
{
	float *X;
	float *P;

	X = (float *) PIOS_malloc(sizeof(float) * AF_NUMX);
	if (X == NULL)
		return false;
	P = (float *) PIOS_malloc(sizeof(float) * AF_NUMP);
	if (P == NULL)
		return false;

	const float q_init[AF_NUMX] = {
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		0.05f, 0.05f, 0.05f
	};

	X[0] = X[1] = X[2] = 0.0f;    // assume no rotation
	X[3] = X[4] = X[5] = 0.0f;    // and no net torque
	X[6] = X[7] = X[8] = 0.0f;    // zero bias

	// P initialization
	// Could zero this like: *P = *((float [AF_NUMP]){});
	P[0] = q_init[0];
	P[1] = q_init[1];
	P[2] = q_init[2];
	P[3] = 0.0f;
	P[4] = q_init[3];
	P[5] = 0.0f;
	P[6] = q_init[4];
	P[7] = 0.0f;
	P[8] = q_init[5];
	P[9] = 0.0f;
	P[10] = 0.0f;
	P[11] = q_init[6];
	P[12] = 0.0f;
	P[13] = 0.0f;
	P[14] = q_init[7];
	P[15] = 0.0f;
	P[16] = 0.0f;
	P[17] = q_init[8];

	*X_in = X;
	*P_in = P;

	return true;
}

/**
 * @}
 * @}
 */

