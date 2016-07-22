/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       rate_torque_si.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      System identication of parameters of this model
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

#include "math.h"
#include "stdint.h"
#include "pios_heap.h"
#include "rate_torque_si.h"

#define RTSI_NUMX 9
#define RTSI_NUMP 29

enum rtsi_state_magic {
  RTSI_STATE_MAGIC = 0x7249993e, // echo "rate_torque_si_2d.c" | md5
};


struct rtsi_state {
	float X[RTSI_NUMX];
	float P[RTSI_NUMP];
	enum rtsi_state_magic magic;
};

bool rtsi_alloc(uintptr_t *rtsi_handle)
{
	struct rtsi_state *rtsi_state = (struct rtsi_state *) PIOS_malloc(sizeof(struct rtsi_state));
	if (rtsi_state == NULL)
		return false;

	rtsi_state->magic = RTSI_STATE_MAGIC;

	rtsi_init((uintptr_t) rtsi_state);

	(*rtsi_handle) = (uintptr_t) rtsi_state;

	return true;
}

bool rtsi_validate(struct rtsi_state *rtsi_state)
{
	if (rtsi_state == NULL)
		return false;

	return (rtsi_state->magic == RTSI_STATE_MAGIC);
}

void rtsi_get_rates(uintptr_t rtsi_handle, float *rates)
{
	struct rtsi_state * rtsi_state= (struct rtsi_state *) rtsi_handle;
	if (!rtsi_validate(rtsi_state))
		return;

	rates[0] = rtsi_state->X[0];
	rates[1] = rtsi_state->X[1];
}

void rtsi_get_gains(uintptr_t rtsi_handle, float *gains)
{
	struct rtsi_state * rtsi_state= (struct rtsi_state *) rtsi_handle;
	if (!rtsi_validate(rtsi_state))
		return;

	gains[0] = rtsi_state->X[4];
	gains[1] = rtsi_state->X[5];
}

void rtsi_get_tau(uintptr_t rtsi_handle, float *tau)
{
	struct rtsi_state * rtsi_state= (struct rtsi_state *) rtsi_handle;
	if (!rtsi_validate(rtsi_state))
		return;

	tau[0] = rtsi_state->X[6];
}

void rtsi_get_bias(uintptr_t rtsi_handle, float *bias)
{
	struct rtsi_state * rtsi_state= (struct rtsi_state *) rtsi_handle;
	if (!rtsi_validate(rtsi_state))
		return;

	bias[0] = rtsi_state->X[7];
	bias[1] = rtsi_state->X[8];
}



 /**
 * Prediction step for EKF on control inputs to quad that
 * learns the system properties
 * @param X the current state estimate which is updated in place
 * @param P the current covariance matrix, updated in place
 * @param[in] the current control inputs (roll, pitch, yaw)
 * @param[in] the gyro measurements
 */
/**
 * Prediction step for EKF on control inputs to quad that
 * learns the system properties
 * @param X the current state estimate which is updated in place
 * @param P the current covariance matrix, updated in place
 * @param[in] the current control inputs (roll, pitch, yaw)
 * @param[in] the gyro measurements
 */
void rtsi_predict(uintptr_t rtsi_handle, const float u_in[3], const float gyro[3], const float dT_s)
{
	struct rtsi_state * rtsi_state= (struct rtsi_state *) rtsi_handle;
	if (!rtsi_validate(rtsi_state))
		return;

	float *X = rtsi_state->X;
	float *P = rtsi_state->P;

	const float Ts = dT_s;
	//const float Tsq = Ts * Ts;
	//const float Tsq3 = Tsq * Ts;
	//const float Tsq4 = Tsq * Tsq;

	// for convenience and clarity code below uses the named versions of
	// the state variables
	float w1 = X[0];           // roll rate estimate
	float w2 = X[1];           // pitch rate estimate
	float u1 = X[2];           // scaled roll torque 
	float u2 = X[3];           // scaled pitch torque
	const float e_b1 = expf(X[4]);   // roll torque scale
	const float b1 = X[4];
	const float e_b2 = expf(X[5]);   // pitch torque scale
	const float b2 = X[5];
	const float e_tau = expf(X[6]); // time response of the motors
	const float tau = X[6];
	const float bias1 = X[7];        // bias in the roll torque
	const float bias2 = X[8];       // bias in the pitch torque

	// inputs to the system (roll, pitch, yaw)
	const float u1_in = u_in[0];
	const float u2_in = u_in[1];

	// measurements from gyro
	const float gyro_x = gyro[0];
	const float gyro_y = gyro[1];

	// update named variables because we want to use predicted
	// values below
	w1 = X[0] = w1 - Ts*bias1*e_b1 + Ts*u1*e_b1;
	w2 = X[1] = w2 - Ts*bias2*e_b2 + Ts*u2*e_b2;
	u1 = X[2] = (Ts*u1_in)/(Ts + e_tau) + (u1*e_tau)/(Ts + e_tau);
	u2 = X[3] = (Ts*u2_in)/(Ts + e_tau) + (u2*e_tau)/(Ts + e_tau);
    // X[5] to X[8] unchanged

	/**** filter parameters ****/
	// core state variables, these were determined from offline analysis and replay of flights
	const float q_w = 1e0f;
	const float q_ud = 1e-5f;
	const float q_bias = 1e-10f;
	const float s_a = 1000.0f;  // expected gyro noise
	// system identification parameters
	const float q_B = 1e-5f;
	const float q_tau = 1e-5f;

	const float Q[RTSI_NUMX] = {q_w, q_w, q_ud, q_ud, q_B, q_B, q_tau, q_bias, q_bias};

	float D[RTSI_NUMP];
	for (uint32_t i = 0; i < RTSI_NUMP; i++)
        D[i] = P[i];

    const float e_tau2    = e_tau * e_tau;
    const float Ts_e_tau2 = (Ts + e_tau) * (Ts + e_tau);
    const float Ts_e_tau3 = Ts_e_tau2 * (Ts + e_tau);
    const float Ts_e_tau4 = Ts_e_tau2 * Ts_e_tau2;

    const float Aeb1 = Ts * e_b1;
    const float Aeb1_2 = Aeb1 * Aeb1;
    const float Aeb2 = Ts * e_b2;
    const float Aeb2_2 = Aeb2 * Aeb2;

	// covariance propagation - D is stored copy of covariance	
	P[0] = (D[3] - 2*D[20] + D[23] - D[7]*(bias1 - u1) + D[21]*(bias1 - u1) + (bias1 - u1)*(D[21] - D[7] + D[8]*(bias1 - u1)))*Aeb1_2 + (2*D[2] - 2*D[19] - 2*D[6]*(bias1 - u1))*Aeb1 + (D[0] + Q[0]);
	P[1] = (D[5] - 2*D[25] + D[28] - D[10]*(bias2 - u2) + D[26]*(bias2 - u2) + (bias2 - u2)*(D[26] - D[10] + D[11]*(bias2 - u2)))*Aeb2_2 + (2*D[4] - 2*D[24] - 2*D[9]*(bias2 - u2))*Aeb2 + (D[1] + Q[1]);
	P[2] = (Ts*e_tau*(u1 - u1_in)*(D[12] + Aeb1*D[14] - Aeb1*D[22] - Aeb1*D[16]*(bias1 - u1)))/Ts_e_tau2 - (Ts/(Ts + e_tau) - 1)*(D[2] + Aeb1*D[3] - Aeb1*D[20] - Aeb1*D[7]*(bias1 - u1));
	P[3] = Q[2] + (e_tau2*(D[3]*Ts + D[3]*e_tau + D[14]*Ts*u1 - D[14]*Ts*u1_in))/Ts_e_tau3 + (Ts*e_tau2*(u1 - u1_in)*(D[14]*Ts + D[14]*e_tau + D[18]*Ts*u1 - D[18]*Ts*u1_in))/Ts_e_tau4;
	P[4] = (Ts*e_tau*(u2 - u2_in)*(D[13] + Aeb2*D[15] - Aeb2*D[27] - Aeb2*D[17]*(bias2 - u2)))/Ts_e_tau2 - (Ts/(Ts + e_tau) - 1)*(D[4] + Aeb2*D[5] - Aeb2*D[25] - Aeb2*D[10]*(bias2 - u2));
	P[5] = Q[3] + (e_tau2*(D[5]*Ts + D[5]*e_tau + D[15]*Ts*u2 - D[15]*Ts*u2_in))/Ts_e_tau3 + (Ts*e_tau2*(u2 - u2_in)*(D[15]*Ts + D[15]*e_tau + D[18]*Ts*u2 - D[18]*Ts*u2_in))/Ts_e_tau4;
	P[6] = (D[7] - D[21] - D[8]*(bias1 - u1))*Aeb1 + D[6];
	P[7] = (e_tau*(D[7]*Ts + D[7]*e_tau + D[16]*Ts*u1 - D[16]*Ts*u1_in))/Ts_e_tau2;
	P[8] = D[8] + Q[4];
	P[9] = D[9] + Aeb2*D[10] - Aeb2*D[26] - Aeb2*D[11]*(bias2 - u2);
	P[10] = (e_tau*(D[10]*Ts + D[10]*e_tau + D[17]*Ts*u2 - D[17]*Ts*u2_in))/Ts_e_tau2;
	P[11] = D[11] + Q[5];
	P[12] = (D[14] - D[22] - D[16]*(bias1 - u1))*Aeb1 + D[12];
	P[13] = (D[15] - D[27] - D[17]*(bias2 - u2))*Aeb2 + D[13];
	P[14] = (e_tau*(D[14]*Ts + D[14]*e_tau + D[18]*Ts*u1 - D[18]*Ts*u1_in))/Ts_e_tau2;
	P[15] = (e_tau*(D[15]*Ts + D[15]*e_tau + D[18]*Ts*u2 - D[18]*Ts*u2_in))/Ts_e_tau2;
	P[16] = D[16];
	P[17] = D[17];
	P[18] = D[18] + Q[6];
	P[19] = (D[20] - D[23] - D[21]*(bias1 - u1))*Aeb1 + D[19];
	P[20] = (e_tau*(D[20]*Ts + D[20]*e_tau + D[22]*Ts*u1 - D[22]*Ts*u1_in))/Ts_e_tau2;
	P[21] = D[21];
	P[22] = D[22];
	P[23] = D[23] + Q[7];
	P[24] = D[24] + Aeb2*D[25] - Aeb2*D[28] - Aeb2*D[26]*(bias2 - u2);
	P[25] = (e_tau*(D[25]*Ts + D[25]*e_tau + D[27]*Ts*u2 - D[27]*Ts*u2_in))/Ts_e_tau2;
	P[26] = D[26];
	P[27] = D[27];
	P[28] = D[28] + Q[8];

    
	/********* this is the update part of the equation ***********/

    float S[2] = {P[0] + s_a, P[1] + s_a,};

	X[0] = w1 + (P[0]*(gyro_x - w1))/S[0];
	X[1] = w2 + (P[1]*(gyro_y - w2))/S[1];
	X[2] = u1 + (P[2]*(gyro_x - w1))/S[0];
	X[3] = u2 + (P[4]*(gyro_y - w2))/S[1];
	X[4] = b1 + (P[6]*(gyro_x - w1))/S[0];
	X[5] = b2 + (P[9]*(gyro_y - w2))/S[1];
	X[6] = tau + (P[12]*(gyro_x - w1))/S[0] + (P[13]*(gyro_y - w2))/S[1];
	X[7] = bias1 + (P[19]*(gyro_x - w1))/S[0];
	X[8] = bias2 + (P[24]*(gyro_y - w2))/S[1];

	// update the duplicate cache
	for (uint32_t i = 0; i < RTSI_NUMP; i++)
        D[i] = P[i];
    
	// This is an approximation that removes some cross axis uncertainty but
	// substantially reduces the number of calculations
	// Covariance calculation
	P[0] = -D[0]*(D[0]/S[0] - 1);
	P[1] = -D[1]*(D[1]/S[1] - 1);
	P[2] = -D[2]*(D[0]/S[0] - 1);
	P[3] = D[3] - D[2]*D[2]/S[0];
	P[4] = -D[4]*(D[1]/S[1] - 1);
	P[5] = D[5] - D[4]*D[4]/S[1];
	P[6] = -D[6]*(D[0]/S[0] - 1);
	P[7] = D[7] - (D[2]*D[6])/S[0];
	P[8] = D[8] - D[6]*D[6]/S[0];
	P[9] = -D[9]*(D[1]/S[1] - 1);
	P[10] = D[10] - (D[4]*D[9])/S[1];
	P[11] = D[11] - D[9]*D[9]/S[1];
	P[12] = -D[12]*(D[0]/S[0] - 1);
	P[13] = -D[13]*(D[1]/S[1] - 1);
	P[14] = D[14] - (D[2]*D[12])/S[0];
	P[15] = D[15] - (D[4]*D[13])/S[1];
	P[16] = D[16] - (D[6]*D[12])/S[0];
	P[17] = D[17] - (D[9]*D[13])/S[1];
	P[18] = D[18] - D[12]*D[12]/S[0] - D[13]*D[13]/S[1];
	P[19] = -D[19]*(D[0]/S[0] - 1);
	P[20] = D[20] - (D[2]*D[19])/S[0];
	P[21] = D[21] - (D[6]*D[19])/S[0];
	P[22] = D[22] - (D[12]*D[19])/S[0];
	P[23] = D[23] - D[19]*D[19]/S[0];
	P[24] = -D[24]*(D[1]/S[1] - 1);
	P[25] = D[25] - (D[4]*D[24])/S[1];
	P[26] = D[26] - (D[9]*D[24])/S[1];
	P[27] = D[27] - (D[13]*D[24])/S[1];
	P[28] = D[28] - D[24]*D[24]/S[1];

}

/**
 * Initialize the state variable and covariance matrix
 * for the system identification EKF
 */
void rtsi_init(uintptr_t rtsi_handle)
{
	struct rtsi_state * rtsi_state= (struct rtsi_state *) rtsi_handle;
	if (!rtsi_validate(rtsi_state))
		return;

	float *X = rtsi_state->X;
	float *P = rtsi_state->P;

	const float q_init[RTSI_NUMX] = {
		1.0f, 1.0f,
		1.0f, 1.0f,
		0.05f, 0.05f,
		0.05f,
		0.05f, 0.05f
	};

	X[0] = X[1] = 0.0f;           // assume no rotation
	X[2] = X[3] = 0.0f;           // and no net torque
	X[4] = X[5] = 10.0f;          // medium amount of strength
	X[6] = -4.0f;                 // and 50 ms time scale
	X[7] = X[8] = 0.0f;           // zero bias

	// P initialization
	// Could zero this like: *P = *((float [AF_NUMP]){});
	P[0] = q_init[0];
	P[1] = q_init[1];
	P[2] = 0.0f;
	P[3] = q_init[2];
	P[4] = 0.0f;
	P[5] = q_init[3];
	P[6] = 0.0f;
	P[7] = 0.0f;
	P[8] = q_init[4];
	P[9] = 0.0f;
	P[10] = 0.0f;
	P[11] = q_init[5];
	P[12] = 0.0f;
	P[13] = 0.0f;
	P[14] = 0.0f;
	P[15] = 0.0f;
	P[16] = 0.0f;
	P[17] = 0.0f;
	P[18] = q_init[6];
	P[19] = 0.0f;
	P[20] = 0.0f;
	P[21] = 0.0f;
	P[22] = 0.0f;
	P[23] = q_init[7];
	P[24] = 0.0f;
	P[25] = 0.0f;
	P[26] = 0.0f;
	P[27] = 0.0f;
	P[28] = q_init[8];
}

/**
 * @}
 * @}
 */

