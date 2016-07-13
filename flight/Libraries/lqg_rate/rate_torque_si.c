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

#define RTSI_NUMX 14
#define RTSI_NUMP 37

enum rtsi_state_magic {
  RTSI_STATE_MAGIC = 0xa6a83bd9, // echo "rate_torque_si.c" | md5
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
	rates[1] = rtsi_state->X[3];
	rates[2] = rtsi_state->X[6];
}

void rtsi_get_torque(uintptr_t rtsi_handle, float *torque)
{
	struct rtsi_state * rtsi_state= (struct rtsi_state *) rtsi_handle;
	if (!rtsi_validate(rtsi_state))
		return;

	torque[0] = rtsi_state->X[1];
	torque[1] = rtsi_state->X[4];
	torque[2] = rtsi_state->X[7];
}

void rtsi_get_bias(uintptr_t rtsi_handle, float *bias)
{
	struct rtsi_state * rtsi_state= (struct rtsi_state *) rtsi_handle;
	if (!rtsi_validate(rtsi_state))
		return;

	bias[0] = rtsi_state->X[2];
	bias[1] = rtsi_state->X[5];
	bias[2] = rtsi_state->X[8];
}

void rtsi_get_gains(uintptr_t rtsi_handle, float *gains)
{
	struct rtsi_state * rtsi_state= (struct rtsi_state *) rtsi_handle;
	if (!rtsi_validate(rtsi_state))
		return;

	gains[0] = rtsi_state->X[9];
	gains[1] = rtsi_state->X[10];
	gains[2] = rtsi_state->X[11];
	gains[3] = rtsi_state->X[12];
}

void rtsi_get_tau(uintptr_t rtsi_handle, float *tau)
{
	struct rtsi_state * rtsi_state= (struct rtsi_state *) rtsi_handle;
	if (!rtsi_validate(rtsi_state))
		return;

	tau[0] = rtsi_state->X[13];
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

	// for convenience and clarity code below uses the named versions of
	// the state variables
	float wr = X[0];           // roll rate estimate
	float wp = X[3];           // pitch rate estimate
	float wy = X[6];           // yaw rate estimate
	float nur = X[1];           // scaled roll torque 
	float nup = X[4];           // scaled pitch torque
	float nuy = X[7];           // scaled yaw torque
	float biasr = X[2];       // bias in the roll torque
	float biasp = X[5];       // bias in the pitch torque
	float biasy = X[8];       // bias in the yaw torque

	const float e_br = expf(X[9]);   // roll torque scale
	const float br = X[9];
	const float e_bp = expf(X[10]);   // pitch torque scale
	const float bp = X[10];
	const float e_by1 = expf(X[11]);   // yaw torque scale
	const float by1 = X[11];
	const float e_by2 = expf(X[12]);
	const float by2 = X[12];
	const float e_tau = expf(X[13]); // time response of the motors
	const float tau = X[13];

	const float ets = expf(-Ts/e_tau);

	// inputs to the system (roll, pitch, yaw)
	const float ur = u_in[0];
	const float up = u_in[1];
	const float uy = u_in[2];

	// measurements from gyro
	const float gyro_x = gyro[0];
	const float gyro_y = gyro[1];
	const float gyro_z = gyro[2];

	// update named variables because we want to use predicted
	// values below
	wr  = X[0] = wr + ur*(Ts*e_br - e_br*e_tau + e_br*e_tau*ets) + nur*(e_br*e_tau - e_br*e_tau*ets) - biasr*(Ts*e_br - e_br*e_tau + e_br*e_tau*ets);
	nur = X[1] = ets*nur + biasr*(ets - 1) - ur*(ets - 1);
	wp  = X[3] = wp + up*(Ts*e_bp - e_bp*e_tau + e_bp*e_tau*ets) + nup*(e_bp*e_tau - e_bp*e_tau*ets) - biasp*(Ts*e_bp - e_bp*e_tau + e_bp*e_tau*ets);
	nup = X[4] = ets*nup + biasp*(ets - 1) - up*(ets - 1);
	wy  = X[6] = wy + nuy*(e_by1*e_tau - e_by2*e_tau - e_by1*e_tau*ets + e_by2*e_tau*ets) - biasy*(Ts*e_by1 - e_by1*e_tau + e_by2*e_tau + e_by1*e_tau*ets - e_by2*e_tau*ets) + uy*(Ts*by2 + Ts*e_by1 - Ts*e_by2 - e_by1*e_tau + e_by2*e_tau + e_by1*e_tau*ets - e_by2*e_tau*ets);
	nuy = X[7] = ets*nuy + biasy*(ets - 1) - uy*(ets - 1);

	/**** filter parameters ****/
	// core state variables, these were determined from offline analysis and replay of flights
	const float q_w = 1e0f;
	const float q_ud = 1e-5f;
	const float q_bias = 1e-10f;
	const float s_a = 1000.0f;  // expected gyro noise
	// system identification parameters
	const float q_B = 1e-5f;
	const float q_tau = 1e-5f;

	const float Q[RTSI_NUMX] = {q_w, q_ud, q_bias, q_w, q_ud, q_bias, q_w, q_ud, q_bias, q_B, q_B, q_B, q_B, q_tau};

	float D[RTSI_NUMP];
	for (uint32_t i = 0; i < RTSI_NUMP; i++)
        D[i] = P[i];

	// covariance propagation - D is stored copy of covariance	
	P[0] = D[0] + Q[0] - e_br*(Ts - e_tau + e_tau*ets)*(D[3] - D[5]*Ts*e_br + D[4]*e_br*e_tau + D[5]*e_br*e_tau - D[4]*e_br*e_tau*ets - D[5]*e_br*e_tau*ets) - D[18]*e_br*(Ts*biasr - Ts*ur + e_tau*(ets - 1)*(biasr + nur - ur)) + e_br*(Ts*biasr - Ts*ur + e_tau*(ets - 1)*(biasr + nur - ur))*(D[20]*e_br*(Ts*biasr - Ts*ur + e_tau*(ets - 1)*(biasr + nur - ur)) - D[18] + D[19]*e_br*e_tau*(ets - 1)) - D[3]*e_br*(Ts - e_tau + e_tau*ets) + e_br*(biasr + nur - ur)*(Ts*ets - e_tau + e_tau*ets)*(D[31]*e_br*e_tau*(ets - 1) - D[30] + D[36]*e_br*(biasr + nur - ur)*(Ts*ets - e_tau + e_tau*ets)) - D[1]*e_br*e_tau*(ets - 1) - D[30]*e_br*(biasr + nur - ur)*(Ts*ets - e_tau + e_tau*ets) + e_br*e_tau*(ets - 1)*(D[19]*e_br*(Ts*biasr - Ts*ur + e_tau*(ets - 1)*(biasr + nur - ur)) - D[1] + D[4]*e_br*(Ts - e_tau + e_tau*ets) + D[2]*e_br*e_tau*(ets - 1) + D[31]*e_br*(biasr + nur - ur)*(Ts*ets - e_tau + e_tau*ets));
	P[1] = - (ets - 1)*(D[5]*e_br*(Ts - e_tau + e_tau*ets) - D[3] + D[4]*e_br*e_tau*(ets - 1)) - ets*(D[19]*e_br*(Ts*biasr - Ts*ur + e_tau*(ets - 1)*(biasr + nur - ur)) - D[1] + D[4]*e_br*(Ts - e_tau + e_tau*ets) + D[2]*e_br*e_tau*(ets - 1) + D[31]*ets*(biasr + nur - ur)*(Ts*e_br + e_br*e_tau - (e_br*e_tau)/ets)) - (Ts*ets*(biasr + nur - ur)*(D[31]*e_br*e_tau*(ets - 1) - D[30] + D[36]*ets*(biasr + nur - ur)*(Ts*e_br + e_br*e_tau - (e_br*e_tau)/ets)))/e_tau;
	P[2] = Q[1] + (ets - 1)*(D[4]*ets + D[5]*(ets - 1)) + ets*(D[2]*ets + D[4]*(ets - 1) + (D[31]*Ts*ets*(biasr + nur - ur))/e_tau) + (Ts*ets*(D[31]*ets + (D[36]*Ts*ets*(biasr + nur - ur))/e_tau)*(biasr + nur - ur))/e_tau;
	P[3] = D[3] - D[5]*e_br*(Ts - e_tau + e_tau*ets) - D[4]*e_br*e_tau*(ets - 1);
	P[4] = D[4]*ets + D[5]*(ets - 1);
	P[5] = D[5] + Q[2];
	P[6] = D[6] + Q[3] - e_bp*(Ts - e_tau + e_tau*ets)*(D[9] - D[11]*Ts*e_bp + D[10]*e_bp*e_tau + D[11]*e_bp*e_tau - D[10]*e_bp*e_tau*ets - D[11]*e_bp*e_tau*ets) - D[21]*e_bp*(Ts*biasp - Ts*up + e_tau*(ets - 1)*(biasp + nup - up)) + e_bp*(Ts*biasp - Ts*up + e_tau*(ets - 1)*(biasp + nup - up))*(D[23]*e_bp*(Ts*biasp - Ts*up + e_tau*(ets - 1)*(biasp + nup - up)) - D[21] + D[22]*e_bp*e_tau*(ets - 1)) - D[9]*e_bp*(Ts - e_tau + e_tau*ets) + e_bp*(biasp + nup - up)*(Ts*ets - e_tau + e_tau*ets)*(D[33]*e_bp*e_tau*(ets - 1) - D[32] + D[36]*e_bp*(biasp + nup - up)*(Ts*ets - e_tau + e_tau*ets)) - D[7]*e_bp*e_tau*(ets - 1) - D[32]*e_bp*(biasp + nup - up)*(Ts*ets - e_tau + e_tau*ets) + e_bp*e_tau*(ets - 1)*(D[22]*e_bp*(Ts*biasp - Ts*up + e_tau*(ets - 1)*(biasp + nup - up)) - D[7] + D[10]*e_bp*(Ts - e_tau + e_tau*ets) + D[8]*e_bp*e_tau*(ets - 1) + D[33]*e_bp*(biasp + nup - up)*(Ts*ets - e_tau + e_tau*ets));
	P[7] = - (ets - 1)*(D[11]*e_bp*(Ts - e_tau + e_tau*ets) - D[9] + D[10]*e_bp*e_tau*(ets - 1)) - ets*(D[22]*e_bp*(Ts*biasp - Ts*up + e_tau*(ets - 1)*(biasp + nup - up)) - D[7] + D[10]*e_bp*(Ts - e_tau + e_tau*ets) + D[8]*e_bp*e_tau*(ets - 1) + D[33]*ets*(biasp + nup - up)*(Ts*e_bp + e_bp*e_tau - (e_bp*e_tau)/ets)) - (Ts*ets*(biasp + nup - up)*(D[33]*e_bp*e_tau*(ets - 1) - D[32] + D[36]*ets*(biasp + nup - up)*(Ts*e_bp + e_bp*e_tau - (e_bp*e_tau)/ets)))/e_tau;
	P[8] = Q[4] + (ets - 1)*(D[10]*ets + D[11]*(ets - 1)) + ets*(D[8]*ets + D[10]*(ets - 1) + (D[33]*Ts*ets*(biasp + nup - up))/e_tau) + (Ts*ets*(D[33]*ets + (D[36]*Ts*ets*(biasp + nup - up))/e_tau)*(biasp + nup - up))/e_tau;
	P[9] = D[9] - D[11]*e_bp*(Ts - e_tau + e_tau*ets) - D[10]*e_bp*e_tau*(ets - 1);
	P[10] = D[10]*ets + D[11]*(ets - 1);
	P[11] = D[11] + Q[5];
	P[12] = D[12] + Q[6] + (Ts*uy*(e_by2 - 1) - e_by2*e_tau*(ets - 1)*(biasy + nuy - uy))*(D[29]*(Ts*uy*(e_by2 - 1) - e_by2*e_tau*(ets - 1)*(biasy + nuy - uy)) - D[27] + D[28]*e_tau*(e_by1 - e_by2)*(ets - 1)) + (D[17]*(Ts*e_by1 - e_by1*e_tau + e_by2*e_tau + e_by1*e_tau*ets - e_by2*e_tau*ets) - D[15] + D[16]*e_tau*(e_by1 - e_by2)*(ets - 1))*(Ts*e_by1 - e_by1*e_tau + e_by2*e_tau + e_by1*e_tau*ets - e_by2*e_tau*ets) - D[15]*(Ts*e_by1 - e_by1*e_tau + e_by2*e_tau + e_by1*e_tau*ets - e_by2*e_tau*ets) - D[27]*(Ts*uy*(e_by2 - 1) - e_by2*e_tau*(ets - 1)*(biasy + nuy - uy)) - D[24]*e_by1*(Ts*biasy - Ts*uy + e_tau*(ets - 1)*(biasy + nuy - uy)) + e_by1*(Ts*biasy - Ts*uy + e_tau*(ets - 1)*(biasy + nuy - uy))*(D[26]*e_by1*(Ts*biasy - Ts*uy + e_tau*(ets - 1)*(biasy + nuy - uy)) - D[24] + D[25]*e_tau*(e_by1 - e_by2)*(ets - 1)) - D[13]*e_tau*(e_by1 - e_by2)*(ets - 1) + (e_by1 - e_by2)*(biasy + nuy - uy)*(D[35]*e_tau*(e_by1 - e_by2)*(ets - 1) - D[34] + D[36]*(e_by1 - e_by2)*(biasy + nuy - uy)*(Ts*ets - e_tau + e_tau*ets))*(Ts*ets - e_tau + e_tau*ets) - D[34]*(e_by1 - e_by2)*(biasy + nuy - uy)*(Ts*ets - e_tau + e_tau*ets) + e_tau*(e_by1 - e_by2)*(ets - 1)*(D[16]*(Ts*e_by1 - e_by1*e_tau + e_by2*e_tau + e_by1*e_tau*ets - e_by2*e_tau*ets) - D[13] + D[28]*(Ts*uy*(e_by2 - 1) - e_by2*e_tau*(ets - 1)*(biasy + nuy - uy)) + D[25]*e_by1*(Ts*biasy - Ts*uy + e_tau*(ets - 1)*(biasy + nuy - uy)) + D[14]*e_tau*(e_by1 - e_by2)*(ets - 1) + D[35]*(e_by1 - e_by2)*(biasy + nuy - uy)*(Ts*ets - e_tau + e_tau*ets));
	P[13] = - (ets - 1)*(D[17]*(Ts*e_by1 - e_by1*e_tau + e_by2*e_tau + e_by1*e_tau*ets - e_by2*e_tau*ets) - D[15] + D[16]*e_tau*(e_by1 - e_by2)*(ets - 1)) - ets*(D[16]*(Ts*e_by1 - e_by1*e_tau + e_by2*e_tau + e_by1*e_tau*ets - e_by2*e_tau*ets) - D[13] + D[28]*(Ts*uy*(e_by2 - 1) - e_by2*e_tau*(ets - 1)*(biasy + nuy - uy)) + D[25]*e_by1*(Ts*biasy - Ts*uy + e_tau*(ets - 1)*(biasy + nuy - uy)) + D[14]*e_tau*(e_by1 - e_by2)*(ets - 1) + D[35]*(e_by1 - e_by2)*(biasy + nuy - uy)*(Ts*ets - e_tau + e_tau*ets)) - (Ts*ets*(biasy + nuy - uy)*(D[35]*e_tau*(e_by1 - e_by2)*(ets - 1) - D[34] + D[36]*(e_by1 - e_by2)*(biasy + nuy - uy)*(Ts*ets - e_tau + e_tau*ets)))/e_tau;
	P[14] = Q[7] + (ets - 1)*(D[16]*ets + D[17]*(ets - 1)) + ets*(D[14]*ets + D[16]*(ets - 1) + (D[35]*Ts*ets*(biasy + nuy - uy))/e_tau) + (Ts*ets*(D[35]*ets + (D[36]*Ts*ets*(biasy + nuy - uy))/e_tau)*(biasy + nuy - uy))/e_tau;
	P[15] = D[15] - D[17]*(Ts*e_by1 - e_by1*e_tau + e_by2*e_tau + e_by1*e_tau*ets - e_by2*e_tau*ets) - D[16]*e_tau*(e_by1 - e_by2)*(ets - 1);
	P[16] = D[16]*ets + D[17]*(ets - 1);
	P[17] = D[17] + Q[8];
	P[18] = D[18] - D[20]*e_br*(Ts*biasr - Ts*ur + e_tau*(ets - 1)*(biasr + nur - ur)) - D[19]*e_br*e_tau*(ets - 1);
	P[19] = D[19]*ets;
	P[20] = D[20] + Q[9];
	P[21] = D[21] - D[23]*e_bp*(Ts*biasp - Ts*up + e_tau*(ets - 1)*(biasp + nup - up)) - D[22]*e_bp*e_tau*(ets - 1);
	P[22] = D[22]*ets;
	P[23] = D[23] + Q[10];
	P[24] = D[24] - D[26]*e_by1*(Ts*biasy - Ts*uy + e_tau*(ets - 1)*(biasy + nuy - uy)) - D[25]*e_tau*(e_by1 - e_by2)*(ets - 1);
	P[25] = D[25]*ets;
	P[26] = D[26] + Q[11];
	P[27] = D[27] - D[29]*(Ts*uy*(e_by2 - 1) - e_by2*e_tau*(ets - 1)*(biasy + nuy - uy)) - D[28]*e_tau*(e_by1 - e_by2)*(ets - 1);
	P[28] = D[28]*ets;
	P[29] = D[29] + Q[12];
	P[30] = D[30] - D[31]*e_br*e_tau*(ets - 1) - D[36]*e_br*(biasr + nur - ur)*(Ts*ets - e_tau + e_tau*ets);
	P[31] = D[31]*ets + (D[36]*Ts*ets*(biasr + nur - ur))/e_tau;
	P[32] = D[32] - D[33]*e_bp*e_tau*(ets - 1) - D[36]*e_bp*(biasp + nup - up)*(Ts*ets - e_tau + e_tau*ets);
	P[33] = D[33]*ets + (D[36]*Ts*ets*(biasp + nup - up))/e_tau;
	P[34] = D[34] - D[35]*e_tau*(e_by1 - e_by2)*(ets - 1) - D[36]*(e_by1 - e_by2)*(biasy + nuy - uy)*(Ts*ets - e_tau + e_tau*ets);
	P[35] = D[35]*ets + (D[36]*Ts*ets*(biasy + nuy - uy))/e_tau;
	P[36] = D[36] + Q[13];
    
	/********* this is the update part of the equation ***********/

    float S[3] = {P[0] + s_a, P[6] + s_a, P[12] + s_a};

	X[0] = wr + (P[0]*(gyro_x - wr))/S[0];
	X[1] = nur + (P[1]*(gyro_x - wr))/S[0];
	X[2] = biasr + (P[3]*(gyro_x - wr))/S[0];
	X[3] = wp + (P[6]*(gyro_y - wp))/S[1];
	X[4] = nup + (P[7]*(gyro_y - wp))/S[1];
	X[5] = biasp + (P[9]*(gyro_y - wp))/S[1];
	X[6] = wy + (P[12]*(gyro_z - wy))/S[2];
	X[7] = nuy + (P[13]*(gyro_z - wy))/S[2];
	X[8] = biasy + (P[15]*(gyro_z - wy))/S[2];
	X[9] = br + (P[18]*(gyro_x - wr))/S[0];
	X[10] = bp + (P[21]*(gyro_y - wp))/S[1];
	X[11] = by1 + (P[24]*(gyro_z - wy))/S[2];
	X[12] = by2 + (P[27]*(gyro_z - wy))/S[2];
	X[13] = tau + (P[32]*(gyro_y - wp))/S[1] + (P[30]*(gyro_x - wr))/S[0] + (P[34]*(gyro_z - wy))/S[2];

	// update the duplicate cache
	for (uint32_t i = 0; i < RTSI_NUMP; i++)
        D[i] = P[i];
    
	// This is an approximation that removes some cross axis uncertainty but
	// substantially reduces the number of calculations
	// Covariance calculation
	P[0] = -D[0]*(D[0]/S[0] - 1);
	P[1] = -D[1]*(D[0]/S[0] - 1);
	P[2] = D[2] - D[1]*D[1]/S[0];
	P[3] = -D[3]*(D[0]/S[0] - 1);
	P[4] = D[4] - (D[1]*D[3])/S[0];
	P[5] = D[5] - D[3]*D[3]/S[0];
	P[6] = -D[6]*(D[6]/S[1] - 1);
	P[7] = -D[7]*(D[6]/S[1] - 1);
	P[8] = D[8] - D[7]*D[7]/S[1];
	P[9] = -D[9]*(D[6]/S[1] - 1);
	P[10] = D[10] - (D[7]*D[9])/S[1];
	P[11] = D[11] - D[9]*D[9]/S[1];
	P[12] = -D[12]*(D[12]/S[2] - 1);
	P[13] = -D[13]*(D[12]/S[2] - 1);
	P[14] = D[14] - D[13]*D[13]/S[2];
	P[15] = -D[15]*(D[12]/S[2] - 1);
	P[16] = D[16] - (D[13]*D[15])/S[2];
	P[17] = D[17] - D[15]*D[15]/S[2];
	P[18] = -D[18]*(D[0]/S[0] - 1);
	P[19] = D[19] - (D[1]*D[18])/S[0];
	P[20] = D[20] - D[18]*D[18]/S[0];
	P[21] = -D[21]*(D[6]/S[1] - 1);
	P[22] = D[22] - (D[7]*D[21])/S[1];
	P[23] = D[23] - D[21]*D[21]/S[1];
	P[24] = -D[24]*(D[12]/S[2] - 1);
	P[25] = D[25] - (D[13]*D[24])/S[2];
	P[26] = D[26] - D[24]*D[24]/S[2];
	P[27] = -D[27]*(D[12]/S[2] - 1);
	P[28] = D[28] - (D[13]*D[27])/S[2];
	P[29] = D[29] - D[27]*D[27]/S[2];
	P[30] = -D[30]*(D[0]/S[0] - 1);
	P[31] = D[31] - (D[1]*D[30])/S[0];
	P[32] = -D[32]*(D[6]/S[1] - 1);
	P[33] = D[33] - (D[7]*D[32])/S[1];
	P[34] = -D[34]*(D[12]/S[2] - 1);
	P[35] = D[35] - (D[13]*D[34])/S[2];
	P[36] = D[36] - D[30]*D[30]/S[0] - D[32]*D[32]/S[1] - D[34]*D[34]/S[2];
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
		1.0f, 1.0f, 1e-5f,
		1.0f, 1.0f, 1e-5f,
		1.0f, 1.0f, 1e-5f,
		0.05f, 0.05f, 0.05f, 0.05f, 0.05f,
	};

	X[0] = X[1] = X[2] = 0.0f;    // roll state
	X[3] = X[4] = X[5] = 0.0f;    // pitch state
	X[6] = X[7] = X[8] = 0.0f;    // yaw state
	X[9]  = X[10]       = 10.0f;  // medium amount of strength
	X[11] = X[12]       = 7.0f;   // yaw gains
	X[13] = -4.0f;                // and 50 ms time scale

	// P initialization
	// Could zero this like: *P = *((float [AF_NUMP]){});
	P[0] = q_init[0];
	P[1] = 0.0f;
	P[2] = q_init[1];
	P[3] = 0.0f;
	P[4] = 0.0f;
	P[5] = q_init[2];
	P[6] = q_init[3];
	P[7] = 0.0f;
	P[8] = q_init[4];
	P[9] = 0.0f;
	P[10] = 0.0f;
	P[11] = q_init[5];
	P[12] = q_init[6];
	P[13] = 0.0f;
	P[14] = q_init[7];
	P[15] = 0.0f;
	P[16] = 0.0f;
	P[17] = q_init[8];
	P[18] = 0.0f;
	P[19] = 0.0f;
	P[20] = q_init[9];
	P[21] = 0.0f;
	P[22] = 0.0f;
	P[23] = q_init[10];
	P[24] = 0.0f;
	P[25] = 0.0f;
	P[26] = q_init[11];
	P[27] = 0.0f;
	P[28] = 0.0f;
	P[29] = q_init[12];
	P[30] = 0.0f;
	P[31] = 0.0f;
	P[32] = 0.0f;
	P[33] = 0.0f;
	P[34] = 0.0f;
	P[35] = 0.0f;
	P[36] = q_init[13];
}

/**
 * @}
 * @}
 */

