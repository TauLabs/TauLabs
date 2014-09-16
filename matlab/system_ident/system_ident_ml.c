#include "math.h"
#include "mex.h"   
#include "string.h"
#include "stdint.h"
#include "stdbool.h"

#define AF_NUMX 13
#define AF_NUMP 43
void af_predict(float X[AF_NUMX], float P[AF_NUMP], float u_in[3], float gyro[3], float noise[3]);
void af_init(float X[AF_NUMX], float P[AF_NUMP]);
bool mlGetFloatArray(const mxArray * mlVal, float * dest, int numel);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // calling convention is [state, p] = autotune_c_imp(state, p, gyro, control)
	float X[AF_NUMX];
    float P[AF_NUMP];
    float control[3];
    float gyro[3];
    float noise[3];
    
    if (nrhs == 0) {
        af_init(X, P);
    } else if (nrhs == 5) {
        if (!mlGetFloatArray(prhs[0], X, AF_NUMX))
            mexErrMsgTxt("Invalid state vector\n");
        
        if (!mlGetFloatArray(prhs[1], P, AF_NUMP))
            mexErrMsgTxt("Invalid covariance array\n");
        
        if (!mlGetFloatArray(prhs[2], gyro, 3))
            mexErrMsgTxt("Invalid control vector\n");

        if (!mlGetFloatArray(prhs[3], control, 3))
            mexErrMsgTxt("Invalid gyro vector\n");
        
        if (!mlGetFloatArray(prhs[4], noise, 3))
            mexErrMsgTxt("Invalid gyro vector\n");
        
        af_predict(X, P, control, gyro, noise);

    } else {
        mexErrMsgTxt("Incorrect number of inputs for state prediction\n");
        
    }

    
    // Return data
    if(nlhs > 0) {
		// return current state vector
		double * data_out;
		int i;

		plhs[0] = mxCreateDoubleMatrix(1,AF_NUMX,0);
		data_out = mxGetData(plhs[0]);
		for(i = 0; i < AF_NUMX; i++)
			data_out[i] = X[i];
	}

	if(nlhs > 1) {
		//return covariance estimate
		double * data_copy = mxCalloc(AF_NUMP, sizeof(double));
		int i, j, k;

		plhs[1] = mxCreateDoubleMatrix(1,AF_NUMP,0);
		for(i = 0; i < AF_NUMP; i++)
            data_copy[i] = P[i];
			
		mxSetData(plhs[1], data_copy);
	}
    
    if(nlhs > 2) {
		// return current state vector
		double * data_out;
		int i;

		plhs[2] = mxCreateDoubleMatrix(1,3,0);
		data_out = mxGetData(plhs[2]);
		for(i = 0; i < 3; i++)
			data_out[i] = noise[i];
	}

}

bool mlGetFloatArray(const mxArray * mlVal, float * dest, int numel)
{
	if(!mxIsNumeric(mlVal) || (!mxIsDouble(mlVal) && !mxIsSingle(mlVal)) || (mxGetNumberOfElements(mlVal) != numel)) {
		mexErrMsgTxt("Data misformatted (either not double or not the right number)");
		return false;
	}
	
	if(mxIsSingle(mlVal)) {
		memcpy(dest,mxGetData(mlVal),numel*sizeof(*dest));
	} else {
		int i;
		double * data_in = mxGetData(mlVal);
		for(i = 0; i < numel; i++)
			dest[i] = data_in[i]; 
	}

	return true;
}

/**
 * Prediction step for EKF on control inputs to quad that
 * learns the system properties
 * @param X the current state estimate which is updated in place
 * @param P the current covariance matrix, updated in place
 * @param[in] the current control inputs (roll, pitch, yaw)
 * @param[in] the gyro measurements
 */
void af_predict(float X[AF_NUMX], float P[AF_NUMP], float u_in[3], float gyro[3], float noise[3]) {

	float Ts = 1.0f / 666.0f;
	float Tsq = Ts * Ts;
	float Tsq3 = Tsq * Ts;
    float Tsq4 = Tsq * Tsq;

	// for convenience and clarity code below uses the named versions of
	// the state variables
	float w1 = X[0];           // roll rate estimate
	float w2 = X[1];           // pitch rate estimate
	float w3 = X[2];           // yaw rate estimate
	float u1 = X[3];           // scaled roll torque 
	float u2 = X[4];           // scaled pitch torque
	float u3 = X[5];           // scaled yaw torque
	const float e_b1 = expf(X[6]);   // roll torque scale
    const float b1 = X[6];
	const float e_b2 = expf(X[7]);   // pitch torque scale
    const float b2 = X[7];
	const float e_b3 = expf(X[8]);   // yaw torque scale
    const float b3 = X[8];
	const float e_tau = expf(X[9]); // time response of the motors
    const float tau = X[9];
	const float bias1 = X[10];        // bias in the roll torque
	const float bias2 = X[11];       // bias in the pitch torque
	const float bias3 = X[12];       // bias in the yaw torque

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
	w3 = X[2] = w3 - Ts*bias3*e_b3 + Ts*u3*e_b3;
	u1 = X[3] = (Ts*u1_in)/(Ts + e_tau) + (u1*e_tau)/(Ts + e_tau);
	u2 = X[4] = (Ts*u2_in)/(Ts + e_tau) + (u2*e_tau)/(Ts + e_tau);
	u3 = X[5] = (Ts*u3_in)/(Ts + e_tau) + (u3*e_tau)/(Ts + e_tau);
    // X[6] to X[12] unchanged

	/**** filter parameters ****/
	const float q_w = 1e-4f;
	const float q_ud = 1e-4f;
	const float q_B = 1e-5f;
	const float q_tau = 1e-5f;
	const float q_bias = 1e-19f;
	const float s_a = 3000.0f;  // expected gyro noise
    const float s_a2 = s_a * s_a;
    const float s_a3 = s_a2 * s_a;

	const float Q[AF_NUMX] = {q_w, q_w, q_w, q_ud, q_ud, q_ud, q_B, q_B, q_B, q_tau, q_bias, q_bias, q_bias};

	float D[AF_NUMP];
	for (uint32_t i = 0; i < AF_NUMP; i++)
        D[i] = P[i];

    const float e_tau2    = e_tau * e_tau;
    const float e_tau3    = e_tau * e_tau2;
    const float e_tau4    = e_tau2 * e_tau2;
    const float Ts_e_tau2 = (Ts + e_tau) * (Ts + e_tau);
    const float Ts_e_tau4 = Ts_e_tau2 * Ts_e_tau2;

	// covariance propagation - D is stored copy of covariance	
	P[0] = D[0] + Q[0] + 2*Ts*e_b1*(D[3] - D[28] - D[9]*bias1 + D[9]*u1) + Tsq*(e_b1*e_b1)*(D[4] - 2*D[29] + D[32] - 2*D[10]*bias1 + 2*D[30]*bias1 + 2*D[10]*u1 - 2*D[30]*u1 + D[11]*(bias1*bias1) + D[11]*(u1*u1) - 2*D[11]*bias1*u1);
	P[1] = D[1] + Q[1] + 2*Ts*e_b2*(D[5] - D[33] - D[12]*bias2 + D[12]*u2) + Tsq*(e_b2*e_b2)*(D[6] - 2*D[34] + D[37] - 2*D[13]*bias2 + 2*D[35]*bias2 + 2*D[13]*u2 - 2*D[35]*u2 + D[14]*(bias2*bias2) + D[14]*(u2*u2) - 2*D[14]*bias2*u2);
	P[2] = D[2] + Q[2] + 2*Ts*e_b3*(D[7] - D[38] - D[15]*bias3 + D[15]*u3) + Tsq*(e_b3*e_b3)*(D[8] - 2*D[39] + D[42] - 2*D[16]*bias3 + 2*D[40]*bias3 + 2*D[16]*u3 - 2*D[40]*u3 + D[17]*(bias3*bias3) + D[17]*(u3*u3) - 2*D[17]*bias3*u3);
	P[3] = (D[3]*e_tau2 + D[3]*Ts*e_tau + D[4]*Ts*(e_b1*e_tau2) - D[29]*Ts*(e_b1*e_tau2) + D[4]*Tsq*(e_b1*e_tau) - D[29]*Tsq*(e_b1*e_tau) + D[18]*Ts*u1*e_tau - D[18]*Ts*u1_in*e_tau - D[10]*Ts*bias1*(e_b1*e_tau2) - D[10]*Tsq*bias1*(e_b1*e_tau) + D[10]*Ts*u1*(e_b1*e_tau2) + D[10]*Tsq*u1*(e_b1*e_tau) + D[21]*Tsq*u1*(e_b1*e_tau) - D[31]*Tsq*u1*(e_b1*e_tau) - D[21]*Tsq*u1_in*(e_b1*e_tau) + D[31]*Tsq*u1_in*(e_b1*e_tau) + D[24]*Tsq*(u1*u1)*(e_b1*e_tau) - D[24]*Tsq*bias1*u1*(e_b1*e_tau) + D[24]*Tsq*bias1*u1_in*(e_b1*e_tau) - D[24]*Tsq*u1*u1_in*(e_b1*e_tau))/Ts_e_tau2;
	P[4] = (Q[3]*Tsq4 + D[4]*e_tau4 + Q[3]*e_tau4 + 2*D[4]*Ts*e_tau3 + 4*Q[3]*Ts*e_tau3 + 4*Q[3]*Tsq3*e_tau + D[4]*Tsq*e_tau2 + 6*Q[3]*Tsq*e_tau2 + 2*D[21]*Tsq*u1*e_tau2 - 2*D[21]*Tsq*u1_in*e_tau2 + D[27]*Tsq*(u1*u1)*e_tau2 + D[27]*Tsq*(u1_in*u1_in)*e_tau2 + 2*D[21]*Ts*u1*e_tau3 - 2*D[21]*Ts*u1_in*e_tau3 - 2*D[27]*Tsq*u1*u1_in*e_tau2)/Ts_e_tau4;
	P[5] = (D[5]*e_tau2 + D[5]*Ts*e_tau + D[6]*Ts*(e_b2*e_tau2) - D[34]*Ts*(e_b2*e_tau2) + D[6]*Tsq*(e_b2*e_tau) - D[34]*Tsq*(e_b2*e_tau) + D[19]*Ts*u2*e_tau - D[19]*Ts*u2_in*e_tau - D[13]*Ts*bias2*(e_b2*e_tau2) - D[13]*Tsq*bias2*(e_b2*e_tau) + D[13]*Ts*u2*(e_b2*e_tau2) + D[13]*Tsq*u2*(e_b2*e_tau) + D[22]*Tsq*u2*(e_b2*e_tau) - D[36]*Tsq*u2*(e_b2*e_tau) - D[22]*Tsq*u2_in*(e_b2*e_tau) + D[36]*Tsq*u2_in*(e_b2*e_tau) + D[25]*Tsq*(u2*u2)*(e_b2*e_tau) - D[25]*Tsq*bias2*u2*(e_b2*e_tau) + D[25]*Tsq*bias2*u2_in*(e_b2*e_tau) - D[25]*Tsq*u2*u2_in*(e_b2*e_tau))/Ts_e_tau2;
	P[6] = (Q[4]*Tsq4 + D[6]*e_tau4 + Q[4]*e_tau4 + 2*D[6]*Ts*e_tau3 + 4*Q[4]*Ts*e_tau3 + 4*Q[4]*Tsq3*e_tau + D[6]*Tsq*e_tau2 + 6*Q[4]*Tsq*e_tau2 + 2*D[22]*Tsq*u2*e_tau2 - 2*D[22]*Tsq*u2_in*e_tau2 + D[27]*Tsq*(u2*u2)*e_tau2 + D[27]*Tsq*(u2_in*u2_in)*e_tau2 + 2*D[22]*Ts*u2*e_tau3 - 2*D[22]*Ts*u2_in*e_tau3 - 2*D[27]*Tsq*u2*u2_in*e_tau2)/Ts_e_tau4;
	P[7] = (D[7]*e_tau2 + D[7]*Ts*e_tau + D[8]*Ts*(e_b3*e_tau2) - D[39]*Ts*(e_b3*e_tau2) + D[8]*Tsq*(e_b3*e_tau) - D[39]*Tsq*(e_b3*e_tau) + D[20]*Ts*u3*e_tau - D[20]*Ts*u3_in*e_tau - D[16]*Ts*bias3*(e_b3*e_tau2) - D[16]*Tsq*bias3*(e_b3*e_tau) + D[16]*Ts*u3*(e_b3*e_tau2) + D[16]*Tsq*u3*(e_b3*e_tau) + D[23]*Tsq*u3*(e_b3*e_tau) - D[41]*Tsq*u3*(e_b3*e_tau) - D[23]*Tsq*u3_in*(e_b3*e_tau) + D[41]*Tsq*u3_in*(e_b3*e_tau) + D[26]*Tsq*(u3*u3)*(e_b3*e_tau) - D[26]*Tsq*bias3*u3*(e_b3*e_tau) + D[26]*Tsq*bias3*u3_in*(e_b3*e_tau) - D[26]*Tsq*u3*u3_in*(e_b3*e_tau))/Ts_e_tau2;
	P[8] = (Q[5]*Tsq4 + D[8]*e_tau4 + Q[5]*e_tau4 + 2*D[8]*Ts*e_tau3 + 4*Q[5]*Ts*e_tau3 + 4*Q[5]*Tsq3*e_tau + D[8]*Tsq*e_tau2 + 6*Q[5]*Tsq*e_tau2 + 2*D[23]*Tsq*u3*e_tau2 - 2*D[23]*Tsq*u3_in*e_tau2 + D[27]*Tsq*(u3*u3)*e_tau2 + D[27]*Tsq*(u3_in*u3_in)*e_tau2 + 2*D[23]*Ts*u3*e_tau3 - 2*D[23]*Ts*u3_in*e_tau3 - 2*D[27]*Tsq*u3*u3_in*e_tau2)/Ts_e_tau4;
	P[9] = D[9] - Ts*(D[30]*e_b1 - D[10]*e_b1 + D[11]*e_b1*(bias1 - u1));
	P[10] = (e_tau*(D[10]*Ts + D[10]*e_tau + D[24]*Ts*u1 - D[24]*Ts*u1_in))/Ts_e_tau2;
	P[11] = D[11] + Q[6];
	P[12] = D[12] - Ts*(D[35]*e_b2 - D[13]*e_b2 + D[14]*e_b2*(bias2 - u2));
	P[13] = (e_tau*(D[13]*Ts + D[13]*e_tau + D[25]*Ts*u2 - D[25]*Ts*u2_in))/Ts_e_tau2;
	P[14] = D[14] + Q[7];
	P[15] = D[15] - Ts*(D[40]*e_b3 - D[16]*e_b3 + D[17]*e_b3*(bias3 - u3));
	P[16] = (e_tau*(D[16]*Ts + D[16]*e_tau + D[26]*Ts*u3 - D[26]*Ts*u3_in))/Ts_e_tau2;
	P[17] = D[17] + Q[8];
	P[18] = D[18] - Ts*(D[31]*e_b1 - D[21]*e_b1 + D[24]*e_b1*(bias1 - u1));
	P[19] = D[19] - Ts*(D[36]*e_b2 - D[22]*e_b2 + D[25]*e_b2*(bias2 - u2));
	P[20] = D[20] - Ts*(D[41]*e_b3 - D[23]*e_b3 + D[26]*e_b3*(bias3 - u3));
	P[21] = (e_tau*(D[21]*Ts + D[21]*e_tau + D[27]*Ts*u1 - D[27]*Ts*u1_in))/Ts_e_tau2;
	P[22] = (e_tau*(D[22]*Ts + D[22]*e_tau + D[27]*Ts*u2 - D[27]*Ts*u2_in))/Ts_e_tau2;
	P[23] = (e_tau*(D[23]*Ts + D[23]*e_tau + D[27]*Ts*u3 - D[27]*Ts*u3_in))/Ts_e_tau2;
	P[24] = D[24];
	P[25] = D[25];
	P[26] = D[26];
	P[27] = D[27] + Q[9];
	P[28] = D[28] - Ts*(D[32]*e_b1 - D[29]*e_b1 + D[30]*e_b1*(bias1 - u1));
	P[29] = (e_tau*(D[29]*Ts + D[29]*e_tau + D[31]*Ts*u1 - D[31]*Ts*u1_in))/Ts_e_tau2;
	P[30] = D[30];
	P[31] = D[31];
	P[32] = D[32] + Q[10];
	P[33] = D[33] - Ts*(D[37]*e_b2 - D[34]*e_b2 + D[35]*e_b2*(bias2 - u2));
	P[34] = (e_tau*(D[34]*Ts + D[34]*e_tau + D[36]*Ts*u2 - D[36]*Ts*u2_in))/Ts_e_tau2;
	P[35] = D[35];
	P[36] = D[36];
	P[37] = D[37] + Q[11];
	P[38] = D[38] - Ts*(D[42]*e_b3 - D[39]*e_b3 + D[40]*e_b3*(bias3 - u3));
	P[39] = (e_tau*(D[39]*Ts + D[39]*e_tau + D[41]*Ts*u3 - D[41]*Ts*u3_in))/Ts_e_tau2;
	P[40] = D[40];
	P[41] = D[41];
	P[42] = D[42] + Q[12];

    
	/********* this is the update part of the equation ***********/

    float S[3] = {P[0] + s_a, P[1] + s_a, P[2] + s_a};

	X[0] = w1 + (P[0]*(gyro_x - w1))/S[0];
	X[1] = w2 + (P[1]*(gyro_y - w2))/S[1];
	X[2] = w3 + (P[2]*(gyro_z - w3))/S[2];
	X[3] = u1 + (P[3]*(gyro_x - w1))/S[0];
	X[4] = u2 + (P[5]*(gyro_y - w2))/S[1];
	X[5] = u3 + (P[7]*(gyro_z - w3))/S[2];
	X[6] = b1 + (P[9]*(gyro_x - w1))/S[0];
	X[7] = b2 + (P[12]*(gyro_y - w2))/S[1];
	X[8] = b3 + (P[15]*(gyro_z - w3))/S[2];
	X[9] = tau + (P[18]*(gyro_x - w1))/S[0] + (P[19]*(gyro_y - w2))/S[1] + (P[20]*(gyro_z - w3))/S[2];
	X[10] = bias1 + (P[28]*(gyro_x - w1))/S[0];
	X[11] = bias2 + (P[33]*(gyro_y - w2))/S[1];
	X[12] = bias3 + (P[38]*(gyro_z - w3))/S[2];

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
	P[18] = -D[18]*(D[0]/S[0] - 1);
	P[19] = -D[19]*(D[1]/S[1] - 1);
	P[20] = -D[20]*(D[2]/S[2] - 1);
	P[21] = D[21] - (D[3]*D[18])/S[0];
	P[22] = D[22] - (D[5]*D[19])/S[1];
	P[23] = D[23] - (D[7]*D[20])/S[2];
	P[24] = D[24] - (D[9]*D[18])/S[0];
	P[25] = D[25] - (D[12]*D[19])/S[1];
	P[26] = D[26] - (D[15]*D[20])/S[2];
	P[27] = D[27] - D[18]*D[18]/S[0] - D[19]*D[19]/S[1] - D[20]*D[20]/S[2];
	P[28] = -D[28]*(D[0]/S[0] - 1);
	P[29] = D[29] - (D[3]*D[28])/S[0];
	P[30] = D[30] - (D[9]*D[28])/S[0];
	P[31] = D[31] - (D[18]*D[28])/S[0];
	P[32] = D[32] - D[28]*D[28]/S[0];
	P[33] = -D[33]*(D[1]/S[1] - 1);
	P[34] = D[34] - (D[5]*D[33])/S[1];
	P[35] = D[35] - (D[12]*D[33])/S[1];
	P[36] = D[36] - (D[19]*D[33])/S[1];
	P[37] = D[37] - D[33]*D[33]/S[1];
	P[38] = -D[38]*(D[2]/S[2] - 1);
	P[39] = D[39] - (D[7]*D[38])/S[2];
	P[40] = D[40] - (D[15]*D[38])/S[2];
	P[41] = D[41] - (D[20]*D[38])/S[2];
	P[42] = D[42] - D[38]*D[38]/S[2];

    
    // apply limits to some of the state variables
    if (X[9] > -1.5f)
        X[9] = -1.5f;
    if (X[9] < -5.0f)
        X[9] = -5.0f;
    if (X[10] > 0.5f)
        X[10] = 0.5f;
    if (X[10] < -0.5f)
        X[10] = -0.5f;
    if (X[11] > 0.5f)
        X[11] = 0.5f;
    if (X[11] < -0.5f)
        X[11] = -0.5f;
    if (X[12] > 0.5f)
        X[12] = 0.5f;
    if (X[12] < -0.5f)
        X[12] = -0.5f;
    
    // Slowly track the residual noise
    noise[0] = noise[0] * 0.99f + powf(gyro_x - w1,2.0f) * 0.01f;
    noise[1] = noise[1] *0.99f + powf(gyro_y - w2,2.0f) * 0.01f;
    noise[2] = noise[2] *  0.99f + powf(gyro_z - w3,2.0f) * 0.01f;
}

/**
 * Initialize the state variable and covariance matrix
 * for the system identification EKF
 */
void af_init(float X[AF_NUMX], float P[AF_NUMP])
{
    const float q_init[AF_NUMX] = {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        0.05f, 0.05f, 0.005f,
        0.05f,
        0.05f, 0.05f, 0.05f};
    
	X[0] = X[1] = X[2] = 0.0f;    // assume no rotation
	X[3] = X[4] = X[5] = 0.0f;    // and no net torque
	X[6] = X[7]        = 10.0f;   // medium amount of strength
    X[8]               = 7.0f;    // yaw
	X[9] = -4.0f;                 // and 50 ms time scale
	X[10] = X[11] = X[12] = 0.0f; // zero bias

	// P initialization
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
	P[18] = 0.0f;
	P[19] = 0.0f;
	P[20] = 0.0f;
	P[21] = 0.0f;
	P[22] = 0.0f;
	P[23] = 0.0f;
	P[24] = 0.0f;
	P[25] = 0.0f;
	P[26] = 0.0f;
	P[27] = q_init[9];
	P[28] = 0.0f;
	P[29] = 0.0f;
	P[30] = 0.0f;
	P[31] = 0.0f;
	P[32] = q_init[10];
	P[33] = 0.0f;
	P[34] = 0.0f;
	P[35] = 0.0f;
	P[36] = 0.0f;
	P[37] = q_init[11];
	P[38] = 0.0f;
	P[39] = 0.0f;
	P[40] = 0.0f;
	P[41] = 0.0f;
	P[42] = q_init[12];

}