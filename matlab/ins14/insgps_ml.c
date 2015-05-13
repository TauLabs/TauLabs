#include "math.h"
#include "mex.h"   
#include "insgps.h"
#include "string.h"
#include "stdint.h"
#include "stdbool.h"

bool mlStringCompare(const mxArray * mlVal, char * cStr);
bool mlGetFloatArray(const mxArray * mlVal, float * dest, int numel);

// constants/macros/typdefs
#define NUMX 16			// number of states, X is the state vector
#define NUMV 10			// number of measurements, v is the measurement noise vector

extern float P[NUMX][NUMX];	// covariance matrix and state vector
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	char * function_name;
	float accel_data[3];
	float gyro_data[3];
	float mag_data[3];
	float pos_data[3];
	float vel_data[3];
	float baro_data;
	float dT;

	//All code and internal function calls go in here!
	if(!mxIsChar(prhs[0])) {
		mexErrMsgTxt("First parameter must be name of a function\n");
		return;
	} 

	if(mlStringCompare(prhs[0], "INSGPSInit")) {
		INSGPSInit();
	} else 	if(mlStringCompare(prhs[0], "INSStatePrediction")) {

		if(nrhs != 4) {
			mexErrMsgTxt("Incorrect number of inputs for state prediction\n");
			return;
		}

		if(!mlGetFloatArray(prhs[1], gyro_data, 3) || 
			!mlGetFloatArray(prhs[2], accel_data, 3) ||
			!mlGetFloatArray(prhs[3], &dT, 1)) 
			return;

		INSStatePrediction(gyro_data, accel_data, dT);
		INSCovariancePrediction(dT);
	} else 	if(mlStringCompare(prhs[0], "INSFullCorrection")) {

		if(nrhs != 5) {
			mexErrMsgTxt("Incorrect number of inputs for correction\n");
			return;
		}

		if(!mlGetFloatArray(prhs[1], mag_data, 3) ||
				!mlGetFloatArray(prhs[2], pos_data, 3) ||
				!mlGetFloatArray(prhs[3], vel_data ,3) ||
				!mlGetFloatArray(prhs[4], &baro_data, 1)) {
			mexErrMsgTxt("Error with the input parameters\n");
			return;
		}

        INSCorrection(mag_data, pos_data, vel_data, baro_data, FULL_SENSORS);
	} else 	if(mlStringCompare(prhs[0], "INSMagCorrection")) {
		if(nrhs != 2) {
			mexErrMsgTxt("Incorrect number of inputs for correction\n");
			return;
		}

		if(!mlGetFloatArray(prhs[1], mag_data, 3)) {
			mexErrMsgTxt("Error with the input parameters\n");
			return;
		}

        INSCorrection(mag_data, pos_data, vel_data, baro_data, MAG_SENSORS);
    } else 	if(mlStringCompare(prhs[0], "INSBaroCorrection")) {
		if(nrhs != 2) {
			mexErrMsgTxt("Incorrect number of inputs for correction\n");
			return;
		}

		if(!mlGetFloatArray(prhs[1], &baro_data, 1)) {
			mexErrMsgTxt("Error with the input parameters\n");
			return;
		}

		INSCorrection(mag_data, pos_data, vel_data, baro_data, BARO_SENSOR);
	} else 	if(mlStringCompare(prhs[0], "INSMagVelBaroCorrection")) {

		if(nrhs != 4) {
			mexErrMsgTxt("Incorrect number of inputs for correction\n");
			return;
		}

		if(!mlGetFloatArray(prhs[1], mag_data, 3) ||
				!mlGetFloatArray(prhs[2], vel_data ,3) ||
				!mlGetFloatArray(prhs[3], &baro_data, 1)) {
			mexErrMsgTxt("Error with the input parameters\n");
			return;
		}

        INSCorrection(mag_data, pos_data, vel_data, baro_data, MAG_SENSORS | BARO_SENSOR);
	} else 	if(mlStringCompare(prhs[0], "INSGpsCorrection")) {

		if(nrhs != 3) {
			mexErrMsgTxt("Incorrect number of inputs for correction\n");
			return;
		}

		if(!mlGetFloatArray(prhs[1], pos_data, 3) ||
				!mlGetFloatArray(prhs[2], vel_data ,3)) {
			mexErrMsgTxt("Error with the input parameters\n");
			return;
		}

        INSCorrection(mag_data, pos_data, vel_data, baro_data, HORIZ_POS_SENSORS | HORIZ_VEL_SENSORS);
	} else 	if(mlStringCompare(prhs[0], "INSVelBaroCorrection")) {

		if(nrhs != 3) {
			mexErrMsgTxt("Incorrect number of inputs for correction\n");
			return;
		}

		if(!mlGetFloatArray(prhs[1], vel_data, 3) ||
				!mlGetFloatArray(prhs[2], &baro_data, 1)) {
			mexErrMsgTxt("Error with the input parameters\n");
			return;
		}

		INSCorrection(mag_data, pos_data, vel_data, baro_data, BARO_SENSOR | HORIZ_VEL_SENSORS);
	} else if (mlStringCompare(prhs[0], "INSSetPosVelVar")) {
		float pos_var;
        float vel_var;
        float vert_pos_var;
		if((nrhs != 4) || !mlGetFloatArray(prhs[1], &pos_var, 1) ||
                !mlGetFloatArray(prhs[2], &vel_var, 1) ||
                !mlGetFloatArray(prhs[3], &vert_pos_var, 1)) {
			mexErrMsgTxt("Error with input parameters\n");
			return;
		}
		INSSetPosVelVar(pos_var, vel_var, vert_pos_var);
	} else if (mlStringCompare(prhs[0], "INSSetGyroBias")) {
		float gyro_bias[3];
		if((nrhs != 2) || !mlGetFloatArray(prhs[1], gyro_bias, 3)) {
			mexErrMsgTxt("Error with input parameters\n");
			return;
		}
		INSSetGyroBias(gyro_bias);
	} else if (mlStringCompare(prhs[0], "INSSetAccelVar")) {
		float accel_var[3];
		if((nrhs != 2) || !mlGetFloatArray(prhs[1], accel_var, 3)) {
			mexErrMsgTxt("Error with input parameters\n");
			return;
		}
		INSSetAccelVar(accel_var);
	} else if (mlStringCompare(prhs[0], "INSSetGyroVar")) {
		float gyro_var[3];
		if((nrhs != 2) || !mlGetFloatArray(prhs[1], gyro_var, 3)) {
			mexErrMsgTxt("Error with input parameters\n");
			return;
		}
		INSSetGyroVar(gyro_var);
	} else if (mlStringCompare(prhs[0], "INSSetMagNorth")) {
		float mag_north[3];
		float Bmag;
		if((nrhs != 2) || !mlGetFloatArray(prhs[1], mag_north, 3)) {
			mexErrMsgTxt("Error with input parameters\n");
			return;
		}
		Bmag = sqrt(mag_north[0] * mag_north[0] + mag_north[1] * mag_north[1] +
				mag_north[2] * mag_north[2]);
		mag_north[0] = mag_north[0] / Bmag;
		mag_north[1] = mag_north[1] / Bmag;
		mag_north[2] = mag_north[2] / Bmag;

		INSSetMagNorth(mag_north);
	} else if (mlStringCompare(prhs[0], "INSSetMagVar")) {
		float mag_var[3];
		if((nrhs != 2) || !mlGetFloatArray(prhs[1], mag_var, 3)) {
			mexErrMsgTxt("Error with input parameters\n");
			return;
		}
		INSSetMagVar(mag_var);
	} else if (mlStringCompare(prhs[0], "INSSetBaroVar")) {
		float baro_var;
		if((nrhs != 2) || !mlGetFloatArray(prhs[1], &baro_var, 1)) {
			mexErrMsgTxt("Error with input parameters\n");
			return;
		}
		INSSetBaroVar(baro_var);
	} else if (mlStringCompare(prhs[0], "INSSetState")) {
        int i;
		float new_state[NUMX];
		if((nrhs != 2) || !mlGetFloatArray(prhs[1], new_state, NUMX)) {
			mexErrMsgTxt("Error with input parameters\n");
			return;
		}
        
        INSSetState(&new_state[0], &new_state[3], &new_state[6], &new_state[10], &new_state[13]);
	} else {
		mexErrMsgTxt("Unknown function");
	}

	if(nlhs > 0) {
		// return current state vector
		double * data_out;
		int i;

        float pos[3], vel[3], q[4], gyro_bias[3], accel_bias[3];        
        INSGetState(pos, vel, q, gyro_bias, accel_bias);

        plhs[0] = mxCreateDoubleMatrix(1,NUMX,0);
		data_out = mxGetData(plhs[0]);
        data_out[0] = pos[0];
        data_out[1] = pos[1];
        data_out[2] = pos[2];
        data_out[3] = vel[0];
        data_out[4] = vel[1];
        data_out[5] = vel[2];
        data_out[6] = q[0];
        data_out[7] = q[1];
        data_out[8] = q[2];
        data_out[9] = q[3];
        data_out[10] = gyro_bias[0];
        data_out[11] = gyro_bias[1];
        data_out[12] = gyro_bias[2];
        data_out[13] = accel_bias[0];
        data_out[14] = accel_bias[1];
        data_out[15] = accel_bias[2];
	}

	if(nlhs > 1) {
		//return covariance estimate
		double * data_copy = mxCalloc(NUMX*NUMX, sizeof(double));
		int i, j, k;

		plhs[1] = mxCreateDoubleMatrix(NUMX,NUMX,0);
		for(i = 0; i < NUMX; i++)
			for(j = 0; j < NUMX; j++)
			{
				data_copy[j + i * NUMX] = P[j][i];
			}

		mxSetData(plhs[1], data_copy);
	}

	return;
}
        
bool mlGetFloatArray(const mxArray * mlVal, float * dest, int numel) {
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

bool mlStringCompare(const mxArray * mlVal, char * cStr) {
	int i;
	char * mlCStr = 0;
	bool val = false;
	int strLen = mxGetNumberOfElements(mlVal);

	mlCStr = mxCalloc((1+strLen),  sizeof(*mlCStr));
	if(!mlCStr)
		return false;

	if(mxGetString(mlVal, mlCStr, strLen+1))
		goto cleanup;

	for(i = 0; i < strLen; i++) {
		if(mlCStr[i] != cStr[i]) 
			goto cleanup;
	}

	if(cStr[i] == '\0')
		val = true;
	
cleanup:
	if(mlCStr) {
		mxFree(mlCStr);
		mlCStr = 0;
	}
	return val;
}
