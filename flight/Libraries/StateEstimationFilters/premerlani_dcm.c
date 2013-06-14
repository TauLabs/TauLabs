/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 * @addtogroup StateEstimationFilters
 * @{
 *
 * @file       premerlani_dcm.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      DCM algorithm implementation used by @ref StateModule
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

#include "pios.h"
#include "physical_constants.h"
#include "premerlani_dcm.h"
#include <pios_board_info.h>
#include "coordinate_conversions.h"

#include "gpsvelocity.h"
#if defined (PIOS_INCLUDE_MAGNETOMETER)	//THIS PIOS DEFINE DOES NOT CURRENTLY EXIST, BUT WE SHOULD ADD IT IN ORDER TO SUPPORT ALL MAGS, NOT JUST THE HMC5883
#include "magnetometer.h"
#endif
#if defined (PIOS_GPS_PROVIDES_AIRSPEED)	//THIS PIOS DEFINE DOES NOT CURRENTLY EXIST, BUT WE SHOULD ADD IT IN ORDER TO SUPPORT ALL MAGS, NOT JUST THE HMC5883
#include "airspeedactual.h"
#endif

//Global variables
extern AttitudeSettingsData attitudeSettings;
extern SensorSettingsData sensorSettings;
extern GyrosBiasData gyrosBias;

struct GlobalDcmDriftVariables {
	float GPSV_old[3];
	
	float accels_e_integrator[3];
	float omegaCorrI[3];
	
	bool gpsPresent_flag;
	volatile uint8_t gpsVelocityDataConsumption_flag;
	bool magNewData_flag;
	
	float accelsKp;
	float rollPitchKp;
	float rollPitchKi;
	float yawKp;
	float yawKi;
	float gyroCalibTau;
	
	//! Accumulator for the time step between GPS updates
	float delT_between_GPS;
};

#define GPS_UNCONSUMED       0x00
#define GPS_CONSUMED_BY_RPY  0x01
#define GPS_CONSUMED_BY_Y    0x02
#define GPS_CONSUMED         0xFF

extern struct GlobalDcmDriftVariables *drft;

// Private constants

// Private types

// Private variables

//#define DRIFT_TYPE CCC
enum DRIFT_CORRECTION_ALGOS {
	CCC,
	PREMERLANI
};

// Private functions
static void rollPitch_drift_GPS(float Rbe[3][3], float accels_e_int[3], float delT, float *errRollPitch_b);
static void gyro_drift(float gyro[3], float errYaw_b[3], float errRollPitch_b[3], float normOmegaScalar, float delT, float *omegaCorrP, float *omegaCorrI);
static void rollPitch_drift_accel(float accels[3], float gyros[3], float Rbe[3][3], float airspeed_tas, float *errRollPitch_b);


/**
 * Correct the sensor drift using Premerlani algorithm
 */


#define GPS_SPEED_MIN 5		// Minimum velocity in [m/s]
#define GPS_YAW_KP .1;
#define MAG_YAW_KP .1;

#define MAXIMUM_SPIN_DCM_INTEGRAL 20.0f	//in [deg/s]

/*
 * Correct sensor drift, using the DCM approach from W. Premerlani et. al
 */
void Premerlani_DCM(float *accels, float *gyros, float Rbe[3][3], const float delT, bool GPS_Drift_Compensation, GlobalAttitudeVariables *glblAtt, float *omegaCorrP)
{
	float errYaw_b[3] = { 0, 0, 0 };
	float errRollPitch_b[3] = { 0, 0, 0 };
	
	float normOmegaScalar = VectorMagnitude(gyros);
	
	//Correct roll-pitch drift via GPS and accelerometer
	//The math is derived from Roll-Pitch Gyro Drift Compensation, Rev.3, by W. Premerlani
#if defined (PIOS_INCLUDE_GPS)
	if (drft->gpsPresent_flag && GPS_Drift_Compensation) {
		float accels_e[3];
		
		//Rotate accelerometer readings into Earth frame. Note that we need to take the transpose of Rbe.
		rot_mult(Rbe, accels, accels_e, TRUE);
		
		//Integrate accelerometer measurements in Earth frame
		drft->accels_e_integrator[0] += accels_e[0] * delT;
		drft->accels_e_integrator[1] += accels_e[1] * delT;
		drft->accels_e_integrator[2] += accels_e[2] * delT;
		
		drft->delT_between_GPS += delT;
		
		//Check if the GPS has new information.
		if (!
		    (drft->
		     gpsVelocityDataConsumption_flag & GPS_CONSUMED_BY_RPY)) {
				 
				 //Compute drift correction, errRollPitch_b, from GPS
				 rollPitch_drift_GPS(Rbe, drft->accels_e_integrator,
											drft->delT_between_GPS,
											errRollPitch_b);
				 
				 //Reset integrator
				 memset(drft->accels_e_integrator, 0,
						  sizeof(drft->accels_e_integrator));
				 
				 //Mark GPS data as consumed by this function
				 drft->gpsVelocityDataConsumption_flag |=
			    GPS_CONSUMED_BY_RPY;
				 
				 drft->delT_between_GPS = 0;
				 
			 }
	}
#endif
	
	if (!GPS_Drift_Compensation) {
#if defined (PIOS_INCLUDE_GPS) && 0 || defined (PIOS_INCLUDE_MAGNETOMETER)
		if (!(drft->gpsVelocityDataConsumption_flag & GPS_CONSUMED_BY_Y)) {
			// We're actually using new GPS data here, but it's already been stored in old by the previous function
			yaw_drift_MagGPS(Rbe, true, drft->magNewData_flag, errYaw_b);	
			
			// Mark GPS data as consumed by this function
			drft->gpsVelocityDataConsumption_flag |= GPS_CONSUMED_BY_Y;
		} else {
			// In addition to calculating the roll-pitch-yaw error, we can calculate yaw drift, errYaw_b, based on GPS and attitude data
			// We're actually using new GPS data here, but it's already been stored in old by the previous function
			yaw_drift_MagGPS(Rbe, false, drft->magNewData_flag, errYaw_b);	
		}
		
		// Reset flag. Not the best place to do it, but it's messy anywhere else
		if (drft->magNewData_flag) {
			drft->magNewData_flag = false;
		}
#endif
		//In addition, we can calculate roll-pitch error with only the aid of an accelerometer
#if defined(PIOS_GPS_PROVIDES_AIRSPEED)
		AirspeedActualData airspeedActualData;
		AirspeedActualGet(&airspeedActualData);
		float airspeed_tas = airspeedActualData.TrueAirspeed;
#else
		float airspeed_tas = 0;
#endif
		rollPitch_drift_accel(accels, gyros, Rbe, airspeed_tas,
									 errRollPitch_b);
	}
	
	// Calculate gyro drift, based on all errors
	gyro_drift(gyros, errYaw_b, errRollPitch_b, normOmegaScalar, delT, omegaCorrP, drft->omegaCorrI);
	
	//Calculate final drift response
	gyros[0] += omegaCorrP[0] + drft->omegaCorrI[0];
	gyros[1] += omegaCorrP[1] + drft->omegaCorrI[1];
	gyros[2] += omegaCorrP[2] + drft->omegaCorrI[2];
	
	//Add 0.0001% of proportional error back into gyroscope bias offset. This keeps DC elements out of the raw gyroscope data.
	glblAtt->gyro_correct_int[0] += omegaCorrP[0] / 1000000.0f;
	glblAtt->gyro_correct_int[1] += omegaCorrP[1] / 1000000.0f;
	
	// Because most crafts wont get enough information from gravity to zero yaw gyro, we try
	// and make it average zero (weakly)
	glblAtt->gyro_correct_int[2] += -gyros[2] * glblAtt->yawBiasRate;
}

/*
 * Calculate the error, as indicated by the accelerometer. It's important to use the true airspeed here, and not CAS or EAS
 */
static void rollPitch_drift_accel(float accels[3], float gyros[3], float Rbe[3][3],
									float airspeed_tas, float *errRollPitch_b)
{
	float g_ref[3];
	float errAccelsRollPitch_b[3];
	
	//Fuselage Z vector is simply the Z column of the rotation matrix
	float fuselageZ[3] = { Rbe[0][2], Rbe[1][2], Rbe[2][2] };
	
	//Calculate centripetal acceleration
	float acc_centripetal[3] =
	{ 0, gyros[2] * airspeed_tas, -gyros[1] * airspeed_tas };
	
	//Combine measurements and centripetal accelerations. This should give the gravity reference vector
	g_ref[0] = accels[0];	//Remember that for fixed-wing, acc_centripetal[0] = 0
	g_ref[1] = accels[1] - acc_centripetal[1];
	g_ref[2] = accels[2] - acc_centripetal[2];
	
	//Normalize acceleration vector
	float normG = sqrtf(g_ref[0] * g_ref[0] + g_ref[1] * g_ref[1] + g_ref[2] * g_ref[2]);
	g_ref[0] /= normG;
	g_ref[1] /= normG;
	g_ref[2] /= normG;
	
	//Error is cross product of reference vector and estimated orientation. Reverse operation in order to have correct sign for application to error sum
	CrossProduct((const float *)fuselageZ, (const float *)g_ref, errAccelsRollPitch_b);
	
	//Add errors into global error vector
	errRollPitch_b[0] += errAccelsRollPitch_b[0] * drft->accelsKp;
	errRollPitch_b[1] += errAccelsRollPitch_b[1] * drft->accelsKp;
	errRollPitch_b[2] += errAccelsRollPitch_b[2] * drft->accelsKp;
}

/*
 * This function takes in a number of different error rotations, weights them, and applies
 *  them to the gyroscope readings. In addition, there is a function to increase the correction 
 *  at high speeds in order to keep the gyro from becoming "dizzy".
 */
static void gyro_drift(float gyro[3], float errYaw_b[3], float errRollPitch_b[3],
					 float normOmegaScalar, float delT, float *omegaCorrP,
					 float *omegaCorrI)
{
	float kpyaw;
	float kprollpitch;
	
	// boost the KPs at high spin rate, to compensate for increased error due to calibration error
	// above 50 degrees/second, scale by rotation rate divided by 50. As per Fast Rotations, by William Premerlani. 
	if (normOmegaScalar < (50.0f)) {
		kpyaw = drft->yawKp;
		kprollpitch = drft->rollPitchKp;
	} else if (normOmegaScalar < (500.0f)) {
		kpyaw = (normOmegaScalar / 50.0f) * drft->yawKp;
		kprollpitch = (normOmegaScalar / 50.0f) * drft->rollPitchKp;
	} else {
		kpyaw = 10.0f * drft->yawKp;
		kprollpitch = 10.0f * drft->rollPitchKp;
	}
	
	//Compute proportional correction.
	omegaCorrP[0] = errRollPitch_b[0] * kprollpitch + errYaw_b[0] * kpyaw;
	omegaCorrP[1] = errRollPitch_b[1] * kprollpitch + errYaw_b[1] * kpyaw;
	omegaCorrP[2] = errRollPitch_b[2] * kprollpitch + errYaw_b[2] * kpyaw;
	
	// Compute integral correction. Turn off the offset integrator while spinning at high speeds,
	// it doesn't work in that case, and it only causes trouble.
	if (normOmegaScalar < MAXIMUM_SPIN_DCM_INTEGRAL) {
		omegaCorrI[0] += (errRollPitch_b[0] * drft->rollPitchKi + errYaw_b[0] * drft->yawKi) * delT;
		omegaCorrI[1] += (errRollPitch_b[1] * drft->rollPitchKi + errYaw_b[1] * drft->yawKi) * delT;
		omegaCorrI[2] += (errRollPitch_b[2] * drft->rollPitchKi + errYaw_b[2] * drft->yawKi) * delT;
	}
}

/*
 * From Roll-Pitch Gyro Drift Compensation, Rev 3. William Premerlani, 2012.
 */
static void rollPitch_drift_GPS(float Rbe[3][3], float accels_e_int[3],
								 float delT_between_updates, float *errRollPitch_b)
{
	float errRollPitch_e[3];
	float dGPSdt_e[3];
	
	GPSVelocityData gpsVelocity;
	GPSVelocityGet(&gpsVelocity);
	
	dGPSdt_e[0] = (gpsVelocity.North - drft->GPSV_old[0]) / delT_between_updates;
	dGPSdt_e[1] = (gpsVelocity.East - drft->GPSV_old[1]) / delT_between_updates;
	dGPSdt_e[2] = -GRAVITY + (gpsVelocity.Down - drft->GPSV_old[2]) / delT_between_updates;
	
	drft->GPSV_old[0] = gpsVelocity.North;
	drft->GPSV_old[1] = gpsVelocity.East;
	drft->GPSV_old[2] = gpsVelocity.Down;
	
	float normdGPSdt_e = VectorMagnitude(dGPSdt_e);
	
	//Take cross product of integrated accelerometer measurements with integrated earth frame accelerations. We should be using normalized dGPSdt, but we perform that calculation in the following line(s).
	CrossProduct((const float *)accels_e_int, (const float *)dGPSdt_e, errRollPitch_e);
	
	//Scale cross product
	errRollPitch_e[0] /= (normdGPSdt_e * delT_between_updates);
	errRollPitch_e[1] /= (normdGPSdt_e * delT_between_updates);
	errRollPitch_e[2] /= (normdGPSdt_e * delT_between_updates);
	
	//Rotate earth drift error back into body frame;
	rot_mult(Rbe, errRollPitch_e, errRollPitch_b, false);
}


/**
 * @}
 * @}
 */
