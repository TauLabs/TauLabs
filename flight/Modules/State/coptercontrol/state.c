/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup CCState Copter Control State Estimation
 * @{
 *
 * @file       state.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      State estimation for CC(3D)
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
#include "state.h"
#include "sensorfetch.h"
#include "attitudedrift.h"

#include "accels.h"
#include "attitudeactual.h"
#include "attitudesettings.h"
#include "flightstatus.h"
#include "gpsposition.h"
#include "gpsvelocity.h"
#include "gyros.h"
#include "homelocation.h"
#include "manualcontrolcommand.h"
#include "positionactual.h"
#include "velocityactual.h"

#if defined(PIOS_GPS_PROVIDES_AIRSPEED)
#include "gps_airspeed.h"
#endif

#include "coordinate_conversions.h"
#include <pios_board_info.h>

// Private constants
#define STACK_SIZE_BYTES 800
#define TASK_PRIORITY (tskIDLE_PRIORITY+3)

#define SENSOR_PERIOD 4
#define LOOP_RATE_MS  25.0f

// Private types

// Private variables
static xTaskHandle taskHandle;
static xQueueHandle gyro_queue;

static bool gpsNew_flag;
static HomeLocationData homeLocation;
struct GlobalAttitudeVariables *glblAtt;
SensorSettingsData sensorSettings;
AttitudeSettingsData attitudeSettings;
GyrosBiasData gyrosBias;

// For running trim flights
uint16_t const MAX_TRIM_FLIGHT_SAMPLES = 65535;

// Private functions
static void StateTask(void *parameters);

//! Get the sensor data and rotate from board into body frame
static int32_t updateIntertialSensors(AccelsData * accelsData, GyrosData * gyrosData, bool cc3d_flag);

//! Update the position estimate
static void updateT3(GPSVelocityData * gpsVelocityData, PositionActualData * positionActualData);

//! Predict attitude forward one time step
static void updateSO3(float *gyros, float dT);

//! Cache settings locally
static void inertialSensorSettingsUpdatedCb(UAVObjEvent * objEv);

// TODO: Move this into a global library that is aware of the home location to share code with revo
//! Convert from LLA to NED accounting for the geoid separation
static int32_t LLA2NED(int32_t LL[2], float altitude, float *NED);

//! Recompute the translation from LLA to NED and force and update of the position
static void HomeLocationUpdatedCb(UAVObjEvent * objEv);

//! Set a flag to tell the algorithm to use this data
static void GPSPositionUpdatedCb(UAVObjEvent * objEv);

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t StateStart(void)
{

	// Start main task
	xTaskCreate(StateTask, (signed char *)"State", STACK_SIZE_BYTES / 4,
		    NULL, TASK_PRIORITY, &taskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_ATTITUDE, taskHandle);
	PIOS_WDG_RegisterFlag(PIOS_WDG_ATTITUDE);

	return 0;
}

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t StateInitialize(void)
{
	//Initialize UAVOs
	AttitudeActualInitialize();
	AttitudeSettingsInitialize();
	SensorSettingsInitialize();
	AccelsInitialize();
	GyrosInitialize();

	// TODO: These should be dependent on planning to run navigation
	// TODO: The fact these ARE registered and GPS is running should be in sanity check
	PositionActualInitialize();
	VelocityActualInitialize();
	GPSPositionInitialize();
	GPSVelocityInitialize();
	HomeLocationInitialize();

	gpsNew_flag = false;
	glblAtt = (GlobalAttitudeVariables *) pvPortMalloc(sizeof(GlobalAttitudeVariables));

	// Initialize quaternion
	AttitudeActualData attitude;
	AttitudeActualGet(&attitude);
	attitude.q1 = 1;
	attitude.q2 = 0;
	attitude.q3 = 0;
	attitude.q4 = 0;
	AttitudeActualSet(&attitude);

	//--------
	// If bootloader runs, cannot trust the global values to init to 0.
	//--------
	memset(glblAtt->gyro_correct_int, 0, sizeof(glblAtt->gyro_correct_int));

	glblAtt->q[0] = 1;
	glblAtt->q[1] = 0;
	glblAtt->q[2] = 0;
	glblAtt->q[3] = 0;
	
	//Set Rsb to Id.
	for (uint8_t i = 0; i < 3; i++) {
		for (uint8_t j = 0; j < 3; j++){
			glblAtt->Rsb[i][j] = 0;
		}
		glblAtt->Rsb[i][i] = 1;
	}

	glblAtt->trim_requested = false;

	AttitudeSettingsConnectCallback(&inertialSensorSettingsUpdatedCb);
	SensorSettingsConnectCallback(&inertialSensorSettingsUpdatedCb);

	HomeLocationConnectCallback(&HomeLocationUpdatedCb);
	GPSPositionConnectCallback(&GPSPositionUpdatedCb);

	return 0;
}

MODULE_INITCALL(StateInitialize, StateStart)

/**
 * Module thread, should not return.
 */
int32_t accel_test;
int32_t gyro_test;
static void StateTask(void *parameters)
{
	uint8_t init = 0;
	AlarmsClear(SYSTEMALARMS_ALARM_ATTITUDE);

	// Set critical error and wait until the accel is producing data
	//THIS IS BOARD SPECIFIC AND DOES NOT BELONG HERE. Can we put it in #if
	//defined(PIOS_INCLUDE_ADXL345)?
	while (PIOS_ADXL345_FifoElements() == 0) {
		AlarmsSet(SYSTEMALARMS_ALARM_ATTITUDE, SYSTEMALARMS_ALARM_CRITICAL);
		PIOS_WDG_UpdateFlag(PIOS_WDG_ATTITUDE);
	}

	const struct pios_board_info *bdinfo = &pios_board_info_blob;

	//Test if board is CopterControl or CC3D
	bool cc3d_flag = (bdinfo->board_rev == 0x02);

	// Force settings update to make sure rotation and home location are loaded
	SensorSettingsUpdatedCb(SensorSettingsHandle());
	SensorSettingsUpdatedCb(AttitudeSettingsHandle());
	HomeLocationUpdatedCb(HomeLocationHandle());

	if (cc3d_flag) {
#if defined(PIOS_INCLUDE_MPU6000)
		gyro_test = PIOS_MPU6000_Test();
#endif
	} else {
#if defined(PIOS_INCLUDE_ADXL345)
		accel_test = PIOS_ADXL345_Test();
#endif

#if defined(PIOS_INCLUDE_ADC)
		// Create queue for passing gyro data, allow 2 back samples in case
		gyro_queue = xQueueCreate(1, sizeof(float) * 4);
		PIOS_Assert(gyro_queue != NULL);
		PIOS_ADC_SetQueue(gyro_queue);
		PIOS_ADC_Config((PIOS_ADC_RATE / 1000.0f) * LOOP_RATE_MS);
#endif

	}

	//Grab temperature at bootup. The hope is that the temperature is close
	//enough to real temperature to have a reasonable density altitude estimate
	{
		float prelim_accels[4];
		float prelim_gyros[4];
		if (cc3d_flag) {
			getSensorsCC3D(prelim_accels, prelim_gyros, glblAtt, &gyrosBias, &sensorSettings);
		} else {
			getSensorsCC(prelim_accels, prelim_gyros, &gyro_queue, glblAtt, &gyrosBias, &sensorSettings);
		}

		int8_t groundTemperature = round(prelim_accels[3]) * 10; // Convert into tenths of degrees

		HomeLocationGroundTemperatureSet(&groundTemperature);
	}

	//Store the original filter specs. This is because we currently have a poor
	//way of calibrating the Premerlani approach
	uint8_t originalFilter = attitudeSettings.FilterChoice;

	//Start clock for delT calculation
	uint32_t rawtime = PIOS_DELAY_GetRaw();

	// ----------------------------- //
	// Main module loop. Never exits //
	// ----------------------------- //
	while (1) {
		FlightStatusData flightStatus;
		FlightStatusGet(&flightStatus);

		//Change gyro calibration parameters...
		if ((xTaskGetTickCount() > 1000) && (xTaskGetTickCount() < 7000)) {
			//...during first 7 seconds or so...
			// For first 7 seconds use accels to get gyro bias
			glblAtt->accelKp = 1;
			glblAtt->accelKi = 0.9;
			glblAtt->yawBiasRate = 0.23;
			init = 0;

			//Force to use the CCC, because of the way it calibrates
			attitudeSettings.FilterChoice = ATTITUDESETTINGS_FILTERCHOICE_CCC;
		} else if (glblAtt->zero_during_arming && (flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMING)) {
			//...during arming...
			glblAtt->accelKp = 1;
			glblAtt->accelKi = 0.9;
			glblAtt->yawBiasRate = 0.23;
			init = 0;

			//Force to use the CCC, because of the way it calibrates
			attitudeSettings.FilterChoice = ATTITUDESETTINGS_FILTERCHOICE_CCC;
		} else if (init == 0) {	//...once fully armed.
			// Reload settings (all the rates)
			AttitudeSettingsAccelKiGet(&glblAtt->accelKi);
			AttitudeSettingsAccelKpGet(&glblAtt->accelKp);
			AttitudeSettingsYawBiasRateGet(&glblAtt->yawBiasRate);

			attitudeSettings.FilterChoice = originalFilter;

			init = 1;
		}
		PIOS_WDG_UpdateFlag(PIOS_WDG_ATTITUDE);

		AccelsData accels;
		GyrosData gyros;
		int8_t retval = 0;

		//Get sensor data, rotate, filter, and output to UAVO. This is the
		//function that calls the wait structures that limit the loop rate
		retval = updateIntertialSensors(&accels, &gyros, cc3d_flag);
		
		//Update UAVOs with most accurate sensor data. Get these out the door ASAP so the stabilization module can act on the gyroscope information.
		GyrosSet(&gyros);
		AccelsSet(&accels);
		
		// Only update attitude when sensor data is good
		if (retval != 0)
			AlarmsSet(SYSTEMALARMS_ALARM_ATTITUDE, SYSTEMALARMS_ALARM_ERROR);
		else {
			// Do not update attitude data in simulation mode
			if (!AttitudeActualReadOnly()) {
				//Calculate delT, the time step between two attitude updates.
				//TODO: Replace this by a constant, as it is known a priori how quickly we sample the sensor data.
				uint16_t dT_us = PIOS_DELAY_DiffuS(rawtime);
				rawtime = PIOS_DELAY_GetRaw();
				dT_us = (dT_us > 0) ? dT_us : 1;
				float delT = dT_us * 1e-6f;
				
				//Update attitude estimation with drift PI feedback on the rate gyroscopes
				if (glblAtt->bias_correct_gyro) {
					updateAttitudeDrift(&accels, &gyros, delT, glblAtt, &attitudeSettings, &sensorSettings);
				}

				updateSO3(&gyros.x, delT);
			}

			AlarmsClear(SYSTEMALARMS_ALARM_ATTITUDE);
		}

		/*=========================*/
		// Perform estimation updates that require GPS
		if (gpsNew_flag == true) {
			uint8_t gpsStatus;
			GPSPositionStatusGet(&gpsStatus);

			if (gpsStatus == GPSPOSITION_STATUS_FIX3D ||
				gpsStatus == GPSPOSITION_STATUS_DIFF3D) {
				//Load UAVOs
				GPSVelocityData gpsVelocityData;
				PositionActualData positionActualData;

				GPSVelocityGet(&gpsVelocityData);
				PositionActualGet(&positionActualData);

				//Estimate position and velocity in NED frame
				updateT3(&gpsVelocityData, &positionActualData);

#if defined(PIOS_GPS_PROVIDES_AIRSPEED)
				//Estimate airspeed from GPS data
				//http://www.engineeringtoolbox.com/air-altitude-pressure-d_462.html
				float staticPressure =
				    homeLocation.SeaLevelPressure * powf(1.0f - 2.2555e-5f *
					    (homeLocation.Altitude - positionActualData. Down), 5.25588f);

				// Convert from millibar to Pa
				float staticAirDensity = staticPressure * 100 * 0.003483613507536f /
				    (homeLocation.GroundTemperature/10 + CELSIUS2KELVIN);

				gps_airspeed_update(&gpsVelocityData, staticAirDensity);
#endif
			}

			gpsNew_flag = false;
		}

	}
}

/**
 * @brief Get sensor data, rotate, filter, and output to UAVO. 
 * @Note This is the function that calls the wait structures that limit the loop rate
 * @param[in] attitudeRaw Populate the UAVO instead of saving right here
 * @return 0 if successfull, -1 if not
 */
static int32_t updateIntertialSensors(AccelsData * accels, GyrosData * gyros, bool cc3d_flag)
{
	int8_t retval;
	float prelim_accels[4];
	float prelim_gyros[4];

	// Get the sensor data in a board specific manner
	if (cc3d_flag) {
		retval = getSensorsCC3D(prelim_accels, prelim_gyros, glblAtt, &gyrosBias, &sensorSettings);
	} else {
		retval = getSensorsCC(prelim_accels, prelim_gyros, &gyro_queue, glblAtt, &gyrosBias, &sensorSettings);
	}

	if (retval < 0) {	// No sensor data.  Alarm set by calling method
		return retval;
	}

	// Rotate sensor board into body frame
	if (glblAtt->rotate) {
		float tmpVec[3];

		// Rotate the vector into a temporary vector, and then copy back into
		// the original vectors.
		rot_mult(glblAtt->Rsb, prelim_accels, tmpVec, true);
		memcpy(prelim_accels, tmpVec, sizeof(tmpVec));

		rot_mult(glblAtt->Rsb, prelim_gyros, tmpVec, true);
		memcpy(prelim_gyros, tmpVec, sizeof(tmpVec));
	}

	// Store rotated accels
	accels->x = prelim_accels[0];
	accels->y = prelim_accels[1];
	accels->z = prelim_accels[2];

	// Store rotated gyros, optionally with bias correction
	if (glblAtt->bias_correct_gyro) {
		// Applying integral component here so it can be seen on the gyros and correct bias
		gyros->x = prelim_gyros[0] + glblAtt->gyro_correct_int[0];
		gyros->y = prelim_gyros[1] + glblAtt->gyro_correct_int[1];
		gyros->z = prelim_gyros[2] + glblAtt->gyro_correct_int[2];
	} else {
		gyros->x = prelim_gyros[0];
		gyros->y = prelim_gyros[1];
		gyros->z = prelim_gyros[2];
	}

	// Estimate accel bias while user flies level
	if (glblAtt->trim_requested) {
		if (glblAtt->trim_samples >= MAX_TRIM_FLIGHT_SAMPLES) {
			glblAtt->trim_requested = false;
		} else {
			uint8_t armed;
			float throttle;
			FlightStatusArmedGet(&armed);
			ManualControlCommandThrottleGet(&throttle);	// Until flight status indicates airborne
			if ((armed == FLIGHTSTATUS_ARMED_ARMED) && (throttle > 0)) {
				glblAtt->trim_samples++;
				// Store the digitally scaled version since that is what we use for bias
				glblAtt->trim_accels[0] += accels->x;
				glblAtt->trim_accels[1] += accels->y;
				glblAtt->trim_accels[2] += accels->z;
			}
		}
	}

	return 0;

}

/**
 * @brief Update the position estimation based on GPS data
 */
static void updateT3(GPSVelocityData * gpsVelocityData, PositionActualData * positionActualData)
{
	//Load UAVOs
	VelocityActualData velocityActualData;
	VelocityActualGet(&velocityActualData);

	// Get a subset of the GPSPosition information
	int32_t LL_int[2];
	float altitude;
	GPSPositionLatitudeGet(&LL_int[0]);
	GPSPositionLongitudeGet(&LL_int[1]);
	GPSPositionAltitudeGet(&altitude);

	// Get NED coordinates from GPS lat-lon
	float gps_NED[3];
	LLA2NED(LL_int, altitude, gps_NED);

	// Calculate filter coefficients
	float dT = .100f;
	float tauPosNorthEast = 0.3f;
	float tauPosDown = 0.5f;
	float tauVelNorthEast = 0.01f;
	float tauVelDown = 0.1f;
	float alphaPosNorthEast = dT / (dT + tauPosNorthEast);
	float alphaPosDown = dT / (dT + tauPosDown);
	float alphaVelNorthEast = dT / (dT + tauVelNorthEast);
	float alphaVelDown = dT / (dT + tauVelDown);

	//Low pass filter for velocity
	velocityActualData.North = (1 - alphaVelNorthEast) * velocityActualData.North +
	    alphaVelNorthEast * gpsVelocityData->North;
	velocityActualData.East = (1 - alphaVelNorthEast) * velocityActualData.East +
	    alphaVelNorthEast * gpsVelocityData->East;
	velocityActualData.Down = (1 - alphaVelDown) * velocityActualData.Down +
	    alphaVelDown * gpsVelocityData->Down;

	//Complementary filter for position
	positionActualData->North = (1 - alphaPosNorthEast) * (positionActualData->North +
				       (velocityActualData.North * dT)) + alphaPosNorthEast * gps_NED[0];
	positionActualData->East = (1 - alphaPosNorthEast) * (positionActualData->East +
				       (velocityActualData.East * dT)) + alphaPosNorthEast * gps_NED[1];
	positionActualData->Down = (1 - alphaPosDown) * (positionActualData->Down +
				  (velocityActualData.Down * dT)) + alphaPosDown * gps_NED[2];

	//Very slowly use GPS heading data to converge. This is a poor way of doing things, but will work in the short term for testing.
	if (fabs(velocityActualData.North) > 3.0f && 	//Instead of calculating velocity norm, use a gauge approach
			(fabs(velocityActualData.North) > 4.0f || fabs(velocityActualData.East) > 3.0f)) {
		float heading = atan2f(velocityActualData.East, velocityActualData.North);

		AttitudeActualData attitudeActual;
		AttitudeActualGet(&attitudeActual);

		while (heading - attitudeActual.Yaw < -180.0f) {
			heading += 360.0f;
		}
		while (heading - attitudeActual.Yaw > 180.0f) {
			heading -= 360.0f;
		}

		attitudeActual.Yaw = .9f * attitudeActual.Yaw + 0.1f * heading;

		// Convert into quaternions (makes assumptions about quaternions and RPY order)
		RPY2Quaternion(&attitudeActual.Roll, &attitudeActual.q1);

		if (!AttitudeActualReadOnly()) {
			AttitudeActualSet(&attitudeActual);
		}
	}

	// Do not update position and velocity estimates when in simulation mode
	PositionActualSet(positionActualData);
	VelocityActualSet(&velocityActualData);
}

/**
 * @brief Predict the attitude forward one step based on gyro data
 */
static void updateSO3(float *gyros, float dT)
{
	{
		// scoping variables to save memory
		// Work out time derivative from INSAlgo writeup
		// Also accounts for the fact that gyros are in deg/s
		float qdot[4];
		qdot[0] =
		    (-glblAtt->q[1] * gyros[0] - glblAtt->q[2] * gyros[1] -
		     glblAtt->q[3] * gyros[2]) * dT * DEG2RAD / 2;
		qdot[1] =
		    (glblAtt->q[0] * gyros[0] - glblAtt->q[3] * gyros[1] +
		     glblAtt->q[2] * gyros[2]) * dT * DEG2RAD / 2;
		qdot[2] =
		    (glblAtt->q[3] * gyros[0] + glblAtt->q[0] * gyros[1] -
		     glblAtt->q[1] * gyros[2]) * dT * DEG2RAD / 2;
		qdot[3] =
		    (-glblAtt->q[2] * gyros[0] + glblAtt->q[1] * gyros[1] +
		     glblAtt->q[0] * gyros[2]) * dT * DEG2RAD / 2;

		// Integrate a time step
		glblAtt->q[0] = glblAtt->q[0] + qdot[0];
		glblAtt->q[1] = glblAtt->q[1] + qdot[1];
		glblAtt->q[2] = glblAtt->q[2] + qdot[2];
		glblAtt->q[3] = glblAtt->q[3] + qdot[3];

		if (glblAtt->q[0] < 0) {
			glblAtt->q[0] = -glblAtt->q[0];
			glblAtt->q[1] = -glblAtt->q[1];
			glblAtt->q[2] = -glblAtt->q[2];
			glblAtt->q[3] = -glblAtt->q[3];
		}
	}

	// Renomalize
	float qmag = sqrtf(powf(glblAtt->q[0], 2.0f) + powf(glblAtt->q[1], 2.0f) +
		  powf(glblAtt->q[2], 2.0f) + powf(glblAtt->q[3], 2.0f));
	glblAtt->q[0] = glblAtt->q[0] / qmag;
	glblAtt->q[1] = glblAtt->q[1] / qmag;
	glblAtt->q[2] = glblAtt->q[2] / qmag;
	glblAtt->q[3] = glblAtt->q[3] / qmag;

	// If quaternion has become inappropriately short or is nan reinit.
	// THIS SHOULD NEVER ACTUALLY HAPPEN
	if ((fabs(qmag) < 1e-3) || (qmag != qmag)) {
		glblAtt->q[0] = 1;
		glblAtt->q[1] = 0;
		glblAtt->q[2] = 0;
		glblAtt->q[3] = 0;
	}

	AttitudeActualData attitudeActual;
	AttitudeActualGet(&attitudeActual);

	quat_copy(glblAtt->q, &attitudeActual.q1);

	// Convert into Euler angles (makes assumptions about quarternions and RPY order)
	Quaternion2RPY(&attitudeActual.q1, &attitudeActual.Roll);

	AttitudeActualSet(&attitudeActual);
}

/**
 * @brief Move the settings variables from the UAVO into a temporary structure
 */
static void inertialSensorSettingsUpdatedCb(UAVObjEvent * objEv)
{
	AttitudeSettingsGet(&attitudeSettings);
	SensorSettingsGet(&sensorSettings);

	glblAtt->accelKp = attitudeSettings.AccelKp;
	glblAtt->accelKi = attitudeSettings.AccelKi;
	glblAtt->yawBiasRate = attitudeSettings.YawBiasRate;

	glblAtt->zero_during_arming = (attitudeSettings.ZeroDuringArming == ATTITUDESETTINGS_ZERODURINGARMING_TRUE);
	glblAtt->bias_correct_gyro = (attitudeSettings.BiasCorrectGyro == ATTITUDESETTINGS_BIASCORRECTGYRO_TRUE);

	//Provide minimum for scale. This keeps the accels from accidentally being "turned off".
	for (int i = 0; i < 3; i++) {
		if (sensorSettings.AccelScale[i] < .01f) {
			sensorSettings.AccelScale[i] = .01f;
		}
	}

	//Load initial gyrobias values into online-estimated gyro bias
	gyrosBias.x = 0;
	gyrosBias.y = 0;
	gyrosBias.z = 0;

	//Calculate sensor to board rotation matrix. If the matrix is the identity,
	//don't expend cycles on rotation
	if (attitudeSettings.BoardRotation[0] == 0 && attitudeSettings.BoardRotation[1] == 0 && attitudeSettings.BoardRotation[2] == 0)
	{
		glblAtt->rotate = false;

		// Shouldn't need to be used, but just to be safe we will anyway
		float rotationQuat[4] = { 1, 0, 0, 0 };
		Quaternion2R(rotationQuat, glblAtt->Rsb);
	} else {
		float rpy[3] = {
		    attitudeSettings.BoardRotation[ATTITUDESETTINGS_BOARDROTATION_ROLL] * DEG2RAD / 100.0f,
		    attitudeSettings.BoardRotation[ATTITUDESETTINGS_BOARDROTATION_PITCH] * DEG2RAD / 100.0f,
            attitudeSettings.BoardRotation[ATTITUDESETTINGS_BOARDROTATION_YAW] * DEG2RAD / 100.0f
		};
		Euler2R(rpy, glblAtt->Rsb);
		glblAtt->rotate = true;
	}

	//Check for trim flight request
	if (attitudeSettings.TrimFlight == ATTITUDESETTINGS_TRIMFLIGHT_START) {
		glblAtt->trim_accels[0] = 0;
		glblAtt->trim_accels[1] = 0;
		glblAtt->trim_accels[2] = 0;
		glblAtt->trim_samples = 0;
		glblAtt->trim_requested = true;
	} else if (attitudeSettings.TrimFlight == ATTITUDESETTINGS_TRIMFLIGHT_LOAD) {
		glblAtt->trim_requested = false;

		// Get sensor data  mean 
		float a_body[3] = { glblAtt->trim_accels[0] / glblAtt->trim_samples,
			glblAtt->trim_accels[1] / glblAtt->trim_samples,
			glblAtt->trim_accels[2] / glblAtt->trim_samples
		};

		// Inverse rotation of sensor data, from body frame into sensor frame
		float a_sensor[3];
		rot_mult(glblAtt->Rsb, a_body, a_sensor, false);

		// Temporary variables
		float psi, theta, phi;

		psi = attitudeSettings.BoardRotation[ATTITUDESETTINGS_BOARDROTATION_YAW] * DEG2RAD / 100.0f;

		float cP = cosf(psi);
		float sP = sinf(psi);

		// In case psi is too small, we have to use a different equation to solve for theta
		if (fabs(psi) > PI / 2)
			theta = atanf((a_sensor[1] + cP * (sP * a_sensor[0] -
					 cP * a_sensor[1])) / (sP * a_sensor[2]));
		else
			theta = atanf((a_sensor[0] - sP * (sP * a_sensor[0] -
					 cP * a_sensor[1])) / (cP * a_sensor[2]));

		phi = atan2f((sP * a_sensor[0] - cP * a_sensor[1]) / (-GRAVITY),
			   (a_sensor[2] / cosf(theta) / (-GRAVITY)));

		attitudeSettings.BoardRotation[ATTITUDESETTINGS_BOARDROTATION_ROLL] = phi * RAD2DEG * 100.0f;
		attitudeSettings.BoardRotation[ATTITUDESETTINGS_BOARDROTATION_PITCH] = theta * RAD2DEG * 100.0f;

		attitudeSettings.TrimFlight = ATTITUDESETTINGS_TRIMFLIGHT_NORMAL;
		AttitudeSettingsSet(&attitudeSettings);
	} else {
		glblAtt->trim_requested = false;
	}

}

/**
 * @brief Convert the GPS LLA position into NED coordinates
 * @note this method uses a taylor expansion around the home coordinates
 * to convert to NED which allows it to be done with all floating
 * calculations
 * @param[in] Current GPS coordinates, (Lat, Lon, Alt)
 * @param[out] NED frame coordinates
 * @returns 0 for success, -1 for failure
 */
static int32_t LLA2NED(int32_t LL[2], float altitude, float *NED)
{
	float *T = glblAtt->T;
	float dL[3] = { (LL[0] - homeLocation.Latitude) / 10.0e6f * DEG2RAD,
		(LL[1] - homeLocation.Longitude) / 10.0e6f * DEG2RAD,
		(altitude - homeLocation.Altitude)
	};

	NED[0] = T[0] * dL[0];
	NED[1] = T[1] * dL[1];
	NED[2] = T[2] * dL[2];

	return 0;
}

//! Set flag for data to be consumed
static void GPSPositionUpdatedCb(UAVObjEvent * objEv)
{
	gpsNew_flag = true;
}

/**
 * @brief Recompute the translation from LLA to NED and force and update of the position
 */
static void HomeLocationUpdatedCb(UAVObjEvent * objEv)
{
	float lat, alt;

	HomeLocationGet(&homeLocation);

	// Compute vector for converting deltaLLA to NED
	lat = homeLocation.Latitude / 10.0e6f * DEG2RAD;
	alt = homeLocation.Altitude;

	float *T = glblAtt->T;

	T[0] = alt + 6.378137E6f;
	T[1] = cosf(lat) * (alt + 6.378137E6f);
	T[2] = -1.0f;

	//Set NED coordinates relative to the new home location
	uint8_t gpsStatus;
	GPSPositionStatusGet(&gpsStatus);

	// TODO: Generate a better criterion
	if (gpsStatus == GPSPOSITION_STATUS_FIX3D ||
		gpsStatus == GPSPOSITION_STATUS_DIFF3D)
	{
		// Get the subset of the GPSPosition data required
		int32_t LL_int[2];
		float altitude;
		GPSPositionLatitudeGet(&LL_int[0]);
		GPSPositionLongitudeGet(&LL_int[1]);
		GPSPositionAltitudeGet(&altitude);

		// Convert LLA into NED and store it.  Assumes field order in PositionActual is NED.
		PositionActualData positionActualData;
		LLA2NED(LL_int, altitude, &positionActualData.North);
		PositionActualSet(&positionActualData);
	}

	// Constrain gravity to reasonable levels (Mars gravity < HomeLocation gravity < Jupiter gravity)
	if (homeLocation.g_e < 3 || homeLocation.g_e > 25) {
		homeLocation.g_e = GRAVITY;
		HomeLocationSet(&homeLocation);
	}
}

/**
 * @}
 * @}
 */
