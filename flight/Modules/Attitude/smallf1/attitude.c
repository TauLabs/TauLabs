/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup AttitudeModuleSmallF1 Attitude estimation for small F1 boards
 * @{
 * @brief      Minimal code for attitude estimation with integrated sensors
 *
 * This is a minimal implementation of the Complementary filter which integrates
 * in the code to fetch the sensor values for @ref Accels and @ref Gyros and store
 * them, and then updates a Complementary filter to estimate @ref AttitudeActual
 * and sets that.
 *
 * @file       attitude.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2015
 * @brief      Update attitude for F1 targets
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
#include "openpilot.h"
#include "physical_constants.h"
#include "gyros.h"
#include "accels.h"
#include "attitudeactual.h"
#include "sensorsettings.h"
#include "attitudesettings.h"
#include "flightstatus.h"
#include "manualcontrolcommand.h"
#include "misc_math.h"
#include "coordinate_conversions.h"
#include <pios_board_info.h>
#include "pios_queue.h"
 
// Private constants
#define STACK_SIZE_BYTES 580
#define TASK_PRIORITY PIOS_THREAD_PRIO_HIGH

#define SENSOR_PERIOD 4
#define GYRO_NEUTRAL 1665

// Private types
enum complimentary_filter_status {
	CF_POWERON,
	CF_INITIALIZING,
	CF_ARMING,
	CF_NORMAL
};

// Private variables
static struct pios_thread *taskHandle;
static SensorSettingsData sensorSettings;

// Private functions
static void AttitudeTask(void *parameters);

static float gyro_correct_int[3] = {0,0,0};

static int32_t updateSensorsDigital(AccelsData * accelsData, GyrosData * gyrosData);
static void updateAttitude(AccelsData *, GyrosData *);
static void settingsUpdatedCb(UAVObjEvent * objEv, void *ctx, void *obj, int len);
static void update_accels(struct pios_sensor_accel_data *accels, AccelsData * accelsData);
static void update_gyros(struct pios_sensor_gyro_data *gyros, GyrosData * gyrosData);
static void updateTemperatureComp(float temperature, float *temp_bias);

//! Compute the mean gyro accumulated and assign the bias
static void accumulate_gyro_compute();

//! Zero the gyro accumulators
static void accumulate_gyro_zero();

//! Store a gyro sample
static void accumulate_gyro(float gyros_out[3]);

static float accelKi = 0;
static float accelKp = 0;
static float accel_alpha = 0;
static bool accel_filter_enabled = false;
static float yawBiasRate = 0;
static const float IDG_GYRO_GAIN = 0.42;
static float q[4] = {1,0,0,0};
static float Rsb[3][3]; // Rotation matrix which transforms from the body frame to the sensor board frame
static int8_t rotate = 0;
static bool zero_during_arming = false;
static bool bias_correct_gyro = true;

// For computing the average gyro during arming
static bool accumulating_gyro = false;
static uint32_t accumulated_gyro_samples = 0;
static float accumulated_gyro[3];

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t AttitudeStart(void)
{
	
	// Start main task
	taskHandle = PIOS_Thread_Create(AttitudeTask, "Attitude", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
	TaskMonitorAdd(TASKINFO_RUNNING_ATTITUDE, taskHandle);
	PIOS_WDG_RegisterFlag(PIOS_WDG_ATTITUDE);
	
	return 0;
}

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t AttitudeInitialize(void)
{
	AttitudeActualInitialize();
	SensorSettingsInitialize();
	AttitudeSettingsInitialize();
	AccelsInitialize();
	GyrosInitialize();
	
	// Initialize quaternion
	AttitudeActualData attitude;
	AttitudeActualGet(&attitude);
	attitude.q1 = 1;
	attitude.q2 = 0;
	attitude.q3 = 0;
	attitude.q4 = 0;
	AttitudeActualSet(&attitude);
	
	// Cannot trust the values to init right above if BL runs
	gyro_correct_int[0] = 0;
	gyro_correct_int[1] = 0;
	gyro_correct_int[2] = 0;
	
	q[0] = 1;
	q[1] = 0;
	q[2] = 0;
	q[3] = 0;
	for(uint8_t i = 0; i < 3; i++)
		for(uint8_t j = 0; j < 3; j++)
			Rsb[i][j] = 0;
	
	AttitudeSettingsConnectCallback(&settingsUpdatedCb);
	SensorSettingsConnectCallback(&settingsUpdatedCb);
	
	return 0;
}

MODULE_INITCALL(AttitudeInitialize, AttitudeStart)

/**
 * Module thread, should not return.
 */
static void AttitudeTask(void *parameters)
{
	AlarmsClear(SYSTEMALARMS_ALARM_ATTITUDE);

	// Force settings update to make sure rotation loaded
	settingsUpdatedCb(NULL, NULL, NULL, 0);
	
	enum complimentary_filter_status complimentary_filter_status;
	complimentary_filter_status = CF_POWERON;

	uint32_t arming_count = 0;

	// Main task loop
	while (1) {

		FlightStatusData flightStatus;
		FlightStatusGet(&flightStatus);

		if (complimentary_filter_status == CF_POWERON) {

			complimentary_filter_status = (PIOS_Thread_Systime() > 1000) ?
				CF_INITIALIZING : CF_POWERON;

		} else if(complimentary_filter_status == CF_INITIALIZING &&
			(PIOS_Thread_Systime() < 7000) && 
			(PIOS_Thread_Systime() > 1000)) {

			// For first 7 seconds use accels to get gyro bias
			accelKp = 0.1f + 0.1f * (PIOS_Thread_Systime() < 4000);
			accelKi = 0.1;
			yawBiasRate = 0.1;
			accel_filter_enabled = false;

		} else if (zero_during_arming && 
			       (flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMING)) {

			// Use a rapidly decrease accelKp to force the attitude to snap back
			// to level and then converge more smoothly
			if (arming_count < 20)
				accelKp = 1.0f;
			else if (accelKp > 0.1f)
				accelKp -= 0.01f;
			arming_count++;

			accelKi = 0.1f;
			yawBiasRate = 0.1f;
			accel_filter_enabled = false;

			// Indicate arming so that after arming it reloads
			// the normal settings
			if (complimentary_filter_status != CF_ARMING) {
				accumulate_gyro_zero();
				complimentary_filter_status = CF_ARMING;
				accumulating_gyro = true;
			}

		} else if (complimentary_filter_status == CF_ARMING ||
			complimentary_filter_status == CF_INITIALIZING) {

			// Reload settings (all the rates)
			AttitudeSettingsAccelKiGet(&accelKi);
			AttitudeSettingsAccelKpGet(&accelKp);
			AttitudeSettingsYawBiasRateGet(&yawBiasRate);
			if(accel_alpha > 0.0f)
				accel_filter_enabled = true;

			// If arming that means we were accumulating gyro
			// samples.  Compute new bias.
			if (complimentary_filter_status == CF_ARMING) {
				accumulate_gyro_compute();
				accumulating_gyro = false;
				arming_count = 0;
			}

			// Indicate normal mode to prevent rerunning this code
			complimentary_filter_status = CF_NORMAL;
		}
		
		PIOS_WDG_UpdateFlag(PIOS_WDG_ATTITUDE);

		AccelsData accels;
		GyrosData gyros;
		int32_t retval = 0;

		retval = updateSensorsDigital(&accels, &gyros);

		// During power on set to angle from accel
		if (complimentary_filter_status == CF_POWERON) {
			float RPY[3];
			float theta = atan2f(accels.x, -accels.z);
			RPY[1] = theta * RAD2DEG;
			RPY[0] = atan2f(-accels.y, -accels.z / cosf(theta)) * RAD2DEG;
			RPY[2] = 0;
			RPY2Quaternion(RPY, q);
		}

		// Only update attitude when sensor data is good
		if (retval != 0)
			AlarmsSet(SYSTEMALARMS_ALARM_ATTITUDE, SYSTEMALARMS_ALARM_ERROR);
		else {
			// Do not update attitude data in simulation mode
			if (!AttitudeActualReadOnly())
				updateAttitude(&accels, &gyros);

			AlarmsClear(SYSTEMALARMS_ALARM_ATTITUDE);
		}
	}
}

/**
 * Get an update from the sensors
 * @param[in] attitudeRaw Populate the UAVO instead of saving right here
 * @return 0 if successfull, -1 if not
 */
static int32_t updateSensorsDigital(AccelsData * accelsData, GyrosData * gyrosData)
{
	struct pios_sensor_gyro_data gyros;
	struct pios_sensor_accel_data accels;
	struct pios_queue *queue;

	queue = PIOS_SENSORS_GetQueue(PIOS_SENSOR_GYRO);
	if(queue == NULL || PIOS_Queue_Receive(queue, (void *) &gyros, 4) == false) {
		return-1;
	}

	// As it says below, because the rest of the code expects the accel to be ready when
	// the gyro is we must block here too
	queue = PIOS_SENSORS_GetQueue(PIOS_SENSOR_ACCEL);
	if(queue == NULL || PIOS_Queue_Receive(queue, (void *) &accels, 1) == false) {
		return -1;
	}
	else
		update_accels(&accels, accelsData);

	// Update gyros after the accels since the rest of the code expects
	// the accels to be available first
	update_gyros(&gyros, gyrosData);

	GyrosSet(gyrosData);
	AccelsSet(accelsData);

	return 0;
}

/**
 * @brief Apply calibration and rotation to the raw accel data
 * @param[in] accels The raw accel data
 */
static void update_accels(struct pios_sensor_accel_data *accels, AccelsData * accelsData)
{
	// Average and scale the accels before rotation
	float accels_out[3] = {accels->x * sensorSettings.AccelScale[0] - sensorSettings.AccelBias[0],
	                       accels->y * sensorSettings.AccelScale[1] - sensorSettings.AccelBias[1],
	                       accels->z * sensorSettings.AccelScale[2] - sensorSettings.AccelBias[2]};

	if (rotate) {
		float accel_rotated[3];
		rot_mult(Rsb, accels_out, accel_rotated, true);
		accelsData->x = accel_rotated[0];
		accelsData->y = accel_rotated[1];
		accelsData->z = accel_rotated[2];
	} else {
		accelsData->x = accels_out[0];
		accelsData->y = accels_out[1];
		accelsData->z = accels_out[2];
	}

	accelsData->temperature = accels->temperature;
}

/**
 * @brief Apply calibration and rotation to the raw gyro data
 * @param[in] gyros The raw gyro data
 */
static void update_gyros(struct pios_sensor_gyro_data *gyros, GyrosData * gyrosData)
{
	static float gyro_temp_bias[3] = {0,0,0};

	// Scale the gyros
	float gyros_out[3] = {gyros->x * sensorSettings.GyroScale[0],
	                      gyros->y * sensorSettings.GyroScale[1],
	                      gyros->z * sensorSettings.GyroScale[2]};

	// Update the bias due to the temperature
	updateTemperatureComp(gyrosData->temperature, gyro_temp_bias);

	// Apply temperature bias correction before the rotation
	if (bias_correct_gyro) {
		gyros_out[0] -= gyro_temp_bias[0];
		gyros_out[1] -= gyro_temp_bias[1];
		gyros_out[2] -= gyro_temp_bias[2];
	}

	// When computing the bias accumulate samples
	accumulate_gyro(gyros_out);


	if (rotate) {
		float gyros[3];
		rot_mult(Rsb, gyros_out, gyros, true);
		gyrosData->x = gyros[0];
		gyrosData->y = gyros[1];
		gyrosData->z = gyros[2];
	} else {
		gyrosData->x = gyros_out[0];
		gyrosData->y = gyros_out[1];
		gyrosData->z = gyros_out[2];
	}

	if(bias_correct_gyro) {
		// Applying integral component here so it can be seen on the gyros and correct bias
		gyrosData->x -= gyro_correct_int[0];
		gyrosData->y -= gyro_correct_int[1];
		gyrosData->z -= gyro_correct_int[2];
	}

	// Because most crafts wont get enough information from gravity to zero yaw gyro, we try
	// and make it average zero (weakly)
	gyro_correct_int[2] += gyrosData->z * yawBiasRate;

	gyrosData->temperature = gyros->temperature;
}

/**
 * If accumulating data and enough samples acquired then recompute
 * the gyro bias based on the mean accumulated
 */
static void accumulate_gyro_compute()
{
	if (accumulating_gyro && 
		accumulated_gyro_samples > 100) {

		gyro_correct_int[0] = accumulated_gyro[0] / accumulated_gyro_samples;
		gyro_correct_int[1] = accumulated_gyro[1] / accumulated_gyro_samples;
		gyro_correct_int[2] = accumulated_gyro[2] / accumulated_gyro_samples;

		accumulate_gyro_zero();

		accumulating_gyro = false;
	}
}

/**
 * Zero the accumulation of gyro data
 */
static void accumulate_gyro_zero()
{
	accumulated_gyro_samples = 0;
	accumulated_gyro[0] = 0;
	accumulated_gyro[1] = 0;
	accumulated_gyro[2] = 0;
}

/**
 * Accumulate a set of gyro samples for computing the
 * bias
 * @param [in] gyrosData The samples of data to accumulate
 * @param [in] gyro_temp_bias The current temperature bias to account for
 */
static void accumulate_gyro(float gyros_out[3])
{
	if (!accumulating_gyro)
		return;

	accumulated_gyro_samples++;
	accumulated_gyro[0] += gyros_out[0];
	accumulated_gyro[1] += gyros_out[1];
	accumulated_gyro[2] += gyros_out[2];
}

static inline void apply_accel_filter(const float * raw, float * filtered)
{
	if(accel_filter_enabled) {
		filtered[0] = filtered[0] * accel_alpha + raw[0] * (1 - accel_alpha);
		filtered[1] = filtered[1] * accel_alpha + raw[1] * (1 - accel_alpha);
		filtered[2] = filtered[2] * accel_alpha + raw[2] * (1 - accel_alpha);
	} else {
		filtered[0] = raw[0];
		filtered[1] = raw[1];
		filtered[2] = raw[2];
	}
}

static void updateAttitude(AccelsData * accelsData, GyrosData * gyrosData)
{
	float dT;
	uint32_t thisSysTime = PIOS_Thread_Systime();
	static uint32_t lastSysTime = 0;
	static float accels_filtered[3] = {0,0,0};
	static float grot_filtered[3] = {0,0,0};

	dT = (thisSysTime == lastSysTime) ? 0.001f : (PIOS_THREAD_TIMEOUT_MAX & (thisSysTime - lastSysTime)) / 1000.0f;
	lastSysTime = thisSysTime;
	
	// Bad practice to assume structure order, but saves memory
	float * gyros = &gyrosData->x;
	float * accels = &accelsData->x;
	
	float grot[3];
	float accel_err[3];

	// Apply smoothing to accel values, to reduce vibration noise before main calculations.
	apply_accel_filter(accels,accels_filtered);
	
	// Rotate gravity to body frame, filter and cross with accels
	grot[0] = -(2 * (q[1] * q[3] - q[0] * q[2]));
	grot[1] = -(2 * (q[2] * q[3] + q[0] * q[1]));
	grot[2] = -(q[0] * q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]);

	// Apply same filtering to the rotated attitude to match delays
	apply_accel_filter(grot,grot_filtered);
	
	// Compute the error between the predicted direction of gravity and smoothed acceleration
	CrossProduct((const float *) accels_filtered, (const float *) grot_filtered, accel_err);
	
	// Account for accel magnitude
	float accel_mag = sqrtf(accels_filtered[0]*accels_filtered[0] + accels_filtered[1]*accels_filtered[1] + accels_filtered[2]*accels_filtered[2]);

	// Account for filtered gravity vector magnitude
	float grot_mag;

	if (accel_filter_enabled)
		grot_mag = sqrtf(grot_filtered[0]*grot_filtered[0] + grot_filtered[1]*grot_filtered[1] + grot_filtered[2]*grot_filtered[2]);
	else
		grot_mag = 1.0f;

	if (grot_mag > 1.0e-3f && accel_mag > 1.0e-3f) {
		accel_err[0] /= (accel_mag*grot_mag);
		accel_err[1] /= (accel_mag*grot_mag);
		accel_err[2] /= (accel_mag*grot_mag);
		
		// Accumulate integral of error.  Scale here so that units are (deg/s) but Ki has units of s
		gyro_correct_int[0] -= accel_err[0] * accelKi;
		gyro_correct_int[1] -= accel_err[1] * accelKi;
		
		// Correct rates based on error, integral component dealt with in updateSensors
		gyros[0] += accel_err[0] * accelKp / dT;
		gyros[1] += accel_err[1] * accelKp / dT;
		gyros[2] += accel_err[2] * accelKp / dT;
	}
	
	{ // scoping variables to save memory
		// Work out time derivative from INSAlgo writeup
		// Also accounts for the fact that gyros are in deg/s
		float qdot[4];
		qdot[0] = (-q[1] * gyros[0] - q[2] * gyros[1] - q[3] * gyros[2]) * dT * DEG2RAD / 2;
		qdot[1] = (q[0] * gyros[0] - q[3] * gyros[1] + q[2] * gyros[2]) * dT * DEG2RAD / 2;
		qdot[2] = (q[3] * gyros[0] + q[0] * gyros[1] - q[1] * gyros[2]) * dT * DEG2RAD / 2;
		qdot[3] = (-q[2] * gyros[0] + q[1] * gyros[1] + q[0] * gyros[2]) * dT * DEG2RAD / 2;
		
		// Take a time step
		q[0] = q[0] + qdot[0];
		q[1] = q[1] + qdot[1];
		q[2] = q[2] + qdot[2];
		q[3] = q[3] + qdot[3];
		
		if(q[0] < 0) {
			q[0] = -q[0];
			q[1] = -q[1];
			q[2] = -q[2];
			q[3] = -q[3];
		}
	}
	
	// Renomalize
	float qmag = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
	q[0] = q[0] / qmag;
	q[1] = q[1] / qmag;
	q[2] = q[2] / qmag;
	q[3] = q[3] / qmag;
	
	// If quaternion has become inappropriately short or is nan reinit.
	// THIS SHOULD NEVER ACTUALLY HAPPEN
	if((fabsf(qmag) < 1e-3f) || IS_NOT_FINITE(qmag)) {
		q[0] = 1;
		q[1] = 0;
		q[2] = 0;
		q[3] = 0;
	}
	
	AttitudeActualData attitudeActual;
	AttitudeActualGet(&attitudeActual);
	
	quat_copy(q, &attitudeActual.q1);
	
	// Convert into eueler degrees (makes assumptions about RPY order)
	Quaternion2RPY(&attitudeActual.q1,&attitudeActual.Roll);
	
	AttitudeActualSet(&attitudeActual);
}

/**
 * Compute the bias expected from temperature variation for each gyro
 * channel
 */
static void updateTemperatureComp(float temperature, float *temp_bias)
{
	static int temp_counter = 0;
	static float temp_accum = 0;

	static const float TEMP_MIN = -10;
	static const float TEMP_MAX = 60;

	if (temperature < TEMP_MIN)
		temperature = TEMP_MIN;
	if (temperature > TEMP_MAX)
		temperature = TEMP_MAX;

	if (temp_counter < 500) {
		temp_accum += temperature;
		temp_counter ++;
	} else {
		float t = temp_accum / temp_counter;
		temp_accum = 0;
		temp_counter = 0;

		// Compute a third order polynomial for each chanel after each 500 samples
		temp_bias[0] = sensorSettings.XGyroTempCoeff[0] + 
		               sensorSettings.XGyroTempCoeff[1] * t + 
		               sensorSettings.XGyroTempCoeff[2] * powf(t,2) + 
		               sensorSettings.XGyroTempCoeff[3] * powf(t,3);
		temp_bias[1] = sensorSettings.YGyroTempCoeff[0] + 
		               sensorSettings.YGyroTempCoeff[1] * t + 
		               sensorSettings.YGyroTempCoeff[2] * powf(t,2) + 
		               sensorSettings.YGyroTempCoeff[3] * powf(t,3);
		temp_bias[2] = sensorSettings.ZGyroTempCoeff[0] + 
		               sensorSettings.ZGyroTempCoeff[1] * t + 
		               sensorSettings.ZGyroTempCoeff[2] * powf(t,2) + 
		               sensorSettings.ZGyroTempCoeff[3] * powf(t,3);
	}
}

static void settingsUpdatedCb(UAVObjEvent * objEv, void *ctx, void *obj, int len) {
	(void) objEv; (void) ctx; (void) obj; (void) len;

	AttitudeSettingsData attitudeSettings;
	AttitudeSettingsGet(&attitudeSettings);
	SensorSettingsGet(&sensorSettings);
	
	
	accelKp = attitudeSettings.AccelKp;
	accelKi = attitudeSettings.AccelKi;
	yawBiasRate = attitudeSettings.YawBiasRate;

	// Calculate accel filter alpha, in the same way as for gyro data in stabilization module.
	const float fakeDt = 0.0025f;
	if(attitudeSettings.AccelTau < 0.0001f) {
		accel_alpha = 0;   // not trusting this to resolve to 0
		accel_filter_enabled = false;
	} else {
		accel_alpha = expf(-fakeDt  / attitudeSettings.AccelTau);
		accel_filter_enabled = true;
	}
	
	zero_during_arming = attitudeSettings.ZeroDuringArming == ATTITUDESETTINGS_ZERODURINGARMING_TRUE;
	bias_correct_gyro = attitudeSettings.BiasCorrectGyro == ATTITUDESETTINGS_BIASCORRECTGYRO_TRUE;
		
	gyro_correct_int[0] = 0;
	gyro_correct_int[1] = 0;
	gyro_correct_int[2] = 0;
	
	// Indicates not to expend cycles on rotation
	if(attitudeSettings.BoardRotation[0] == 0 && attitudeSettings.BoardRotation[1] == 0 &&
	   attitudeSettings.BoardRotation[2] == 0) {
		rotate = 0;
		
		// Shouldn't be used but to be safe
		float rotationQuat[4] = {1,0,0,0};
		Quaternion2R(rotationQuat, Rsb);
	} else {
		float rotationQuat[4];
		const float rpy[3] = {attitudeSettings.BoardRotation[ATTITUDESETTINGS_BOARDROTATION_ROLL] / 100.0f,
			attitudeSettings.BoardRotation[ATTITUDESETTINGS_BOARDROTATION_PITCH] / 100.0f,
			attitudeSettings.BoardRotation[ATTITUDESETTINGS_BOARDROTATION_YAW] / 100.0f};
		RPY2Quaternion(rpy, rotationQuat);
		Quaternion2R(rotationQuat, Rsb);
		rotate = 1;
	}
}
/**
 * @}
 * @}
 */
