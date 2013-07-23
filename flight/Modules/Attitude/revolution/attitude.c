/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup AttitudeModule Attitude and state estimation module
 * @{
 * @brief      Full attitude estimation method with selectable algorithms
 *
 * Based on the value of @ref StateEstimation this module will select between
 * the complementary filter and the INSGPS to fuse data from @ref Gyros, @ref
 * Accels, @ref Magnetometer, @ref GPSPosition, @ref GPSVelocity, and @ref
 * BaroAltitude to estimation @ref PositionActual, @ref VelocityActual and 
 * @ref AttitudeActual.
 *
 * @file       attitude.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org Copyright (C) 2012-2013
 * @brief      Full attitude estimation algorithm
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

/**
 * Input objects: None, takes sensor data via pios
 * Output objects: @ref AttitudeActual, @ref PositionActual, @ref VelocityActual
 *
 * This module performs a state estimate from the sensor data using either the
 * INSGPS algorithm for attitude, velocity, and position, or the complementary
 * filter algorithm for just attitude.
 */

#include "pios.h"
#include "openpilot.h"
#include "physical_constants.h"

#include "accels.h"
#include "attitudeactual.h"
#include "attitudesettings.h"
#include "baroaltitude.h"
#include "flightstatus.h"
#include "gpsposition.h"
#include "gpsvelocity.h"
#include "gyros.h"
#include "gyrosbias.h"
#include "homelocation.h"
#include "sensorsettings.h"
#include "inssettings.h"
#include "insstate.h"
#include "magnetometer.h"
#include "nedposition.h"
#include "positionactual.h"
#include "stateestimation.h"
#include "velocityactual.h"
#include "coordinate_conversions.h"

// Private constants
#define STACK_SIZE_BYTES 2448
#define TASK_PRIORITY (tskIDLE_PRIORITY+3)
#define FAILSAFE_TIMEOUT_MS 10

// low pass filter configuration to calculate offset
// of barometric altitude sensor
// reasoning: updates at: 10 Hz, tau= 300 s settle time
// exp(-(1/f) / tau ) ~=~ 0.9997
#define BARO_OFFSET_LOWPASS_ALPHA 0.9997f 

// Private types


// Track the initialization state of the complementary filter
enum complementary_filter_status {
	CF_POWERON,
	CF_INITIALIZING,
	CF_ARMING,
	CF_NORMAL
};

struct complementary_filter_state {
	//! Track how many cycles the system has been arming to accelerate convergence
	uint32_t   arming_count;

	//! Coefficient for the accelerometer LPF
	float      accel_alpha;
	//! Store the low pass filtered accelerometer
	float      accels_filtered[3];
	//! Low pass filtered gravity vector
	float      grot_filtered[3];
	//! If the accelerometer LPF is enabled
	bool       accel_filter_enabled;

	//! The accumulator of gyros during arming
	float      accumulated_gyro[3];
	//! How many gyro samples were acquired
	uint32_t   accumulated_gyro_samples;
	//! Indicate if currently acquiring gyro samples
	bool       accumulating_gyro;

	//! Store when the function is initialized to time arming and convergence
	uint32_t   reset_timeval;

	//! Tracks the initialization state of the complementary filter
	enum complementary_filter_status     initialization;
};

// Private variables
static xTaskHandle attitudeTaskHandle;

static xQueueHandle gyroQueue;
static xQueueHandle accelQueue;
static xQueueHandle magQueue;
static xQueueHandle baroQueue;
static xQueueHandle gpsQueue;
static xQueueHandle gpsVelQueue;

static AttitudeSettingsData attitudeSettings;
static HomeLocationData homeLocation;
static INSSettingsData insSettings;
static StateEstimationData stateEstimation;
static bool gyroBiasSettingsUpdated = false;
const uint32_t SENSOR_QUEUE_SIZE = 10;
static const float zeros[3] = {0.0f, 0.0f, 0.0f};

static struct complementary_filter_state complementary_filter_state;

// Private functions
static void AttitudeTask(void *parameters);

//! Set the navigation information to the raw estimates
static int32_t setNavigationRaw();

//! Update the complementary filter attitude estimate
static int32_t updateAttitudeComplementary(bool first_run, bool secondary);
//! Set the @ref AttitudeActual to the complementary filter estimate
static int32_t setAttitudeComplementary();

//! Update the INSGPS attitude estimate
static int32_t updateAttitudeINSGPS(bool first_run, bool outdoor_mode);
//! Set the attitude to the current INSGPS estimate
static int32_t setAttitudeINSGPS();
//! Set the navigation to the current INSGPS estimate
static int32_t setNavigationINSGPS();

static void settingsUpdatedCb(UAVObjEvent * objEv);

//! A low pass filter on the accels which helps with vibration resistance
static void apply_accel_filter(const float * raw, float * filtered);
static int32_t getNED(GPSPositionData * gpsPosition, float * NED);

//! Compute the mean gyro accumulated and assign the bias
static void accumulate_gyro_compute();

//! Zero the gyro accumulators
static void accumulate_gyro_zero();

//! Store a gyro sample
static void accumulate_gyro(GyrosData *gyrosData);

/**
 * API for sensor fusion algorithms:
 * Configure(xQueueHandle gyro, xQueueHandle accel, xQueueHandle mag, xQueueHandle baro)
 *   Stores all the queues the algorithm will pull data from
 * FinalizeSensors() -- before saving the sensors modifies them based on internal state (gyro bias)
 * Update() -- queries queues and updates the attitude estiamte
 */


/**
 * Initialise the module.  Called before the start function
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t AttitudeInitialize(void)
{
	AttitudeActualInitialize();
	AttitudeSettingsInitialize();
	SensorSettingsInitialize();
	INSSettingsInitialize();
	INSStateInitialize();
	NEDPositionInitialize();
	PositionActualInitialize();
	StateEstimationInitialize();
	VelocityActualInitialize();

	// Initialize this here while we aren't setting the homelocation in GPS
	HomeLocationInitialize();

	AttitudeSettingsConnectCallback(&settingsUpdatedCb);
	HomeLocationConnectCallback(&settingsUpdatedCb);
	SensorSettingsConnectCallback(&settingsUpdatedCb);
	INSSettingsConnectCallback(&settingsUpdatedCb);
	StateEstimationConnectCallback(&settingsUpdatedCb);

	return 0;
}

/**
 * Start the task.  Expects all objects to be initialized by this point.
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t AttitudeStart(void)
{
	// Create the queues for the sensors
	gyroQueue = xQueueCreate(1, sizeof(UAVObjEvent));
	accelQueue = xQueueCreate(1, sizeof(UAVObjEvent));
	magQueue = xQueueCreate(2, sizeof(UAVObjEvent));
	baroQueue = xQueueCreate(1, sizeof(UAVObjEvent));
	gpsQueue = xQueueCreate(1, sizeof(UAVObjEvent));
	gpsVelQueue = xQueueCreate(1, sizeof(UAVObjEvent));

	// Initialize quaternion
	AttitudeActualData attitude;
	AttitudeActualGet(&attitude);
	attitude.q1 = 1;
	attitude.q2 = 0;
	attitude.q3 = 0;
	attitude.q4 = 0;
	AttitudeActualSet(&attitude);

	// Cannot trust the values to init right above if BL runs
	GyrosBiasData gyrosBias;
	GyrosBiasGet(&gyrosBias);
	gyrosBias.x = 0;
	gyrosBias.y = 0;
	gyrosBias.z = 0;
	GyrosBiasSet(&gyrosBias);

	GyrosConnectQueue(gyroQueue);
	AccelsConnectQueue(accelQueue);
	if (MagnetometerHandle())
		MagnetometerConnectQueue(magQueue);
	if (BaroAltitudeHandle())
		BaroAltitudeConnectQueue(baroQueue);
	if (GPSPositionHandle())
		GPSPositionConnectQueue(gpsQueue);
	if (GPSVelocityHandle())
		GPSVelocityConnectQueue(gpsVelQueue);

	// Start main task
	xTaskCreate(AttitudeTask, (signed char *)"Attitude", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &attitudeTaskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_ATTITUDE, attitudeTaskHandle);
	PIOS_WDG_RegisterFlag(PIOS_WDG_ATTITUDE);

	return 0;
}

MODULE_INITCALL(AttitudeInitialize, AttitudeStart)

/**
 * Module thread, should not return.
 */
static void AttitudeTask(void *parameters)
{
	bool first_run = true;
	uint32_t last_algorithm;
	bool     last_complementary;
	AlarmsClear(SYSTEMALARMS_ALARM_ATTITUDE);

	// Force settings update to make sure rotation loaded
	settingsUpdatedCb(NULL);

	// Wait for all the sensors be to read
	vTaskDelay(100);

	// Invalidate previous algorithm to trigger a first run
	last_algorithm = 0xfffffff;
	last_complementary = false;

	// Main task loop
	while (1) {

		int32_t ret_val = -1;

		// When changing the attitude filter reinitialize
		if (last_algorithm != stateEstimation.AttitudeFilter) {
			last_algorithm = stateEstimation.AttitudeFilter;
			first_run = true;
		}

		// There are two options to select:
		//   Attitude filter - what sets the attitude
		//   Navigation filter - what sets the position and velocity
		// If the INS is used for either then it should run
		bool ins = (stateEstimation.AttitudeFilter == STATEESTIMATION_ATTITUDEFILTER_INSOUTDOOR) ||
		           (stateEstimation.AttitudeFilter == STATEESTIMATION_ATTITUDEFILTER_INSINDOOR) ||
		           (stateEstimation.NavigationFilter == STATEESTIMATION_NAVIGATIONFILTER_INS);

		// INS outdoor mode when used for navigation OR explicit outdoor mode
		bool outdoor = (stateEstimation.AttitudeFilter == STATEESTIMATION_ATTITUDEFILTER_INSOUTDOOR) ||
		               ((stateEstimation.AttitudeFilter == STATEESTIMATION_ATTITUDEFILTER_COMPLEMENTARY) &&
		                (stateEstimation.NavigationFilter == STATEESTIMATION_NAVIGATIONFILTER_INS));

		// Complementary filter only needed when used for attitude
		bool complementary = stateEstimation.AttitudeFilter == STATEESTIMATION_ATTITUDEFILTER_COMPLEMENTARY;

		// Update one or both filters
		if (ins) {
			ret_val = updateAttitudeINSGPS(first_run, outdoor);
			if (complementary)
				 updateAttitudeComplementary(first_run || complementary != last_complementary, true);
		} else {
			ret_val = updateAttitudeComplementary(first_run, false);
		}

		last_complementary = complementary;

		// Get the requested data
		// This  function blocks on data queue
		switch (stateEstimation.AttitudeFilter ) {
		case STATEESTIMATION_ATTITUDEFILTER_COMPLEMENTARY:
			setAttitudeComplementary();
			break;
		case STATEESTIMATION_ATTITUDEFILTER_INSOUTDOOR:
		case STATEESTIMATION_ATTITUDEFILTER_INSINDOOR:
			setAttitudeINSGPS();
			break;
		}

		// Use the selected source for position and velocity
		switch (stateEstimation.NavigationFilter) {
		case STATEESTIMATION_NAVIGATIONFILTER_INS:
				// TODO: When running in dual mode and the INS is not initialized set
				// an error here
				setNavigationINSGPS();
				break;
		default:
				setNavigationRaw();		
		}

		if(ret_val == 0)
			first_run = false;

		PIOS_WDG_UpdateFlag(PIOS_WDG_ATTITUDE);
	}
}

//! The complementary filter attitude estimate
static float cf_q[4];

/**
 * Update the complementary filter estimate of attitude
 * @param[in] first_run indicates the filter was just selected
 * @param[in] secondary indicates the EKF is running as well
 */
static int32_t updateAttitudeComplementary(bool first_run, bool secondary)
{
	UAVObjEvent ev;
	GyrosData gyrosData;
	AccelsData accelsData;
	static int32_t timeval;
	float dT;

	// Wait until the accel and gyro object is updated, if a timeout then go to failsafe
	if (!secondary && (
		 xQueueReceive(gyroQueue, &ev, MS2TICKS(FAILSAFE_TIMEOUT_MS)) != pdTRUE ||
		 xQueueReceive(accelQueue, &ev, MS2TICKS(1)) != pdTRUE ) )
	{
		// When one of these is updated so should the other
		// Do not set attitude timeout warnings in simulation mode
		if (!AttitudeActualReadOnly()){
			AlarmsSet(SYSTEMALARMS_ALARM_ATTITUDE,SYSTEMALARMS_ALARM_WARNING);
			return -1;
		}
	}

	AccelsGet(&accelsData);

	// When this algorithm is first run force it to a known condition
	if(first_run) {
		MagnetometerData magData;
		magData.x = 100;
		magData.y = 0;
		magData.z = 0;

		// Wait for a mag reading if a magnetometer was registered
		if (PIOS_SENSORS_GetQueue(PIOS_SENSOR_MAG) != NULL) {
			if ( !secondary && xQueueReceive(magQueue, &ev, MS2TICKS(20)) != pdTRUE ) {
				return -1;
			}
			MagnetometerGet(&magData);
		}

		float RPY[3];
		float theta = atan2f(accelsData.x, -accelsData.z);
		RPY[1] = theta * RAD2DEG;
		RPY[0] = atan2f(-accelsData.y, -accelsData.z / cosf(theta)) * RAD2DEG;
		RPY[2] = atan2f(-magData.y, magData.x) * RAD2DEG;
		RPY2Quaternion(RPY, cf_q);

		complementary_filter_state.initialization = CF_POWERON;
		complementary_filter_state.reset_timeval = PIOS_DELAY_GetRaw();
		timeval = PIOS_DELAY_GetRaw();

		complementary_filter_state.arming_count = 0;

		return 0;
	}

	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);

	uint32_t ms_since_reset = PIOS_DELAY_DiffuS(complementary_filter_state.reset_timeval) / 1000;
	if (complementary_filter_state.initialization == CF_POWERON) {
		// Wait one second before starting to initialize
		complementary_filter_state.initialization = 
		    (ms_since_reset  > 1000) ?
			CF_INITIALIZING : 
			CF_POWERON;
	} else if(complementary_filter_state.initialization == CF_INITIALIZING &&
		(ms_since_reset < 7000) && 
		(ms_since_reset > 1000)) {

		// For first 7 seconds use accels to get gyro bias
		attitudeSettings.AccelKp = 0.1f + 0.1f * (xTaskGetTickCount() < 4000);
		attitudeSettings.AccelKi = 0.1f;
		attitudeSettings.YawBiasRate = 0.1f;
		attitudeSettings.MagKp = 0.1f;
	} else if ((attitudeSettings.ZeroDuringArming == ATTITUDESETTINGS_ZERODURINGARMING_TRUE) && 
	           (flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMING)) {

		// Use a rapidly decrease accelKp to force the attitude to snap back
		// to level and then converge more smoothly
		if (complementary_filter_state.arming_count < 20)
			attitudeSettings.AccelKp = 1.0f;
		else if (attitudeSettings.AccelKp > 0.1f)
			attitudeSettings.AccelKp -= 0.01f;
		complementary_filter_state.arming_count++;

		// Set the other parameters to drive faster convergence
		attitudeSettings.AccelKi = 0.1f;
		attitudeSettings.YawBiasRate = 0.1f;
		attitudeSettings.MagKp = 0.1f;

		// Don't apply LPF to the accels during arming
		complementary_filter_state.accel_filter_enabled = false;

		// Indicate arming so that after arming it reloads
		// the normal settings
		if (complementary_filter_state.initialization != CF_ARMING) {
			accumulate_gyro_zero();
			complementary_filter_state.initialization = CF_ARMING;
			complementary_filter_state.accumulating_gyro = true;
		}

	} else if (complementary_filter_state.initialization == CF_ARMING ||
	           complementary_filter_state.initialization == CF_INITIALIZING) {

		AttitudeSettingsGet(&attitudeSettings);
		if(complementary_filter_state.accel_alpha > 0.0f)
			complementary_filter_state.accel_filter_enabled = true;

		// If arming that means we were accumulating gyro
		// samples.  Compute new bias.
		if (complementary_filter_state.initialization == CF_ARMING) {
			accumulate_gyro_compute();
			complementary_filter_state.accumulating_gyro = false;
			complementary_filter_state.arming_count = 0;
		}

		// Indicate normal mode to prevent rerunning this code
		complementary_filter_state.initialization = CF_NORMAL;
	}

	GyrosGet(&gyrosData);
	accumulate_gyro(&gyrosData);

	// Compute the dT using the cpu clock
	dT = PIOS_DELAY_DiffuS(timeval) / 1000000.0f;
	timeval = PIOS_DELAY_GetRaw();

	float grot[3];
	float accel_err[3];
	float *grot_filtered = complementary_filter_state.grot_filtered;
	float *accels_filtered = complementary_filter_state.accels_filtered;

	// Apply smoothing to accel values, to reduce vibration noise before main calculations.
	apply_accel_filter(&accelsData.x,accels_filtered);

	// Rotate gravity to body frame and cross with accels
	grot[0] = -(2 * (cf_q[1] * cf_q[3] - cf_q[0] * cf_q[2]));
	grot[1] = -(2 * (cf_q[2] * cf_q[3] + cf_q[0] * cf_q[1]));
	grot[2] = -(cf_q[0]*cf_q[0] - cf_q[1]*cf_q[1] - cf_q[2]*cf_q[2] + cf_q[3]*cf_q[3]);
	CrossProduct((const float *) &accelsData.x, (const float *) grot, accel_err);

	// Apply same filtering to the rotated attitude to match delays
	apply_accel_filter(grot,grot_filtered);

	// Compute the error between the predicted direction of gravity and smoothed acceleration
	CrossProduct((const float *) accels_filtered, (const float *) grot_filtered, accel_err);

	float grot_mag;
	if (complementary_filter_state.accel_filter_enabled)
		grot_mag = sqrtf(grot_filtered[0]*grot_filtered[0] + grot_filtered[1]*grot_filtered[1] + grot_filtered[2]*grot_filtered[2]);
	else
		grot_mag = 1.0f;

	// Account for accel magnitude
	float accel_mag;
	accel_mag = accels_filtered[0]*accels_filtered[0] + accels_filtered[1]*accels_filtered[1] + accels_filtered[2]*accels_filtered[2];
	accel_mag = sqrtf(accel_mag);
	if (grot_mag > 1.0e-3f && accel_mag > 1.0e-3f) {
		accel_err[0] /= (accel_mag * grot_mag);
		accel_err[1] /= (accel_mag * grot_mag);
		accel_err[2] /= (accel_mag * grot_mag);
	} else {
		accel_err[0] = 0;
		accel_err[1] = 0;
		accel_err[2] = 0;
	}

	float mag_err[3];
	if ( secondary || xQueueReceive(magQueue, &ev, 0) == pdTRUE )
	{
		// Rotate gravity to body frame and cross with accels
		float brot[3];
		float Rbe[3][3];
		MagnetometerData mag;
		
		Quaternion2R(cf_q, Rbe);
		MagnetometerGet(&mag);

		// If the mag is producing bad data don't use it (normally bad calibration)
		if  (mag.x == mag.x && mag.y == mag.y && mag.z == mag.z &&
			 homeLocation.Set == HOMELOCATION_SET_TRUE) {
			rot_mult(Rbe, homeLocation.Be, brot, false);

			float mag_len = sqrtf(mag.x * mag.x + mag.y * mag.y + mag.z * mag.z);
			mag.x /= mag_len;
			mag.y /= mag_len;
			mag.z /= mag_len;

			float bmag = sqrtf(brot[0] * brot[0] + brot[1] * brot[1] + brot[2] * brot[2]);
			brot[0] /= bmag;
			brot[1] /= bmag;
			brot[2] /= bmag;

			// Only compute if neither vector is null
			if (bmag < 1 || mag_len < 1)
				mag_err[0] = mag_err[1] = mag_err[2] = 0;
			else
				CrossProduct((const float *) &mag.x, (const float *) brot, mag_err);

			if (mag_err[2] != mag_err[2])
				mag_err[2] = 0;
		} else
			mag_err[2] = 0;
	} else {
		mag_err[2] = 0;
	}

	// Accumulate integral of error.  Scale here so that units are (deg/s) but Ki has units of s
	GyrosBiasData gyrosBias;
	GyrosBiasGet(&gyrosBias);
	gyrosBias.x -= accel_err[0] * attitudeSettings.AccelKi;
	gyrosBias.y -= accel_err[1] * attitudeSettings.AccelKi;
	gyrosBias.z -= mag_err[2] * attitudeSettings.MagKi;
	GyrosBiasSet(&gyrosBias);

	// Correct rates based on error, integral component dealt with in updateSensors
	gyrosData.x += accel_err[0] * attitudeSettings.AccelKp / dT;
	gyrosData.y += accel_err[1] * attitudeSettings.AccelKp / dT;
	gyrosData.z += mag_err[2] * attitudeSettings.MagKp / dT;

	// Work out time derivative from INSAlgo writeup
	// Also accounts for the fact that gyros are in deg/s
	float qdot[4];
	qdot[0] = (-cf_q[1] * gyrosData.x - cf_q[2] * gyrosData.y - cf_q[3] * gyrosData.z) * dT * DEG2RAD / 2;
	qdot[1] = (cf_q[0] * gyrosData.x - cf_q[3] * gyrosData.y + cf_q[2] * gyrosData.z) * dT * DEG2RAD / 2;
	qdot[2] = (cf_q[3] * gyrosData.x + cf_q[0] * gyrosData.y - cf_q[1] * gyrosData.z) * dT * DEG2RAD / 2;
	qdot[3] = (-cf_q[2] * gyrosData.x + cf_q[1] * gyrosData.y + cf_q[0] * gyrosData.z) * dT * DEG2RAD / 2;

	// Take a time step
	cf_q[0] = cf_q[0] + qdot[0];
	cf_q[1] = cf_q[1] + qdot[1];
	cf_q[2] = cf_q[2] + qdot[2];
	cf_q[3] = cf_q[3] + qdot[3];

	if(cf_q[0] < 0) {
		cf_q[0] = -cf_q[0];
		cf_q[1] = -cf_q[1];
		cf_q[2] = -cf_q[2];
		cf_q[3] = -cf_q[3];
	}

	// Renomalize
	float qmag;
	qmag = sqrtf(cf_q[0]*cf_q[0] + cf_q[1]*cf_q[1] + cf_q[2]*cf_q[2] + cf_q[3]*cf_q[3]);
	cf_q[0] = cf_q[0] / qmag;
	cf_q[1] = cf_q[1] / qmag;
	cf_q[2] = cf_q[2] / qmag;
	cf_q[3] = cf_q[3] / qmag;

	// If quaternion has become inappropriately short or is nan reinit.
	// THIS SHOULD NEVER ACTUALLY HAPPEN
	if((fabsf(qmag) < 1.0e-3f) || (qmag != qmag)) {
		cf_q[0] = 1;
		cf_q[1] = 0;
		cf_q[2] = 0;
		cf_q[3] = 0;
	}

	if (!secondary)
		AlarmsClear(SYSTEMALARMS_ALARM_ATTITUDE);

	return 0;
}

//! Set the navigation information to the raw estimates
static int32_t setNavigationRaw()
{
	UAVObjEvent ev;

	// Flush these queues for avoid errors
	xQueueReceive(baroQueue, &ev, 0);
	if ( xQueueReceive(gpsQueue, &ev, 0) == pdTRUE && homeLocation.Set == HOMELOCATION_SET_TRUE ) {
		float NED[3];
		// Transform the GPS position into NED coordinates
		GPSPositionData gpsPosition;
		GPSPositionGet(&gpsPosition);
		getNED(&gpsPosition, NED);

		NEDPositionData nedPosition;
		NEDPositionGet(&nedPosition);
		nedPosition.North = NED[0];
		nedPosition.East = NED[1];
		nedPosition.Down = NED[2];
		NEDPositionSet(&nedPosition);

		PositionActualData positionActual;
		PositionActualGet(&positionActual);
		positionActual.North = NED[0];
		positionActual.East = NED[1];
		positionActual.Down = NED[2];
		PositionActualSet(&positionActual);
	}

	if ( xQueueReceive(gpsVelQueue, &ev, 0) == pdTRUE ) {
		// Transform the GPS position into NED coordinates
		GPSVelocityData gpsVelocity;
		GPSVelocityGet(&gpsVelocity);

		VelocityActualData velocityActual;
		VelocityActualGet(&velocityActual);
		velocityActual.North = gpsVelocity.North;
		velocityActual.East = gpsVelocity.East;
		velocityActual.Down = gpsVelocity.Down;
		VelocityActualSet(&velocityActual);
	}

	return 0;
}

/**
 * Set the @ref AttitudeActual UAVO to the complementary filter
 * estimate
 */
static int32_t setAttitudeComplementary()
{
	AttitudeActualData attitude;
	quat_copy(cf_q, &attitude.q1);
	Quaternion2RPY(&attitude.q1,&attitude.Roll);
	AttitudeActualSet(&attitude);

	return 0;
}

/**
 * If accumulating data and enough samples acquired then recompute
 * the gyro bias based on the mean accumulated
 */
static void accumulate_gyro_compute()
{
	if (complementary_filter_state.accumulating_gyro && 
		complementary_filter_state.accumulated_gyro_samples > 100) {

		// Accumulate integral of error.  Scale here so that units are (deg/s) but Ki has units of s
		GyrosBiasData gyrosBias;
		GyrosBiasGet(&gyrosBias);
		gyrosBias.x = complementary_filter_state.accumulated_gyro[0] / complementary_filter_state.accumulated_gyro_samples;
		gyrosBias.y = complementary_filter_state.accumulated_gyro[1] / complementary_filter_state.accumulated_gyro_samples;
		gyrosBias.z = complementary_filter_state.accumulated_gyro[2] / complementary_filter_state.accumulated_gyro_samples;
		GyrosBiasSet(&gyrosBias);

		accumulate_gyro_zero();

		complementary_filter_state.accumulating_gyro = false;
	}
}

/**
 * Zero the accumulation of gyro data
 */
static void accumulate_gyro_zero()
{
	complementary_filter_state.accumulated_gyro_samples = 0;
	complementary_filter_state.accumulated_gyro[0] = 0;
	complementary_filter_state.accumulated_gyro[1] = 0;
	complementary_filter_state.accumulated_gyro[2] = 0;
}

/**
 * Accumulate a set of gyro samples for computing the
 * bias
 * @param [in] gyrosData The samples of data to accumulate
 */
static void accumulate_gyro(GyrosData *gyrosData)
{
	if (!complementary_filter_state.accumulating_gyro)
		return;

	complementary_filter_state.accumulated_gyro_samples++;

	// bias_correct_gyro
	if (true) {
		// Apply bias correction to the gyros from the state estimator
		GyrosBiasData gyrosBias;
		GyrosBiasGet(&gyrosBias);

		complementary_filter_state.accumulated_gyro[0] += gyrosData->x + gyrosBias.x;
		complementary_filter_state.accumulated_gyro[1] += gyrosData->y + gyrosBias.y;
		complementary_filter_state.accumulated_gyro[2] += gyrosData->z + gyrosBias.z;
	} else {
		complementary_filter_state.accumulated_gyro[0] += gyrosData->x;
		complementary_filter_state.accumulated_gyro[1] += gyrosData->y;
		complementary_filter_state.accumulated_gyro[2] += gyrosData->z;
	}
}


#include "insgps.h"
static bool home_location_updated;
/**
 * @brief Use the INSGPS fusion algorithm in either indoor or outdoor mode (use GPS)
 * @params[in] first_run This is the first run so trigger reinitialization
 * @params[in] outdoor_mode If true use the GPS for position, if false weakly pull to (0,0)
 * @return 0 for success, -1 for failure
 */
static int32_t updateAttitudeINSGPS(bool first_run, bool outdoor_mode)
{
	UAVObjEvent ev;
	GyrosData gyrosData;
	AccelsData accelsData;
	MagnetometerData magData;
	GPSVelocityData gpsVelData;
	GyrosBiasData gyrosBias;

	// These should be static as their values are checked multiple times per update
	static BaroAltitudeData baroData;
	static GPSPositionData gpsData;

	static bool mag_updated = false;
	static bool baro_updated;
	static bool gps_updated;
	static bool gps_vel_updated;

	static float baro_offset = 0;

	static uint32_t ins_last_time = 0;
	static bool inited;

	float NED[3] = {0.0f, 0.0f, 0.0f};
	float vel[3] = {0.0f, 0.0f, 0.0f};

	// Perform the update
	uint16_t sensors = 0;
	float dT;

	// When the home location is adjusted the filter should be
	// reinitialized to correctly offset the baro and make sure it 
	// does not blow up.  This flag should only be set when not armed.
	if (first_run || home_location_updated) {
		inited = false;

		mag_updated = 0;
		baro_updated = 0;
		gps_updated = 0;
		gps_vel_updated = 0;

		home_location_updated = false;

		ins_last_time = PIOS_DELAY_GetRaw();

		return 0;
	}

	mag_updated |= (xQueueReceive(magQueue, &ev, MS2TICKS(0)) == pdTRUE);
	baro_updated |= xQueueReceive(baroQueue, &ev, MS2TICKS(0)) == pdTRUE;
	gps_updated |= (xQueueReceive(gpsQueue, &ev, MS2TICKS(0)) == pdTRUE) && outdoor_mode;
	gps_vel_updated |= (xQueueReceive(gpsVelQueue, &ev, MS2TICKS(0)) == pdTRUE) && outdoor_mode;

	// Wait until the gyro and accel object is updated, if a timeout then go to failsafe
	if ( (xQueueReceive(gyroQueue, &ev, MS2TICKS(FAILSAFE_TIMEOUT_MS)) != pdTRUE) ||
		 (xQueueReceive(accelQueue, &ev, MS2TICKS(1)) != pdTRUE) )
	{
		return -1;
	}

	// Get most recent data
	GyrosGet(&gyrosData);
	AccelsGet(&accelsData);
	GyrosBiasGet(&gyrosBias);

	// Need to get these values before initializing
	if (mag_updated)
       MagnetometerGet(&magData);

	if (gps_updated)
		GPSPositionGet(&gpsData);

	// Discard mag if it has NAN (normally from bad calibration)
	mag_updated &= (magData.x == magData.x && magData.y == magData.y && magData.z == magData.z);

	// Don't require HomeLocation.Set to be true but at least require a mag configuration (allows easily
	// switching between indoor and outdoor mode with Set = false)
	mag_updated &= (homeLocation.Be[0] != 0 || homeLocation.Be[1] != 0 || homeLocation.Be[2]);

	// A more stringent requirement for GPS to initialize the filter
	bool gps_init_usable = gps_updated & (gpsData.Satellites >= 7) && (gpsData.PDOP <= 3.5f) && (homeLocation.Set == HOMELOCATION_SET_TRUE);

	if (!inited)
		AlarmsSet(SYSTEMALARMS_ALARM_ATTITUDE,SYSTEMALARMS_ALARM_ERROR);
	else if (outdoor_mode && (gpsData.Satellites < 6 || gpsData.PDOP > 4.0f))
		AlarmsSet(SYSTEMALARMS_ALARM_ATTITUDE,SYSTEMALARMS_ALARM_ERROR);
	else
		AlarmsClear(SYSTEMALARMS_ALARM_ATTITUDE);

	if (!inited && mag_updated && baro_updated && (gps_init_usable || !outdoor_mode)) {

		INSGPSInit();
		INSSetMagVar(insSettings.mag_var);
		INSSetAccelVar(insSettings.accel_var);
		INSSetGyroVar(insSettings.gyro_var);
		INSSetBaroVar(insSettings.baro_var);

		// Set initial variances, selected by trial and error
		float Pdiag[16]={25.0f,25.0f,25.0f,5.0f,5.0f,5.0f,1e-5f,1e-5f,1e-5f,1e-5f,1e-5f,1e-5f,1e-5f,1e-4f,1e-4f,1e-4f};
		INSResetP(Pdiag);

		// Initialize the gyro bias from the settings
		float gyro_bias[3] = {gyrosBias.x * DEG2RAD, gyrosBias.y * DEG2RAD, gyrosBias.z * DEG2RAD};
		INSSetGyroBias(gyro_bias);

		BaroAltitudeGet(&baroData);

		float RPY[3], q[4];
		RPY[0] = atan2f(-accelsData.y, -accelsData.z) * RAD2DEG;
		RPY[1] = atan2f(accelsData.x, -accelsData.z) * RAD2DEG;
		RPY[2] = atan2f(-magData.y, magData.x) * RAD2DEG;
		RPY2Quaternion(RPY,q);

		// Don't initialize until all sensors are read
		if (!outdoor_mode) {
			float pos[3] = {0.0f, 0.0f, 0.0f};

			// Initialize barometric offset to current altitude
			baro_offset = -baroData.Altitude;
			pos[2] = -(baroData.Altitude + baro_offset);

			// Hard coded fake variances for indoor mode
			INSSetPosVelVar(0.1f, 0.1f, 0.1f);

			if (homeLocation.Set == HOMELOCATION_SET_TRUE)
				INSSetMagNorth(homeLocation.Be);
			else {
				// Reasonable default is safe for indoor
				float Be[3] = {100,0,500};
				INSSetMagNorth(Be);
			}

			INSSetState(pos, zeros, q, zeros, zeros);
		} else {
			float NED[3];

			// Use the UAVO for the position variance	
			INSSetPosVelVar(insSettings.gps_var[INSSETTINGS_GPS_VAR_POS], insSettings.gps_var[INSSETTINGS_GPS_VAR_VEL], insSettings.gps_var[INSSETTINGS_GPS_VAR_VERTPOS]);
			INSSetMagNorth(homeLocation.Be);

			// Initialize the gyro bias from the settings
			float gyro_bias[3] = {gyrosBias.x * DEG2RAD, gyrosBias.y * DEG2RAD, gyrosBias.z * DEG2RAD};
			INSSetGyroBias(gyro_bias);

			// Initialize to current location
			getNED(&gpsData, NED);

			// Initialize barometric offset to cirrent GPS NED coordinate
			baro_offset = -NED[2] - baroData.Altitude;

			INSSetState(NED, zeros, q, zeros, zeros);
		} 

		inited = true;

		ins_last_time = PIOS_DELAY_GetRaw();	

		return 0;
	}

	if (!inited)
		return 0;

	// Have a minimum requirement for gps usage a little more liberal than initialization
	gps_updated &= (gpsData.Satellites >= 6) && (gpsData.PDOP <= 4.0f) && (homeLocation.Set == HOMELOCATION_SET_TRUE);

	dT = PIOS_DELAY_DiffuS(ins_last_time) / 1.0e6f;
	ins_last_time = PIOS_DELAY_GetRaw();

	// This should only happen at start up or at mode switches
	if(dT > 0.01f)
		dT = 0.01f;
	else if(dT <= 0.001f)
		dT = 0.001f;

	// If the gyro bias setting was updated we should reset
	// the state estimate of the EKF
	if(gyroBiasSettingsUpdated) {
		float gyro_bias[3] = {gyrosBias.x * DEG2RAD, gyrosBias.y * DEG2RAD, gyrosBias.z * DEG2RAD};
		INSSetGyroBias(gyro_bias);
		gyroBiasSettingsUpdated = false;
	}

	// Because the sensor module remove the bias we need to add it
	// back in here so that the INS algorithm can track it correctly
	float gyros[3] = {gyrosData.x * DEG2RAD, gyrosData.y * DEG2RAD, gyrosData.z * DEG2RAD};
	if (insSettings.ComputeGyroBias == INSSETTINGS_COMPUTEGYROBIAS_TRUE && 
	    (attitudeSettings.BiasCorrectGyro == ATTITUDESETTINGS_BIASCORRECTGYRO_TRUE)) {
		gyros[0] += gyrosBias.x * DEG2RAD;
		gyros[1] += gyrosBias.y * DEG2RAD;
		gyros[2] += gyrosBias.z * DEG2RAD;
	} else {
		INSSetGyroBias(zeros);
	}

	// Advance the state estimate
	INSStatePrediction(gyros, &accelsData.x, dT);

	// Advance the covariance estimate
	INSCovariancePrediction(dT);

	if(mag_updated) {
		sensors |= MAG_SENSORS;
		mag_updated = false;
	}
	
	if(baro_updated) {
		sensors |= BARO_SENSOR;
		BaroAltitudeGet(&baroData);
		baro_updated = false;
	}

	// GPS Position update
	if (gps_updated && outdoor_mode)
	{
		sensors |= HORIZ_POS_SENSORS | VERT_POS_SENSORS;

		// Transform the GPS position into NED coordinates
		getNED(&gpsData, NED);

		// Track barometric altitude offset with a low pass filter
		baro_offset = BARO_OFFSET_LOWPASS_ALPHA * baro_offset +
		    (1.0f - BARO_OFFSET_LOWPASS_ALPHA )
		    * ( -NED[2] - baroData.Altitude );

		// Store this for inspecting offline
		NEDPositionData nedPos;
		NEDPositionGet(&nedPos);
		nedPos.North = NED[0];
		nedPos.East = NED[1];
		nedPos.Down = NED[2];
		NEDPositionSet(&nedPos);

		gps_updated = false;
	}

	// GPS Velocity update
	if (gps_vel_updated && outdoor_mode) {
		sensors |= HORIZ_SENSORS | VERT_SENSORS;
		GPSVelocityGet(&gpsVelData);
		vel[0] = gpsVelData.North;
		vel[1] = gpsVelData.East;
		vel[2] = gpsVelData.Down;

		gps_vel_updated = false;
	}

	// Update fake position at 10 hz
	static uint32_t indoor_pos_time;
	if (!outdoor_mode && PIOS_DELAY_DiffuS(indoor_pos_time) > 100000) {
		indoor_pos_time = PIOS_DELAY_GetRaw();
		vel[0] = vel[1] = vel[2] = 0;
		NED[0] = NED[1] = 0;
		NED[2] = -(baroData.Altitude + baro_offset);
		sensors |= HORIZ_SENSORS | HORIZ_POS_SENSORS;
		sensors |= VERT_SENSORS;
	}

	/*
	 * TODO: Need to add a general sanity check for all the inputs to make sure their kosher
	 * although probably should occur within INS itself
	 */
	if (sensors)
		INSCorrection(&magData.x, NED, vel, ( baroData.Altitude + baro_offset ), sensors);

	// Export the state and variance for monitoring the EKF
	INSStateData state;
	INSGetVariance(state.Var);
	INSGetState(&state.State[0], &state.State[3], &state.State[6], &state.State[10]);
	INSStateSet(&state);

	return 0;
}

//! Set the attitude to the current INSGPS estimate
static int32_t setAttitudeINSGPS()
{
	float gyro_bias[3];
	AttitudeActualData attitude;

	INSGetState(NULL, NULL, &attitude.q1, gyro_bias);
	Quaternion2RPY(&attitude.q1,&attitude.Roll);
	AttitudeActualSet(&attitude);

	if (insSettings.ComputeGyroBias == INSSETTINGS_COMPUTEGYROBIAS_TRUE && 
	    (attitudeSettings.BiasCorrectGyro == ATTITUDESETTINGS_BIASCORRECTGYRO_TRUE && 
	    !gyroBiasSettingsUpdated)) {
		// Copy the gyro bias into the UAVO except when it was updated
		// from the settings during the calculation, then consume it
		// next cycle
		GyrosBiasData gyrosBias;
		gyrosBias.x = gyro_bias[0] * RAD2DEG;
		gyrosBias.y = gyro_bias[1] * RAD2DEG;
		gyrosBias.z = gyro_bias[2] * RAD2DEG;
		GyrosBiasSet(&gyrosBias);
	}

	return 0;
}

//! Set the navigation to the current INSGPS estimate
static int32_t setNavigationINSGPS()
{
	PositionActualData positionActual;
	VelocityActualData velocityActual;

	INSGetState(&positionActual.North, &velocityActual.North, NULL, NULL);

	PositionActualSet(&positionActual);
	VelocityActualSet(&velocityActual);

	return 0;
}

static void apply_accel_filter(const float * raw, float * filtered)
{
	const float alpha = complementary_filter_state.accel_alpha;
	if(complementary_filter_state.accel_filter_enabled) {
		filtered[0] = filtered[0] * alpha + raw[0] * (1 - alpha);
		filtered[1] = filtered[1] * alpha + raw[1] * (1 - alpha);
		filtered[2] = filtered[2] * alpha + raw[2] * (1 - alpha);
	} else {
		filtered[0] = raw[0];
		filtered[1] = raw[1];
		filtered[2] = raw[2];
	}
}

/**
 * @brief Convert the GPS LLA position into NED coordinates
 * @note this method uses a taylor expansion around the home coordinates
 * to convert to NED which allows it to be done with all floating
 * calculations
 * @param[in] Current GPS coordinates
 * @param[out] NED frame coordinates
 * @returns 0 for success, -1 for failure
 */
float T[3];
static int32_t getNED(GPSPositionData * gpsPosition, float * NED)
{
	float dL[3] = {(gpsPosition->Latitude - homeLocation.Latitude) / 10.0e6f * DEG2RAD,
		(gpsPosition->Longitude - homeLocation.Longitude) / 10.0e6f * DEG2RAD,
		(gpsPosition->Altitude + gpsPosition->GeoidSeparation - homeLocation.Altitude)};

	NED[0] = T[0] * dL[0];
	NED[1] = T[1] * dL[1];
	NED[2] = T[2] * dL[2];

	return 0;
}

static void settingsUpdatedCb(UAVObjEvent * ev) 
{
	if (ev == NULL || ev->obj == SensorSettingsHandle()) {
		SensorSettingsData sensorSettings;
		SensorSettingsGet(&sensorSettings);
		
		/* When the revo calibration is updated, update the GyroBias object */
		GyrosBiasData gyrosBias;
		GyrosBiasGet(&gyrosBias);
		gyrosBias.x = 0;
		gyrosBias.y = 0;
		gyrosBias.z = 0;
		GyrosBiasSet(&gyrosBias);

		gyroBiasSettingsUpdated = true;
	}
	if (ev == NULL || ev->obj == INSSettingsHandle()) {
		INSSettingsGet(&insSettings);
		// In case INS currently running
		INSSetMagVar(insSettings.mag_var);
		INSSetAccelVar(insSettings.accel_var);
		INSSetGyroVar(insSettings.gyro_var);
		INSSetBaroVar(insSettings.baro_var);
	}
	if(ev == NULL || ev->obj == HomeLocationHandle()) {
		uint8_t armed;
		FlightStatusArmedGet(&armed);

		// Do not update the home location while armed as this can blow up the 
		// filter.  This will need to be overhauled to handle long distance
		// flights
		if (armed == FLIGHTSTATUS_ARMED_DISARMED) {
			HomeLocationGet(&homeLocation);
			// Compute matrix to convert deltaLLA to NED
			float lat, alt;
			lat = homeLocation.Latitude / 10.0e6f * DEG2RAD;
			alt = homeLocation.Altitude;

			T[0] = alt+6.378137E6f;
			T[1] = cosf(lat)*(alt+6.378137E6f);
			T[2] = -1.0f;

			home_location_updated = true;
		}
	}
	if (ev == NULL || ev->obj == AttitudeSettingsHandle()) {
		AttitudeSettingsGet(&attitudeSettings);
			
		// Calculate accel filter alpha, in the same way as for gyro data in stabilization module.
		const float fakeDt = 0.0025f;
		if(attitudeSettings.AccelTau < 0.0001f) {
			complementary_filter_state.accel_alpha = 0;   // not trusting this to resolve to 0
			complementary_filter_state.accel_filter_enabled = false;
		} else {
			complementary_filter_state.accel_alpha = expf(-fakeDt  / attitudeSettings.AccelTau);
			complementary_filter_state.accel_filter_enabled = true;
		}
	}
	if (ev == NULL || ev->obj == StateEstimationHandle())
		StateEstimationGet(&stateEstimation);
}
/**
 * @}
 * @}
 */
