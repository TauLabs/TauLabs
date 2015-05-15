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
 * @author     Tau Labs, http://taulabs.org Copyright (C) 2012-2014
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
 * filter algorithm for just attitude. In complementary mode it also runs a
 * simple filter to smooth the altitude.
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
#include "gpstime.h"
#include "gpsvelocity.h"
#include "gyros.h"
#include "gyrosbias.h"
#include "homelocation.h"
#include "sensorsettings.h"
#include "inssettings.h"
#include "insstate.h"
#include "magnetometer.h"
#include "nedaccel.h"
#include "nedposition.h"
#include "positionactual.h"
#include "stateestimation.h"
#include "systemalarms.h"
#include "velocityactual.h"
#include "coordinate_conversions.h"
#include "WorldMagModel.h"
#include "pios_thread.h"
#include "pios_queue.h"

// Private constants
#define STACK_SIZE_BYTES 2200
#define TASK_PRIORITY PIOS_THREAD_PRIO_HIGH
#define FAILSAFE_TIMEOUT_MS 10

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

//! Struture that tracks the data for the vertical complementary filter
struct cfvert {
	float velocity_z;
	float position_z;
	float time_constant_z;
	float accel_correction_z;
	float position_base_z;
	float position_error_z;
	float position_correction_z;
	float baro_zero;
};

// Private variables
static struct pios_thread *attitudeTaskHandle;

static struct pios_queue *gyroQueue;
static struct pios_queue *accelQueue;
static struct pios_queue *magQueue;
static struct pios_queue *baroQueue;
static struct pios_queue *gpsQueue;
static struct pios_queue *gpsVelQueue;

static AttitudeSettingsData attitudeSettings;
static HomeLocationData homeLocation;
static INSSettingsData insSettings;
static StateEstimationData stateEstimation;
static bool gyroBiasSettingsUpdated = false;
const uint32_t SENSOR_QUEUE_SIZE = 10;
static const float zeros[3] = {0.0f, 0.0f, 0.0f};

static struct complementary_filter_state complementary_filter_state;
static struct cfvert cfvert; //!< State information for vertical filter

// Private functions
static void AttitudeTask(void *parameters);

//! Set the navigation information to the raw estimates
static int32_t setNavigationRaw();

//! Provide no navigation updates (indoor flying or without gps)
static int32_t setNavigationNone();

//! Update the complementary filter attitude estimate
static int32_t updateAttitudeComplementary(bool first_run, bool secondary, bool raw_gps);
//! Set the @ref AttitudeActual to the complementary filter estimate
static int32_t setAttitudeComplementary();

static float calc_ned_accel(float *q, float *accels);
static void cfvert_reset(struct cfvert *cf, float baro, float time_constant);
static void cfvert_predict_pos(struct cfvert *cf, float z_accel, float dt);
static void cfvert_update_baro(struct cfvert *cf, float baro, float dt);

//! Update the INSGPS attitude estimate
static int32_t updateAttitudeINSGPS(bool first_run, bool outdoor_mode);
//! Set the attitude to the current INSGPS estimate
static int32_t setAttitudeINSGPS();
//! Set the navigation to the current INSGPS estimate
static int32_t setNavigationINSGPS();
static void updateNedAccel();
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

//! Set alarm and alarm code
static void set_state_estimation_error(SystemAlarmsStateEstimationOptions error_code);

//! Determine if it is safe to set the home location then do it
static void check_home_location();

/**
 * API for sensor fusion algorithms:
 * Configure(struct pios_queue *gyro, struct pios_queue *accel, struct pios_queue *mag, struct pios_queue *baro)
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
	NedAccelInitialize();
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
	gyroQueue = PIOS_Queue_Create(1, sizeof(UAVObjEvent));
	accelQueue = PIOS_Queue_Create(1, sizeof(UAVObjEvent));
	magQueue = PIOS_Queue_Create(2, sizeof(UAVObjEvent));
	baroQueue = PIOS_Queue_Create(1, sizeof(UAVObjEvent));
	gpsQueue = PIOS_Queue_Create(1, sizeof(UAVObjEvent));
	gpsVelQueue = PIOS_Queue_Create(1, sizeof(UAVObjEvent));

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
	attitudeTaskHandle = PIOS_Thread_Create(AttitudeTask, "Attitude", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
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
	set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_UNDEFINED);

	// Force settings update to make sure rotation loaded
	settingsUpdatedCb(NULL);

	// Wait for all the sensors be to read
	PIOS_Thread_Sleep(100);

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

		// Determine if we can set the home location. This is done here to share the stack
		// space with the INS which is the largest stack on the code.
		check_home_location();

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
				 updateAttitudeComplementary(first_run || complementary != last_complementary,
				                               true,     // the secondary filter
				                               false);   // no raw gps is used
		} else {
			ret_val = updateAttitudeComplementary(first_run,
			                                       false,
			                                       stateEstimation.NavigationFilter == STATEESTIMATION_NAVIGATIONFILTER_RAW);
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
		case STATEESTIMATION_NAVIGATIONFILTER_RAW:
			setNavigationRaw();
			break;
		case STATEESTIMATION_NAVIGATIONFILTER_NONE:
		default:
			setNavigationNone();
			break;
		}

		updateNedAccel();

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
static int32_t updateAttitudeComplementary(bool first_run, bool secondary, bool raw_gps)
{
	UAVObjEvent ev;
	GyrosData gyrosData;
	AccelsData accelsData;
	static int32_t timeval;
	float dT;

	// If this is the primary estimation filter, wait until the accel and
	// gyro objects are updated. If it timeouts then go to failsafe.
	if (!secondary) {
		bool gyroTimeout  = PIOS_Queue_Receive(gyroQueue, &ev, FAILSAFE_TIMEOUT_MS) != true;
		bool accelTimeout = PIOS_Queue_Receive(accelQueue, &ev, 1) != true;

		// When one of these is updated so should the other.
		if (gyroTimeout || accelTimeout) {
			// Do not set attitude timeout warnings in simulation mode
			if (!AttitudeActualReadOnly()) {
				if (gyroTimeout)
					set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_GYROQUEUENOTUPDATING);
				else if (accelTimeout)
					set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_ACCELEROMETERQUEUENOTUPDATING);
				else
					set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_UNDEFINED);

				return -1;
			}
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
			if (!secondary && PIOS_Queue_Receive(magQueue, &ev, 20) != true) {
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

		float baro;
		BaroAltitudeAltitudeGet(&baro);
		cfvert_reset(&cfvert, baro, attitudeSettings.VertPositionTau);

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
		attitudeSettings.AccelKp = 0.1f + 0.1f * (PIOS_Thread_Systime() < 4000);
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

		// Reset the filter for barometric data
		float baro;
		BaroAltitudeAltitudeGet(&baro);
		cfvert_reset(&cfvert, baro, attitudeSettings.VertPositionTau);

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

		// Reset the filter for barometric data
		float baro;
		BaroAltitudeAltitudeGet(&baro);
		cfvert_reset(&cfvert, baro, attitudeSettings.VertPositionTau);

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
	if (secondary || PIOS_Queue_Receive(magQueue, &ev, 0) == true)
	{
		MagnetometerData mag;
		MagnetometerGet(&mag);

		// If the mag is producing bad data (NAN) don't use it (normally bad calibration)
		if  (mag.x == mag.x && mag.y == mag.y && mag.z == mag.z) {
			float bmag = 1.0f;
			float brot[3];
			float Rbe[3][3];

			// Get rotation to bring earth magnetic field into body frame		
			Quaternion2R(cf_q, Rbe);

			if (homeLocation.Set == HOMELOCATION_SET_TRUE) {
				rot_mult(Rbe, homeLocation.Be, brot, false);
				bmag = sqrtf(brot[0] * brot[0] + brot[1] * brot[1] + brot[2] * brot[2]);
				brot[0] /= bmag;
				brot[1] /= bmag;
				brot[2] /= bmag;
			} else {
				const float Be[3] = {1.0f, 0.0f, 0.0f};
				rot_mult(Rbe, Be, brot, false);
			}

			float mag_len = sqrtf(mag.x * mag.x + mag.y * mag.y + mag.z * mag.z);
			mag.x /= mag_len;
			mag.y /= mag_len;
			mag.z /= mag_len;

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
	gyrosData.z += accel_err[2] * attitudeSettings.AccelKp / dT + mag_err[2] * attitudeSettings.MagKp / dT;

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

	if (!secondary) {

		// Calculate the NED acceleration and get the z-component
		float z_accel = calc_ned_accel(cf_q, &accelsData.x);

		// When this is the only filter compute th vertical state from baro data
		// Reset the filter for barometric data
		cfvert_predict_pos(&cfvert, z_accel, dT);
		if (PIOS_Queue_Receive(baroQueue, &ev, 0) == true) {
			float baro;
			BaroAltitudeAltitudeGet(&baro);
			cfvert_update_baro(&cfvert, baro, dT);
		}

	}
	if (!secondary && !raw_gps) {
		// When in raw GPS mode, it will set the error to none if
		// reception is good
		set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_NONE);
	}

	return 0;
}

/**
 * Calculate the acceleration in the NED frame. This is used
 * by the altitude controller. Returns the down component for
 * convenience.
 */
static float calc_ned_accel(float *q, float *accels)
{
	float accel_ned[3];
	float Rbe[3][3];

	// rotate the accels into the NED frame and remove
	// the influence of gravity
	Quaternion2R(q, Rbe);
	rot_mult(Rbe, accels, accel_ned, true);
	accel_ned[2] += GRAVITY;

	NedAccelData nedAccel;
	nedAccel.North = accel_ned[0];
	nedAccel.East = accel_ned[1];
	nedAccel.Down = accel_ned[2];
	NedAccelSet(&nedAccel);

	return accel_ned[2];
}

//! Resets the vertical baro complementary filter and zeros the altitude
static void cfvert_reset(struct cfvert *cf, float baro, float time_constant)
{
	cf->velocity_z = 0;
	cf->position_z = 0;
	cf->time_constant_z = time_constant;
	cf->accel_correction_z = 0;
	cf->position_base_z = 0;
	cf->position_error_z = 0;
	cf->position_correction_z = 0;
	cf->baro_zero = baro;
}

//! Predict the position in the future
static void cfvert_predict_pos(struct cfvert *cf, float z_accel, float dt)
{
	float k1_z = 3 / cf->time_constant_z;
	float k2_z = 3 / powf(cf->time_constant_z, 2);
	float k3_z = 1 / powf(cf->time_constant_z, 3);

	cf->accel_correction_z += cf->position_error_z * k3_z * dt;
	cf->velocity_z += cf->position_error_z * k2_z * dt;
	cf->position_correction_z += cf->position_error_z * k1_z * dt;

	float velocity_increase;
	velocity_increase = (z_accel + cf->accel_correction_z) * dt;
	cf->position_base_z += (cf->velocity_z + velocity_increase * 0.5f) * dt;
	cf->position_z = cf->position_base_z + cf->position_correction_z;
	cf->velocity_z += velocity_increase;
}

//! Update the baro feedback
static void cfvert_update_baro(struct cfvert *cf, float baro, float dt)
{
	float down = -(baro - cf->baro_zero);

	// TODO: get from a queue of previous position updates (150 ms latency)
	float hist_position_base_d = cf->position_base_z;

	cf->position_error_z = down - (hist_position_base_d + cf->position_correction_z);
}

//! Set the navigation information to the raw estimates
static int32_t setNavigationRaw()
{
	UAVObjEvent ev;

	if (homeLocation.Set == HOMELOCATION_SET_FALSE) {
		set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_NOHOME);
		PIOS_Queue_Receive(gpsQueue, &ev, 0);
	} else if (PIOS_Queue_Receive(gpsQueue, &ev, 0) == true) {
		float NED[3];
		// Transform the GPS position into NED coordinates
		GPSPositionData gpsPosition;
		GPSPositionGet(&gpsPosition);

		if (gpsPosition.Satellites < 6)
			set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_TOOFEWSATELLITES);
		else if (gpsPosition.PDOP > 4.0f)
			set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_PDOPTOOHIGH);
		else
			set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_NONE);

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
		positionActual.Down = cfvert.position_z;
		PositionActualSet(&positionActual);
	} else {
		PositionActualDownSet(&cfvert.position_z);
	}

	if (PIOS_Queue_Receive(gpsVelQueue, &ev, 0) == true) {
		// Transform the GPS position into NED coordinates
		GPSVelocityData gpsVelocity;
		GPSVelocityGet(&gpsVelocity);

		VelocityActualData velocityActual;
		VelocityActualGet(&velocityActual);
		velocityActual.North = gpsVelocity.North;
		velocityActual.East = gpsVelocity.East;
		velocityActual.Down = cfvert.velocity_z;
		VelocityActualSet(&velocityActual);
	} else {
		VelocityActualDownSet(&cfvert.velocity_z);
	}

	return 0;
}

//! Set the navigation information to the raw estimates
static int32_t setNavigationNone()
{
	UAVObjEvent ev;

	// Throw away data to prevent queue overflows
	PIOS_Queue_Receive(gpsQueue, &ev, 0);
	PIOS_Queue_Receive(gpsVelQueue, &ev, 0);

	PositionActualDownSet(&cfvert.position_z);
	VelocityActualDownSet(&cfvert.velocity_z);

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
	static uint32_t ins_init_time = 0;

	static enum {INS_INIT, INS_WARMUP, INS_RUNNING} ins_state;

	float NED[3] = {0.0f, 0.0f, 0.0f};
	float vel[3] = {0.0f, 0.0f, 0.0f};

	// Perform the update
	uint16_t sensors = 0;
	float dT;

	// When the home location is adjusted the filter should be
	// reinitialized to correctly offset the baro and make sure it 
	// does not blow up.  This flag should only be set when not armed.
	if (first_run || home_location_updated) {
		ins_state = INS_INIT;

		mag_updated = false;
		baro_updated = false;
		gps_updated = false;
		gps_vel_updated = false;

		home_location_updated = false;

		ins_last_time = PIOS_DELAY_GetRaw();

		return 0;
	}

	mag_updated = mag_updated || PIOS_Queue_Receive(magQueue, &ev, 0);
	baro_updated = baro_updated || PIOS_Queue_Receive(baroQueue, &ev, 0);
	gps_updated = gps_updated || (PIOS_Queue_Receive(gpsQueue, &ev, 0) && outdoor_mode);
	gps_vel_updated = gps_vel_updated || (PIOS_Queue_Receive(gpsVelQueue, &ev, 0) && outdoor_mode);

	// Wait until the gyro and accel object is updated, if a timeout then go to failsafe
	if (PIOS_Queue_Receive(gyroQueue, &ev, FAILSAFE_TIMEOUT_MS) != true ||
		PIOS_Queue_Receive(accelQueue, &ev, 1) != true)
	{
		return -1;
	}

	// Get most recent data
	GyrosGet(&gyrosData);
	AccelsGet(&accelsData);
	GyrosBiasGet(&gyrosBias);

	// Need to get these values before initializing
	if(baro_updated)
		BaroAltitudeGet(&baroData);

	if (mag_updated)
       MagnetometerGet(&magData);

	if (gps_updated)
		GPSPositionGet(&gpsData);

	if (gps_vel_updated)
		GPSVelocityGet(&gpsVelData);

	// Discard mag if it has NAN (normally from bad calibration)
	mag_updated &= (magData.x == magData.x && magData.y == magData.y && magData.z == magData.z);

	// Indoor mode will fall back to reasonable Be and that is ok. For outdoor make sure home
	// Be is set and a good value
	mag_updated &= !outdoor_mode || (homeLocation.Be[0] != 0 || homeLocation.Be[1] != 0 || homeLocation.Be[2]);

	// A more stringent requirement for GPS to initialize the filter
	bool gps_init_usable = gps_updated & (gpsData.Satellites >= 7) && (gpsData.PDOP <= 3.5f) && (homeLocation.Set == HOMELOCATION_SET_TRUE);

	// Set user-friendly alarms appropriately based on state
	if (ins_state == INS_INIT) {
		if (!gps_init_usable && outdoor_mode)
			set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_NOGPS);
		else if (!mag_updated)
			set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_NOMAGNETOMETER);
		else if (!baro_updated)
			set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_NOBAROMETER);
		else
			set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_UNDEFINED);

	} else if (outdoor_mode && (gpsData.Satellites < 6 || gpsData.PDOP > 4.0f)) {
		if (gpsData.Satellites < 6)
			set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_TOOFEWSATELLITES);
		else if (gpsData.PDOP > 4.0f)
			set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_PDOPTOOHIGH);
		else if (homeLocation.Set == HOMELOCATION_SET_FALSE)
			set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_NOHOME);
		else
			set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_UNDEFINED);
	} else {
		set_state_estimation_error(SYSTEMALARMS_STATEESTIMATION_NONE);
	}

	if (ins_state == INS_INIT &&
	      mag_updated && baro_updated &&
	      (gps_init_usable || !outdoor_mode)) {

		INSGPSInit();
		INSSetMagVar(insSettings.MagVar);
		INSSetAccelVar(insSettings.AccelVar);
		INSSetGyroVar(insSettings.GyroVar);
		INSSetBaroVar(insSettings.BaroVar);
		INSSetPosVelVar(insSettings.GpsVar[INSSETTINGS_GPSVAR_POS], insSettings.GpsVar[INSSETTINGS_GPSVAR_VEL], insSettings.GpsVar[INSSETTINGS_GPSVAR_VERTPOS]);

		// Initialize the gyro bias from the settings
		float gyro_bias[3] = {gyrosBias.x * DEG2RAD, gyrosBias.y * DEG2RAD, gyrosBias.z * DEG2RAD};
		INSSetGyroBias(gyro_bias);
		INSSetAccelBias(zeros);

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

			if (homeLocation.Set == HOMELOCATION_SET_TRUE &&
			    (homeLocation.Be[0] != 0 || homeLocation.Be[1] != 0 || homeLocation.Be[2]))
			    // Use the configured mag, if one is available
				INSSetMagNorth(homeLocation.Be);
			else {
				// Reasonable default is safe for indoor
				float Be[3] = {100,0,500};
				INSSetMagNorth(Be);
			}

			INSSetState(pos, zeros, q, zeros, zeros);
		} else {
			float NED[3];

			INSSetMagNorth(homeLocation.Be);

			// Initialize the gyro bias from the settings
			float gyro_bias[3] = {gyrosBias.x * DEG2RAD, gyrosBias.y * DEG2RAD, gyrosBias.z * DEG2RAD};
			INSSetGyroBias(gyro_bias);

			// Initialize to current location
			getNED(&gpsData, NED);

			// Initialize barometric offset to current GPS NED coordinate
			baro_offset = -baroData.Altitude;

			INSSetState(NED, zeros, q, zeros, zeros);
		} 

		// Once all sensors have been updated and initialized then enter warmup
		// state to make sure filter converges
		ins_state = INS_WARMUP;

		ins_last_time = PIOS_DELAY_GetRaw();	
		ins_init_time = ins_last_time;

		return 0;
	} else if (ins_state == INS_INIT)
		return 0;

	// Keep in warmup for first 10 seconds. This zeros biases.
	if (ins_state == INS_WARMUP && PIOS_DELAY_DiffuS(ins_init_time) > 10e6f)
		ins_state = INS_RUNNING;

	// Let the filter know when we are armed
	uint8_t armed;
	FlightStatusArmedGet(&armed);
	INSSetArmed (armed == FLIGHTSTATUS_ARMED_ARMED);
	

	// Have a minimum requirement for gps usage a little more liberal than during initialization
	gps_updated &= (gpsData.Satellites >= 6) && (gpsData.PDOP <= 4.0f) && (homeLocation.Set == HOMELOCATION_SET_TRUE);

	dT = PIOS_DELAY_DiffuS(ins_last_time) / 1.0e6f;
	ins_last_time = PIOS_DELAY_GetRaw();

	// This should only happen at start up or at mode switches
	if(dT > 0.01f)
		dT = 0.01f;
	else if(dT <= 0.001f)
		dT = 0.001f;

	// When the sensor settings are updated, reset the biases. Also
	// while warming up, lock these at zero.
	if (gyroBiasSettingsUpdated || ins_state == INS_WARMUP) {
		gyroBiasSettingsUpdated = false;
		INSSetGyroBias(zeros);
		INSSetAccelBias(zeros);
	}

	// Because the sensor module remove the bias we need to add it
	// back in here so that the INS algorithm can track it correctly
	// this effectively means the INS is observing the "raw" data.
	float gyros[3];
	if (attitudeSettings.BiasCorrectGyro == ATTITUDESETTINGS_BIASCORRECTGYRO_TRUE) {
		gyros[0] = (gyrosData.x + gyrosBias.x) * DEG2RAD;
		gyros[1] = (gyrosData.y + gyrosBias.y) * DEG2RAD;
		gyros[2] = (gyrosData.z + gyrosBias.z) * DEG2RAD;
	} else {
		gyros[0] = gyrosData.x * DEG2RAD;
		gyros[1] = gyrosData.y * DEG2RAD;
		gyros[2] = gyrosData.z * DEG2RAD;
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
		baro_updated = false;
	}

	// GPS Position update
	if (gps_updated) { // only sets during outdoor mode
		sensors |= HORIZ_POS_SENSORS;

		// Transform the GPS position into NED coordinates
		getNED(&gpsData, NED);

		// Store this for inspecting offline
		NEDPositionData nedPos;
		nedPos.North = NED[0];
		nedPos.East = NED[1];
		nedPos.Down = NED[2];
		NEDPositionSet(&nedPos);

		gps_updated = false;
	}

	// GPS Velocity update
	if (gps_vel_updated) { // only sets during outdoor mode
		sensors |= HORIZ_VEL_SENSORS | VERT_VEL_SENSORS;

		vel[0] = gpsVelData.North;
		vel[1] = gpsVelData.East;
		vel[2] = gpsVelData.Down;

		gps_vel_updated = false;
	}

	// Update fake position at 10 hz
	static uint32_t indoor_pos_time;
	if (!outdoor_mode && PIOS_DELAY_DiffuS(indoor_pos_time) > 100000) {
		sensors |= HORIZ_VEL_SENSORS | HORIZ_POS_SENSORS;

		indoor_pos_time = PIOS_DELAY_GetRaw();
		vel[0] = vel[1] = vel[2] = 0;
		NED[0] = NED[1] = 0;
		NED[2] = -(baroData.Altitude + baro_offset);
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
	INSGetState(&state.State[0], &state.State[3], &state.State[6], &state.State[10], &state.State[13]);
	INSStateSet(&state); // this sets the UAVO

	if (insSettings.ComputeGyroBias == INSSETTINGS_COMPUTEGYROBIAS_FALSE)
		INSSetGyroBias(zeros);

	float accel_bias_corrected[3] = {accelsData.x - state.State[13], accelsData.y - state.State[14], accelsData.z - state.State[15]};
	calc_ned_accel(&state.State[6], accel_bias_corrected);

	return 0;
}

//! Set the attitude to the current INSGPS estimate
static int32_t setAttitudeINSGPS()
{
	float gyro_bias[3];
	AttitudeActualData attitude;

	INSGetState(NULL, NULL, &attitude.q1, gyro_bias, NULL);
	Quaternion2RPY(&attitude.q1,&attitude.Roll);
	AttitudeActualSet(&attitude);

	if (insSettings.ComputeGyroBias == INSSETTINGS_COMPUTEGYROBIAS_TRUE && 
	    !gyroBiasSettingsUpdated) {
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

	INSGetState(&positionActual.North, &velocityActual.North, NULL, NULL, NULL);

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
 * @param[in] Current lat-lon coordinates on WGS84 ellipsoid, altitude referenced to MSL geoid (likely EGM 1996, but no guarantees)
 * @param[out] NED frame coordinates
 * @returns 0 for success, -1 for failure
 */
float T[3];
static int32_t getNED(GPSPositionData * gpsPosition, float * NED)
{
	float dL[3] = {(gpsPosition->Latitude - homeLocation.Latitude) / 10.0e6f * DEG2RAD,
                   (gpsPosition->Longitude - homeLocation.Longitude) / 10.0e6f * DEG2RAD,
                   (gpsPosition->Altitude - homeLocation.Altitude)};

	NED[0] = T[0] * dL[0];
	NED[1] = T[1] * dL[1];
	NED[2] = T[2] * dL[2];

	return 0;
}

/**
 * Keep a running filtered version of the acceleration in the NED frame
 */
static void updateNedAccel()
{
	float accel[3];
	float q[4];
	float Rbe[3][3];
	float accel_ned[3];
	const float TAU = 0.95f;

	// Collect downsampled attitude data
	AccelsData accels;
	AccelsGet(&accels);		
	accel[0] = accels.x;
	accel[1] = accels.y;
	accel[2] = accels.z;
	
	//rotate avg accels into earth frame and store it
	AttitudeActualData attitudeActual;
	AttitudeActualGet(&attitudeActual);
	q[0]=attitudeActual.q1;
	q[1]=attitudeActual.q2;
	q[2]=attitudeActual.q3;
	q[3]=attitudeActual.q4;
	Quaternion2R(q, Rbe);
	for (uint8_t i = 0; i < 3; i++) {
		accel_ned[i] = 0;
		for (uint8_t j = 0; j < 3; j++)
			accel_ned[i] += Rbe[j][i] * accel[j];
	}
	accel_ned[2] += GRAVITY;
	
	NedAccelData accelData;
	NedAccelGet(&accelData);
	accelData.North = accelData.North * TAU + accel_ned[0] * (1 - TAU);
	accelData.East = accelData.East * TAU + accel_ned[1] * (1 - TAU);
	accelData.Down = accelData.Down * TAU + accel_ned[2] * (1 - TAU);
	NedAccelSet(&accelData);
}

/**
 * Check if it is safe to update the home location and do it
 */
static void check_home_location()
{
	// Do not attempt this calculation while armed
	uint8_t armed;
	FlightStatusArmedGet(&armed);
	if (armed != FLIGHTSTATUS_ARMED_DISARMED)
		return;

	// Do not calculate if already set
	if (homeLocation.Set == HOMELOCATION_SET_TRUE)
		return;

	GPSPositionData gps;
	GPSPositionGet(&gps);
	GPSTimeData gpsTime;
	GPSTimeGet(&gpsTime);
	
	// Check for valid data for the calculation
	if (gps.PDOP < 3.5f && 
	     gps.Satellites >= 7 &&
	     (gps.Status == GPSPOSITION_STATUS_FIX3D ||
	     gps.Status == GPSPOSITION_STATUS_DIFF3D) &&
	     gpsTime.Year >= 2000)
	{
		// Store LLA
		homeLocation.Latitude = gps.Latitude;
		homeLocation.Longitude = gps.Longitude;
		homeLocation.Altitude = gps.Altitude; // Altitude referenced to mean sea level geoid (likely EGM 1996, but no guarantees)

		// Compute home ECEF coordinates and the rotation matrix into NED
		double LLA[3] = { ((double)homeLocation.Latitude) / 10e6, ((double)homeLocation.Longitude) / 10e6, ((double)homeLocation.Altitude) };

		// Compute magnetic flux direction at home location
		if (WMM_GetMagVector(LLA[0], LLA[1], LLA[2], gpsTime.Month, gpsTime.Day, gpsTime.Year, &homeLocation.Be[0]) >= 0)
		{   // calculations appeared to go OK

			// Compute local acceleration due to gravity.  Vehicles that span a very large
			// range of altitude (say, weather balloons) may need to update this during the
			// flight.
			homeLocation.Set = HOMELOCATION_SET_TRUE;
			HomeLocationSet(&homeLocation);
		}
	}
}

static void settingsUpdatedCb(UAVObjEvent * ev) 
{
	if (ev == NULL || ev->obj == SensorSettingsHandle()) {
		SensorSettingsData sensorSettings;
		SensorSettingsGet(&sensorSettings);
		
		/* When the calibration is updated, update the GyroBias object */
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
		INSSetMagVar(insSettings.MagVar);
		INSSetAccelVar(insSettings.AccelVar);
		INSSetGyroVar(insSettings.GyroVar);
		INSSetBaroVar(insSettings.BaroVar);
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
 * Set the error code and alarm state
 * @param[in] error code
 */
static void set_state_estimation_error(SystemAlarmsStateEstimationOptions error_code)
{
	// Get the severity of the alarm given the error code
	SystemAlarmsAlarmOptions severity;
	switch (error_code) {
	case SYSTEMALARMS_STATEESTIMATION_NONE:
		severity = SYSTEMALARMS_ALARM_OK;
		break;
	case SYSTEMALARMS_STATEESTIMATION_ACCELEROMETERQUEUENOTUPDATING:
	case SYSTEMALARMS_STATEESTIMATION_GYROQUEUENOTUPDATING:
		severity = SYSTEMALARMS_ALARM_WARNING;
		break;
	case SYSTEMALARMS_STATEESTIMATION_NOGPS:
	case SYSTEMALARMS_STATEESTIMATION_NOMAGNETOMETER:
	case SYSTEMALARMS_STATEESTIMATION_NOBAROMETER:
	case SYSTEMALARMS_STATEESTIMATION_NOHOME:
	case SYSTEMALARMS_STATEESTIMATION_TOOFEWSATELLITES:
	case SYSTEMALARMS_STATEESTIMATION_PDOPTOOHIGH:
		severity = SYSTEMALARMS_ALARM_ERROR;
		break;
	case SYSTEMALARMS_STATEESTIMATION_UNDEFINED:
	default:
		severity = SYSTEMALARMS_ALARM_CRITICAL;
		error_code = SYSTEMALARMS_STATEESTIMATION_UNDEFINED;
		break;
	}

	// Make sure not to set the error code if it didn't change
	SystemAlarmsStateEstimationOptions current_error_code;
	SystemAlarmsStateEstimationGet((uint8_t *) &current_error_code);
	if (current_error_code != error_code) {
		SystemAlarmsStateEstimationSet((uint8_t *) &error_code);
	}

	// AlarmSet checks only updates on toggle
	AlarmsSet(SYSTEMALARMS_ALARM_ATTITUDE, (uint8_t) severity);
}

/**
 * @}
 * @}
 */
