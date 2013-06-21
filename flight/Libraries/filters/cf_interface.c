/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @file       cf_interface.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Interface from the SE(3)+ infrastructure to the complimentary filter
 *
 * @see        The GNU Public License (GPL) Version 3
 *
 *****************************************************************************/
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

#include "filter_interface.h"
#include "filter_infrastructure_se3.h"

#include "openpilot.h"
#include "attitudesettings.h"
#include "flightstatus.h"
#include "homelocation.h"

#include "physical_constants.h"
#include "coordinate_conversions.h"

#include "stdint.h"
#include "stdbool.h"

static int32_t cf_interface_init(uintptr_t *id);
static int32_t cf_interface_reset(uintptr_t id);
static int32_t cf_interface_update(uintptr_t id, float gyros[3], float accels[3], 
		float mag[3], float pos[3], float vel[3], float baro[1],
		float airspeed[1], float dt);
static int32_t cf_interface_get_state(uintptr_t id, float pos[3], float vel[3],
		float attitude[4], float gyro_bias[3], float airspeed[1]);

struct filter_driver cf_filter_driver = {
	.class = FILTER_CLASS_S3,

	// this will initialize the SE(3)+ infrastrcture too
	.init = cf_interface_init,

	// connects the SE(3)+ queues
	.start = filter_infrastructure_se3_start,
	.reset = cf_interface_reset,
	.process = filter_infrastructure_se3_process,
	.sub_driver = {
		.driver_s3 = {
			.update_filter = cf_interface_update,
			.get_state = cf_interface_get_state,
			.magic = FILTER_S3_MAGIC,
		}
	}
};


enum cf_interface_magic {
	CF_INTERFACE_MAGIC = 0x18AEBE6C,
};

enum complimentary_filter_status {
	CF_POWERON,
	CF_INITIALIZING,
	CF_ARMING,
	CF_NORMAL
};

struct cf_interface_data {
	struct filter_infrastructure_se3_data *s3_data;
	float     q[4];
	float     accel_alpha;
	float     grot_filtered[3];
	float     accels_filtered[3];
	uint32_t  reset_timeval;
	uint8_t   arming_count;
	bool      accel_filter_enabled;
	enum complimentary_filter_status   initialization;
	enum      cf_interface_magic magic;
};

/**
 * Allocate a new complimentary filter instance
 * @return pointer to device or null if failure
 */
static struct cf_interface_data *cf_interface_alloc()
{
	struct cf_interface_data *cf;

	cf = pvPortMalloc(sizeof(*cf));
	if (cf == NULL)
		return NULL;

	cf->q[0]         = 1;
	cf->q[1]         = 0;
	cf->q[2]         = 0;
	cf->q[3]         = 0;
	cf->initialization  = false;
	cf->magic        = CF_INTERFACE_MAGIC;

	return cf;
}

/**
 * Validate a CF filter handle
 * @return true if a valid interface
 */
static bool cf_interface_validate(struct cf_interface_data *dev)
{
	if (dev == NULL)
		return false;
	if (dev->magic != CF_INTERFACE_MAGIC)
		return false;
	return true;
}


/**
 * Initialize the complimentary filter and the SE(3)+ infrastructure
 * @param[out]  id   the handle for this filter instance
 * @return 0 if successful, -1 if not
 */
static int32_t cf_interface_init(uintptr_t *id)
{
	// Allocate the data structure
	struct cf_interface_data *cf_interface_data = cf_interface_alloc();
	if (cf_interface_data == NULL)
		return -1;

	// Initialize the infrastructure
	if (filter_infrastructure_se3_init(&cf_interface_data->s3_data) != 0)
		return -2;
	
	// Return the handle
	(*id) = (uintptr_t) cf_interface_data;

	return 0;
}


/********* formatting sensor data to the core math code goes below here *********/

/**
 * Reset the filter state to default
 * @param[in]  id        the filter handle to reset
 */
static int32_t cf_interface_reset(uintptr_t id)
{
	struct cf_interface_data *cf = (struct cf_interface_data *) id;
	if (!cf_interface_validate(cf))
		return -1;

	cf->q[0]         = 1;
	cf->q[1]         = 0;
	cf->q[2]         = 0;
	cf->q[3]         = 0;
	cf->initialization  = CF_POWERON;

	return 0;
}

/**
 * get_sensors Update the filter one time step
 * @param[in] id         the running filter handle
 * @param[in] gyros      new gyro data [deg/s] or NULL
 * @param[in] accels     new accel data [m/s^2] or NULL
 * @param[in] mag        new mag data [mGau] or NULL
 * @param[in] pos        new position measurement in NED [m] or NULL
 * @param[in] vel        new velocity meansurement in NED [m/s] or NULL
 * @param[in] baro       new baro data [m] or NULL
 * @param[in] airspeed   estimate of the airspeed
 * @param[in] dt         time step [s]
 * @returns 0 if sufficient data to run update
 */
static int32_t cf_interface_update(uintptr_t id, float gyros[3], float accels[3], 
		float mag[3], float pos[3], float vel[3], float baro[1],
		float airspeed[1], float dt)
{
	struct cf_interface_data *cf = (struct cf_interface_data *) id;
	if (!cf_interface_validate(cf))
		return -1;

	AttitudeSettingsData attitudeSettings;
	AttitudeSettingsGet(&attitudeSettings);

	// When this algorithm is first run force it to a known condition
	if(!cf->initialization) {
		if (mag == NULL)
			return -1;

		float RPY[3];
		float theta = atan2f(accels[0], -accels[2]);
		RPY[1] = theta * RAD2DEG;
		RPY[0] = atan2f(-accels[1], -accels[2] / cosf(theta)) * RAD2DEG;
		RPY[2] = atan2f(-mag[1], mag[0]) * RAD2DEG;
		RPY2Quaternion(RPY, cf->q);

		// TODO: move to dev
		cf->initialization = CF_POWERON;
		cf->reset_timeval = PIOS_DELAY_GetRaw();
		cf->arming_count = 0;

		return 0;
	}

	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);

	uint32_t ms_since_reset = PIOS_DELAY_DiffuS(cf->reset_timeval) / 1000;
	if (cf->initialization == CF_POWERON) {
		// Wait one second before starting to initialize
		cf->initialization = 
		    (ms_since_reset  > 1000) ?
			CF_INITIALIZING : 
			CF_POWERON;
	} else if(cf->initialization == CF_INITIALIZING &&
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
		if (cf->arming_count < 20)
			attitudeSettings.AccelKp = 1.0f;
		else if (attitudeSettings.AccelKp > 0.1f)
			attitudeSettings.AccelKp -= 0.01f;
		cf->arming_count++;

		// Set the other parameters to drive faster convergence
		attitudeSettings.AccelKi = 0.1f;
		attitudeSettings.YawBiasRate = 0.1f;
		attitudeSettings.MagKp = 0.1f;

		// Don't apply LPF to the accels during arming
		cf->accel_filter_enabled = false;

		// Indicate arming so that after arming it reloads
		// the normal settings
		if (cf->initialization != CF_ARMING) {
			accumulate_gyro_zero();
			cf->initialization = CF_ARMING;
			cf->accumulating_gyro = true;
		}

	} else if (cf->initialization == CF_ARMING ||
	           cf->initialization == CF_INITIALIZING) {

		AttitudeSettingsGet(&attitudeSettings);
		if(cf->accel_alpha > 0.0f)
			cf->accel_filter_enabled = true;

		// If arming that means we were accumulating gyro
		// samples.  Compute new bias.
		if (cf->initialization == CF_ARMING) {
			accumulate_gyro_compute();
			cf->accumulating_gyro = false;
			cf->arming_count = 0;
		}

		// Indicate normal mode to prevent rerunning this code
		cf->initialization = CF_NORMAL;
	}

	accumulate_gyro(gyros);


	float grot[3];
	float accel_err[3];
	float *grot_filtered = cf->grot_filtered;
	float *accels_filtered = cf->accels_filtered;

	// Apply smoothing to accel values, to reduce vibration noise before main calculations.
	apply_accel_filter(accels, accels_filtered);

	// Rotate gravity to body frame and cross with accels
	grot[0] = -(2 * (cf->q[1] * cf->q[3] - cf->q[0] * cf->q[2]));
	grot[1] = -(2 * (cf->q[2] * cf->q[3] + cf->q[0] * cf->q[1]));
	grot[2] = -(cf->q[0]*cf->q[0] - cf->q[1]*cf->q[1] - cf->q[2]*cf->q[2] + cf->q[3]*cf->q[3]);
	CrossProduct((const float *) accels, (const float *) grot, accel_err);

	// Apply same filtering to the rotated attitude to match delays
	apply_accel_filter(grot, grot_filtered);

	// Compute the error between the predicted direction of gravity and smoothed acceleration
	CrossProduct((const float *) accels_filtered, (const float *) grot_filtered, accel_err);

	float grot_mag;
	if (cf->accel_filter_enabled)
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
	if ( mag != NULL )
	{
		// Rotate gravity to body frame and cross with accels
		float brot[3];
		float Rbe[3][3];
		
		Quaternion2R(cf->q, Rbe);

		// If the mag is producing bad data don't use it (normally bad calibration)
		if  (homeLocation.Set == HOMELOCATION_SET_TRUE) {
			rot_mult(Rbe, homeLocation.Be, brot, false);

			float mag_len = sqrtf(mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2]);
			mag[0] /= mag_len;
			mag[1] /= mag_len;
			mag[2] /= mag_len;

			float bmag = sqrtf(brot[0] * brot[0] + brot[1] * brot[1] + brot[2] * brot[2]);
			brot[0] /= bmag;
			brot[1] /= bmag;
			brot[2] /= bmag;

			// Only compute if neither vector is null
			if (bmag < 1 || mag_len < 1)
				mag_err[0] = mag_err[1] = mag_err[2] = 0;
			else
				CrossProduct((const float *) mag, (const float *) brot, mag_err);

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
	gyrosData.x += accel_err[0] * attitudeSettings.AccelKp / dt;
	gyrosData.y += accel_err[1] * attitudeSettings.AccelKp / dt;
	gyrosData.z += mag_err[2] * attitudeSettings.MagKp / dt;

	// Work out time derivative from INSAlgo writeup
	// Also accounts for the fact that gyros are in deg/s
	float qdot[4];
	qdot[0] = (-cf->q[1] * gyrosData.x - cf->q[2] * gyrosData.y - cf->q[3] * gyrosData.z) * dt * DEG2RAD / 2;
	qdot[1] = (cf->q[0] * gyrosData.x - cf->q[3] * gyrosData.y + cf->q[2] * gyrosData.z) * dt * DEG2RAD / 2;
	qdot[2] = (cf->q[3] * gyrosData.x + cf->q[0] * gyrosData.y - cf->q[1] * gyrosData.z) * dt * DEG2RAD / 2;
	qdot[3] = (-cf->q[2] * gyrosData.x + cf->q[1] * gyrosData.y + cf->q[0] * gyrosData.z) * dt * DEG2RAD / 2;

	// Take a time step
	cf->q[0] = cf->q[0] + qdot[0];
	cf->q[1] = cf->q[1] + qdot[1];
	cf->q[2] = cf->q[2] + qdot[2];
	cf->q[3] = cf->q[3] + qdot[3];

	if(cf->q[0] < 0) {
		cf->q[0] = -cf->q[0];
		cf->q[1] = -cf->q[1];
		cf->q[2] = -cf->q[2];
		cf->q[3] = -cf->q[3];
	}

	// Renomalize
	float qmag;
	qmag = sqrtf(cf->q[0]*cf->q[0] + cf->q[1]*cf->q[1] + cf->q[2]*cf->q[2] + cf->q[3]*cf->q[3]);
	cf->q[0] = cf->q[0] / qmag;
	cf->q[1] = cf->q[1] / qmag;
	cf->q[2] = cf->q[2] / qmag;
	cf->q[3] = cf->q[3] / qmag;

	// If quaternion has become inappropriately short or is nan reinit.
	// THIS SHOULD NEVER ACTUALLY HAPPEN
	if((fabs(qmag) < 1.0e-3f) || (qmag != qmag)) {
		cf->q[0] = 1;
		cf->q[1] = 0;
		cf->q[2] = 0;
		cf->q[3] = 0;
	}

	AlarmsClear(SYSTEMALARMS_ALARM_ATTITUDE);

	return 0;
}

/**
 * get_state Retrieve the state from the S(3) filter
 * any param can be null indicating it is not being fetched
 * @param[in]  id        the running filter handle
 * @param[out] pos       the updated position in NED [m]
 * @param[out] vel       the updated velocity in NED [m/s]
 * @param[out] attitude  the updated attitude quaternion
 * @param[out] gyro_bias the update gyro bias [deg/s]
 * @param[out] airspeed  estimate of the airspeed
 */
static int32_t cf_interface_get_state(uintptr_t id, float pos[3], float vel[3],
		float attitude[4], float gyro_bias[3], float airspeed[1])
{
	return 0;
}

/**
 * @}
 */

