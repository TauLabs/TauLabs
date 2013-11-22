/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @file       cfnav_interface.c
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

static int32_t cfnav_interface_init(uintptr_t *id);
static int32_t cfnav_interface_reset(uintptr_t id);
static int32_t cfnav_interface_update(uintptr_t id, float gyros[3], float accels[3], 
		float mag[3], float pos[3], float vel[3], float baro[1],
		float airspeed[1], float dt);
static int32_t cfnav_interface_get_state(uintptr_t id, float pos[3], float vel[3],
		float attitude[4], float gyro_bias[3], float airspeed[1]);

const struct filter_driver cfnav_filter_driver = {
	.class = FILTER_CLASS_S3,

	// this will initialize the SE(3)+ infrastrcture too
	.init = cfnav_interface_init,

	// connects the SE(3)+ queues
	.start = filter_infrastructure_se3_start,
	.reset = cfnav_interface_reset,
	.process = filter_infrastructure_se3_process,
	.sub_driver = {
		.driver_s3 = {
			.update_filter = cfnav_interface_update,
			.get_state = cfnav_interface_get_state,
			.magic = FILTER_S3_MAGIC,
		}
	}
};


enum cfnav_interface_magic {
	CFNAV_INTERFACE_MAGIC = 0x18AEBEC6,
};

enum complimentary_filter_status {
	CFNAV_RESET,
	CFNAV_INITIALIZING,
	CFNAV_NORMAL
};

#define NAV_HISTORY_LEN 4
struct cfnav_interface_data {
	struct    filter_infrastructure_se3_data  *s3_data;

	// Flags to indicate all data set
	bool      mag_updated;
	bool      baro_updated;
	bool      pos_updated;

	// State information
	float     q[4];
	float     accel_alpha;
	float     grot_filtered[3];
	float     accels_filtered[3];
	uint32_t  reset_timeval;
	uint8_t   arming_count;
	bool      accel_filter_enabled;
	float     gyros_bias[3];

	// Variables for the position estimation
	float     position_base[3];
	float     position_error[3];
	float     position_correction[3];
	float     accel_correction[3];
	float     position[3];
	float     velocity[3];

	// Settings for the position state variable
	float     time_constant_xy;
	float     time_constant_z;

	// Position history (stored at gps rate)
	uint8_t   position_history_idx;
	float     position_history[NAV_HISTORY_LEN][2];

	//! The accumulator of gyros during arming
	float      accumulated_gyro[3];
	//! How many gyro samples were acquired
	uint32_t   accumulated_gyro_samples;
	//! Indicate if currently acquiring gyro samples
	bool       accumulating_gyro;

	enum       complimentary_filter_status   initialization;
	enum       cfnav_interface_magic         magic;
};

/**
 * Allocate a new complimentary filter instance
 * @return pointer to device or null if failure
 */
static struct cfnav_interface_data *cfnav_interface_alloc()
{
	struct cfnav_interface_data *cf;

	cf = pvPortMalloc(sizeof(*cf));
	if (cf == NULL)
		return NULL;

	cf->q[0]         = 1;
	cf->q[1]         = 0;
	cf->q[2]         = 0;
	cf->q[3]         = 0;

	cf->time_constant_xy = 1;
	cf->time_constant_z  = 2;

	cf->initialization  = false;
	cf->magic        = CFNAV_INTERFACE_MAGIC;

	return cf;
}

/**
 * Validate a CF filter handle
 * @return true if a valid interface
 */
static bool cfnav_interface_validate(struct cfnav_interface_data *dev)
{
	if (dev == NULL)
		return false;
	if (dev->magic != CFNAV_INTERFACE_MAGIC)
		return false;
	return true;
}


/**
 * Initialize the complimentary filter and the SE(3)+ infrastructure
 * @param[out]  id   the handle for this filter instance
 * @return 0 if successful, -1 if not
 */
static int32_t cfnav_interface_init(uintptr_t *id)
{
	// Allocate the data structure
	struct cfnav_interface_data *cfnav_interface_data = cfnav_interface_alloc();
	if (cfnav_interface_data == NULL)
		return -1;

	// Initialize the infrastructure
	if (filter_infrastructure_se3_init(&cfnav_interface_data->s3_data) != 0)
		return -2;

	// Reset to known starting condition	
	cfnav_interface_reset((uintptr_t) cfnav_interface_data);

	// Return the handle
	(*id) = (uintptr_t) cfnav_interface_data;

	return 0;
}


/********* formatting sensor data to the core math code goes below here *********/

//! Compute the mean gyro accumulated and assign the bias
static void accumulate_gyro_compute(struct cfnav_interface_data *cf);

//! Zero the gyro accumulators
static void accumulate_gyro_zero(struct cfnav_interface_data *cf);

//! Store a gyro sample
static void accumulate_gyro(struct cfnav_interface_data *cf, float *gyros);

//! Apply LPF to sensor data
static void apply_accel_filter(struct cfnav_interface_data *cf, const float * raw, float * filtered);

//! Update the position feedback from the GPS
static void update_pos(struct cfnav_interface_data *cf, float *pos, float dt);

//! Update the baro feedback
static void update_baro(struct cfnav_interface_data *, float baro, float dt);

//! Predict the position in the future
static void predict_pos(struct cfnav_interface_data *cf, float *accel, float dt);

/**
 * Reset the filter state to default
 * @param[in]  id        the filter handle to reset
 */
static int32_t cfnav_interface_reset(uintptr_t id)
{
	struct cfnav_interface_data *cf = (struct cfnav_interface_data *) id;
	if (!cfnav_interface_validate(cf))
		return -1;

	cf->q[0]               = 1;
	cf->q[1]               = 0;
	cf->q[2]               = 0;
	cf->q[3]               = 0;
	cf->accel_alpha        = 0;
	cf->reset_timeval      = 0;
	cf->arming_count       = 0;

	for (uint8_t i = 0; i < 3; i++) {
		cf->grot_filtered[i] = 0;
		cf->accels_filtered[i] = 0;
		cf->gyros_bias[i] = 0;

		cf->position_base[i] = 0;
		cf->position_error[i] = 0;
		cf->position_correction[i] = 0;
		cf->accel_correction[i] = 0;
		cf->position[i] = 0;
		cf->velocity[i] = 0;
	}

	cf->accel_filter_enabled = false;
	cf->initialization     = CFNAV_RESET;
	cf->reset_timeval      = PIOS_DELAY_GetRaw();

	return 0;
}

/**
 * get_sensors Update the filter one time step
 * @param[in] id         the running filter handle
 * @param[in] gyros      new gyro data [rad/s] or NULL
 * @param[in] accels     new accel data [m/s^2] or NULL
 * @param[in] mag        new mag data [mGau] or NULL
 * @param[in] pos        new position measurement in NED [m] or NULL
 * @param[in] vel        new velocity meansurement in NED [m/s] or NULL
 * @param[in] baro       new baro data [m] or NULL
 * @param[in] airspeed   estimate of the airspeed
 * @param[in] dt         time step [s]
 * @returns 0 if sufficient data to run update
 */
static int32_t cfnav_interface_update(uintptr_t id, float gyros[3], float accels[3], 
		float mag[3], float pos[3], float vel[3], float baro[1],
		float airspeed[1], float dt)
{
	struct cfnav_interface_data *cf = (struct cfnav_interface_data *) id;
	if (!cfnav_interface_validate(cf))
		return -1;

	// This architecture currently always provides both but sanity check this
	if (accels == NULL || gyros == NULL)
		return -1;

	AttitudeSettingsData attitudeSettings;
	AttitudeSettingsGet(&attitudeSettings);

	cf->time_constant_xy = attitudeSettings.HorizPositionTau;
	cf->time_constant_z  = attitudeSettings.VertPositionTau;

	uint32_t ms_since_reset = PIOS_DELAY_DiffuS(cf->reset_timeval) / 1000;

	switch(cf->initialization) {
	case CFNAV_RESET:
	{
		static float last_mag[3];

		// Store the most recent mag reset for initializing heading
		if (mag != NULL) {
			cf->mag_updated = true;
			last_mag[0] = mag[0];
			last_mag[1] = mag[1];
			last_mag[2] = mag[2];
		}

		if (baro != NULL) {
			cf->baro_updated = true;
			// TODO: fill baro history with first sample
		}

		if (pos != NULL) {
			cf->pos_updated = true;
			for (uint8_t i = 0; i < NAV_HISTORY_LEN; i++) {
				cf->position_history[i][0] = pos[0];
				cf->position_history[i][1] = pos[1];
				cf->position[0] = pos[0];
				cf->position[1] = pos[1];
			}
		}

		if (cf->pos_updated && cf->baro_updated && cf->mag_updated &&
			ms_since_reset > 1000) {
			float RPY[3];
			float theta = atan2f(accels[0], -accels[2]);
			RPY[1] = theta * RAD2DEG;
			RPY[0] = atan2f(-accels[1], -accels[2] / cosf(theta)) * RAD2DEG;
			RPY[2] = atan2f(-last_mag[1], last_mag[0]) * RAD2DEG;
			RPY2Quaternion(RPY, cf->q);

			accumulate_gyro_zero(cf);

			// Next state
			cf->initialization = CFNAV_INITIALIZING;
		}

		// Do not attempt to process data while in the reset state
		return 0;
	}
		break;
	case CFNAV_INITIALIZING:
		// For first 7 seconds use accels to get gyro bias
		attitudeSettings.AccelKp = 0.1f + 0.1f * (xTaskGetTickCount() < 4000);
		attitudeSettings.AccelKi = 0.1f;
		attitudeSettings.YawBiasRate = 0.1f;
		attitudeSettings.MagKp = 0.1f;

		// Use real attitude settings and restore state
		if (ms_since_reset > 7000) {
			AttitudeSettingsGet(&attitudeSettings);
			if(cf->accel_alpha > 0.0f)
				cf->accel_filter_enabled = true;

			cf->initialization = CFNAV_NORMAL;
		}
		break;
	case CFNAV_NORMAL:
		break;
	}

	// TODO: restore zeroing during arming
	if (0) accumulate_gyro_compute(cf);

	if (gyros)
		accumulate_gyro(cf, gyros);

	float grot[3];
	float accel_err[3];
	float *grot_filtered = cf->grot_filtered;
	float *accels_filtered = cf->accels_filtered;

	// Apply smoothing to accel values, to reduce vibration noise before main calculations.
	apply_accel_filter(cf, accels, accels_filtered);

	// Rotate gravity to body frame and cross with accels
	grot[0] = -(2 * (cf->q[1] * cf->q[3] - cf->q[0] * cf->q[2]));
	grot[1] = -(2 * (cf->q[2] * cf->q[3] + cf->q[0] * cf->q[1]));
	grot[2] = -(cf->q[0]*cf->q[0] - cf->q[1]*cf->q[1] - cf->q[2]*cf->q[2] + cf->q[3]*cf->q[3]);
	CrossProduct(accels, grot, accel_err);

	// Apply same filtering to the rotated attitude to match delays
	apply_accel_filter(cf, grot, grot_filtered);

	// Compute the error between the predicted direction of gravity and smoothed acceleration
	CrossProduct(accels_filtered, grot_filtered, accel_err);

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
	if ( mag != NULL ) {
		// Rotate gravity to body frame and cross with accels
		float brot[3];
		float Rbe[3][3];
		
		Quaternion2R(cf->q, Rbe);

		HomeLocationData homeLocation;
		HomeLocationGet(&homeLocation);

		// Only use the mag when home location is available
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
				CrossProduct(mag, brot, mag_err);

			if (mag_err[2] != mag_err[2])
				mag_err[2] = 0;
		} else
			mag_err[2] = 0;
	} else {
		mag_err[2] = 0;
	}

	// Accumulate integral of error.  Scale here so that units are (deg/s) but Ki has units of s
	cf->gyros_bias[0] -= accel_err[0] * attitudeSettings.AccelKi * DEG2RAD;
	cf->gyros_bias[1] -= accel_err[1] * attitudeSettings.AccelKi * DEG2RAD;
	cf->gyros_bias[2] -= mag_err[2] * attitudeSettings.MagKi * DEG2RAD;

	// Correct rates based on error, integral component dealt with in updateSensors
	gyros[0] += accel_err[0] * attitudeSettings.AccelKp * DEG2RAD / dt;
	gyros[1] += accel_err[1] * attitudeSettings.AccelKp * DEG2RAD / dt;
	gyros[2] += mag_err[2] * attitudeSettings.MagKp * DEG2RAD / dt;

	// Work out time derivative from INSAlgo writeup
	// Also accounts for the fact that gyros are in deg/s
	float qdot[4];
	qdot[0] = (-cf->q[1] * gyros[0] - cf->q[2] * gyros[1] - cf->q[3] * gyros[2]) * dt / 2;
	qdot[1] = (cf->q[0] * gyros[0] - cf->q[3] * gyros[1] + cf->q[2] * gyros[2]) * dt / 2;
	qdot[2] = (cf->q[3] * gyros[0] + cf->q[0] * gyros[1] - cf->q[1] * gyros[2]) * dt / 2;
	qdot[3] = (-cf->q[2] * gyros[0] + cf->q[1] * gyros[1] + cf->q[0] * gyros[2]) * dt / 2;

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

	// Renormalize
	float qmag;
	qmag = sqrtf(cf->q[0]*cf->q[0] + cf->q[1]*cf->q[1] + cf->q[2]*cf->q[2] + cf->q[3]*cf->q[3]);
	cf->q[0] = cf->q[0] / qmag;
	cf->q[1] = cf->q[1] / qmag;
	cf->q[2] = cf->q[2] / qmag;
	cf->q[3] = cf->q[3] / qmag;

	// If quaternion has become inappropriately short or is nan reinit.
	// THIS SHOULD NEVER ACTUALLY HAPPEN
	if((fabsf(qmag) < 1.0e-3f) || (qmag != qmag)) {
		cf->q[0] = 1;
		cf->q[1] = 0;
		cf->q[2] = 0;
		cf->q[3] = 0;
	}

	/*********************************************/
	/* 2. check for gps updates and process them */
	/*********************************************/
	if (pos) {
		update_pos(cf, pos, dt);

		// Store the current state in the history queue
		cf->position_history[cf->position_history_idx][0] = cf->position_base[0];
		cf->position_history[cf->position_history_idx][1] = cf->position_base[1];
		cf->position_history_idx = (cf->position_history_idx + 1) % NAV_HISTORY_LEN;
	}

	/**********************************************/
	/* 3. check for baro updates and process them */
	/**********************************************/
	if (baro) {
		update_baro(cf, *baro, dt);
	}

	/**********************************************/
	/* 4. update position estimate                */
	/**********************************************/
	float accel_ned[3], Rbe[3][3];

	// rotate the accels into the NED frame and remove
	// the influence of gravity
	Quaternion2R(cf->q, Rbe);
	rot_mult(Rbe, accels, accel_ned, true);
	accel_ned[2] += GRAVITY;

	// Predict the state forwmare after applying the correction
	predict_pos(cf, accel_ned, dt);

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
static int32_t cfnav_interface_get_state(uintptr_t id, float pos[3], float vel[3],
		float attitude[4], float gyro_bias[3], float airspeed[1])
{
	struct cfnav_interface_data *cf = (struct cfnav_interface_data *) id;
	if (!cfnav_interface_validate(cf))
		return -1;

	if (attitude) {
		attitude[0] = cf->q[0];
		attitude[1] = cf->q[1];
		attitude[2] = cf->q[2];
		attitude[3] = cf->q[3];
	}

	if (pos) {
		pos[0] = cf->position[0];
		pos[1] = cf->position[1];
		pos[2] = cf->position[2];
	}

	if (vel) {
		vel[0] = cf->velocity[0];
		vel[1] = cf->velocity[1];
		vel[2] = cf->velocity[2];;
	}

	if (gyro_bias) {
		gyro_bias[0] = cf->gyros_bias[0];
		gyro_bias[1] = cf->gyros_bias[1];
		gyro_bias[2] = cf->gyros_bias[2];
	}

	if (airspeed)
		airspeed[0] = 0;

	return 0;
}

/**
 * If accumulating data and enough samples acquired then recompute
 * the gyro bias based on the mean accumulated
 */
static void accumulate_gyro_compute(struct cfnav_interface_data *cf)
{
	if (cf->accumulating_gyro && 
		cf->accumulated_gyro_samples > 100) {

		// Accumulate integral of error.  Scale here so that units are (deg/s) but Ki has units of s
		cf->gyros_bias[0] = cf->accumulated_gyro[0] / cf->accumulated_gyro_samples;
		cf->gyros_bias[1] = cf->accumulated_gyro[1] / cf->accumulated_gyro_samples;
		cf->gyros_bias[2] = cf->accumulated_gyro[2] / cf->accumulated_gyro_samples;

		accumulate_gyro_zero(cf);

		cf->accumulating_gyro = false;
	}
}

/**
 * Zero the accumulation of gyro data
 */
static void accumulate_gyro_zero(struct cfnav_interface_data *cf)
{
	cf->accumulated_gyro_samples = 0;
	cf->accumulated_gyro[0] = 0;
	cf->accumulated_gyro[1] = 0;
	cf->accumulated_gyro[2] = 0;
}

/**
 * Accumulate a set of gyro samples for computing the
 * bias
 * @param [in] gyrosData The samples of data to accumulate
 */
static void accumulate_gyro(struct cfnav_interface_data *cf, float *gyros)
{
	if (!cf->accumulating_gyro)
		return;

	cf->accumulated_gyro_samples++;

	// bias_correct_gyro
	uint8_t bias_correct_gyro;
	AttitudeSettingsBiasCorrectGyroGet(&bias_correct_gyro);
	if (bias_correct_gyro == ATTITUDESETTINGS_BIASCORRECTGYRO_TRUE) {
		// Apply bias correction to the gyros from the state estimator
		cf->accumulated_gyro[0] += gyros[0] + cf->gyros_bias[0];
		cf->accumulated_gyro[1] += gyros[1] + cf->gyros_bias[1];
		cf->accumulated_gyro[2] += gyros[2] + cf->gyros_bias[2];
	} else {
		cf->accumulated_gyro[0] += gyros[0];
		cf->accumulated_gyro[1] += gyros[1];
		cf->accumulated_gyro[2] += gyros[2];
	}
}

/**
 * Apply LPF to the accel and gyros before using the data
 * @param[in] raw the raw sensor data
 * @param[out] filtered the low pass filtered data
 */
static void apply_accel_filter(struct cfnav_interface_data *cf, const float * raw, float * filtered)
{
	const float alpha = cf->accel_alpha;
	if (cf->accel_filter_enabled) {
		filtered[0] = filtered[0] * alpha + raw[0] * (1 - alpha);
		filtered[1] = filtered[1] * alpha + raw[1] * (1 - alpha);
		filtered[2] = filtered[2] * alpha + raw[2] * (1 - alpha);
	} else {
		filtered[0] = raw[0];
		filtered[1] = raw[1];
		filtered[2] = raw[2];
	}
}


//! Predict the position in the future
static void predict_pos(struct cfnav_interface_data *cf, float *accel, float dt)
{
	float k1_xy = 3 / cf->time_constant_xy;
	float k2_xy = 3 / powf(cf->time_constant_xy, 2);
	float k3_xy = 3 / powf(cf->time_constant_xy, 3);

	float k1_z = 3 / cf->time_constant_z;
	float k2_z = 3 / powf(cf->time_constant_z, 2);
	float k3_z = 3 / powf(cf->time_constant_z, 3);

	float tmp;

	tmp = k3_xy * dt;
	cf->accel_correction[0] += cf->position_error[0] * tmp;
	cf->accel_correction[1] += cf->position_error[1] * tmp;
	cf->accel_correction[2] += cf->position_error[2] * k3_z * dt;

	tmp = k2_xy * dt;
	cf->velocity[0] += cf->position_error[0] * tmp;
	cf->velocity[1] += cf->position_error[1] * tmp;
	cf->velocity[2] += cf->position_error[2] * k2_z * dt;

	tmp = k1_xy * dt;
	cf->position_correction[0] += cf->position_error[0] * tmp;
	cf->position_correction[1] += cf->position_error[1] * tmp;
	cf->position_correction[2] += cf->position_error[2] * k1_z * dt;

	for (uint8_t i = 0; i < 3; i++) {
		float velocity_increase;
		velocity_increase = (accel[i] + cf->accel_correction[i]) * dt;
		cf->position_base[i] += (cf->velocity[i] + velocity_increase * 0.5f) * dt;
		cf->position[i] = cf->position_base[i] + cf->position_correction[i];
		cf->velocity[i] += velocity_increase;
	}

	// TODO: queue the old position estimates
}

//! Update the position feedback from the GPS
static void update_pos(struct cfnav_interface_data *cf, float *pos, float dt)
{
	// Compute the error between the latest GPS sample and a previous
	// state estimate.
	uint8_t sample_idx = cf->position_history_idx;
	cf->position_error[0] = pos[0] - (cf->position_history[sample_idx][0] + cf->position_correction[0]);
	cf->position_error[1] = pos[1] - (cf->position_history[sample_idx][1] + cf->position_correction[1]);
}

//! Update the baro feedback
static void update_baro(struct cfnav_interface_data *cf, float baro, float dt)
{
	float down = -baro;

	// TODO: get from the queue of previous position updates (150 ms latency)
	float hist_position_base_d = cf->position_base[2];

	cf->position_error[2] = down - (hist_position_base_d + cf->position_correction[2]);
}

/**
 * @}
 */

