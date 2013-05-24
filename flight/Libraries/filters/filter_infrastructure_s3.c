/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @file       filter_infrastructure.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Infrastructure for managing S3 filters
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

static xQueueHandle gyroQueue;
static xQueueHandle accelQueue;
static xQueueHandle magQueue;
static xQueueHandle baroQueue;
static xQueueHandle gpsQueue;
static xQueueHandle gpsVelQueue;

int32_t filter_infrastructure_s3_init(uintptr_t *id)
{
	AttitudeActualInitialize();
	AttitudeSettingsInitialize();
	SensorSettingsInitialize();
	NEDPositionInitialize();
	PositionActualInitialize();
	VelocityActualInitialize();
}

//! Connect the queues used for S3 filters
int32_t filter_infrastructure_s3_start(uintptr_t id)
{
	if (GyrosHandle())
		GyrosConnectQueue(gyroQueue);
	if (AccelsHandle())
		AccelsConnectQueue(accelQueue);
	if (MagnetometerHandle())
		MagnetometerConnectQueue(magQueue);
	if (BaroAltitudeHandle())
		BaroAltitudeConnectQueue(baroQueue);
	if (GPSPositionHandle())
		GPSPositionConnectQueue(gpsQueue);
	if (GPSVelocityHandle())
		GPSVelocityConnectQueue(gpsVelQueue);
}

/**
 * process_filter_generic Compute an update of an S3 filter
 * @param[in] driver The S3 filter driver
 * @param[in] dT the update time in seconds
 * @return 0 if succesfully updated or error code
 */
int32_t filter_infrastructure_s3_process(struct filter_driver_s3 *driver, uintptr_t id)
{
	// TODO: check error codes

	/* 1. fetch the data from queues and pass to filter                    */
	/* if we want to start running multiple instances of this filter class */
	/* simultaneously, then this step should be done once and then all     */
	/* filters should be processed with the same data                      */

	// Potential measurements
	float *gyros = NULL;
	float *accels = NULL;
	float *mag = NULL;
	float *pos = NULL;
	float *vel = NULL;
	float *baro = NULL;
	float *airspeed = NLL;

	// Check whether the measurements were updated and fetch if so
	UAVObjEvent ev;
	GyrosData gyrosData;
	AccelsData accelsData;
	MagData magData;
	BaroData baroData;
	GPSPosition gpsPosition;
	GPSVelocity gpsVelocity;
	float NED[3];

	if (gyroQueue, &ev, FAILSAFE_TIMEOUT_MS / portTICK_RATE_MS) == pdTRUE) {
		GyrosGet(&gyrosData);
		gyros = &gyrosData.x;
	}

	if (xQueueReceive(accelQueue, &ev, 1 / portTICK_RATE_MS) == pdTRUE) {
		AcceslGet(&accelsData);
		accels = &accelsData.x;
	}

	if (xQueueReceive(magQueue, &ev, 0 / portTICK_RATE_MS) == pdTRUE)) {
		MagsGet(&magData);
		mags = &magData.x;
	}

	if (xQueueReceive(baroQueue, &ev, 0 / portTICK_RATE_MS) == pdTRUE) {
		BaroGet(&baroData);
		baro = &baroData.Altitude;
	}

	if (xQueueReceive(gpsQueue, &ev, 0 / portTICK_RATE_MS) == pdTRUE) {
		GPSPositionGet(&gpsPosition);
		getNED(gpsPosition, NED);
		pos = NED;
	}

	if (xQueueReceive(gpsVelQueue, &ev, 0 / portTICK_RATE_MS) == pdTRUE) {
		GPSVelocityGet(&gpsVelocity);
		vel = &gpsVelocity.North;
	}

	// Store the measurements in the driver
	driver->get_sensors(id, gyros, accels, mag, pos, vel, baro, airspeed);

	/* 2. compute update */
	driver->update_filter(id, dT);

	/* 3. get the state update from the filter */
	float pos_state[3];
	float vel_state[3];
	float q_state[4];
	float gyro_bias_state[3];

	driver->get_state(id, pos_state, vel_state, q_state, gyro_bias_state);

	// Store the data in UAVOs
	PositionActualData positionActual;
	positionActual.North = pos_state[0];
	positionActual.East  = pos_state[1];
	positionActual.Down  = pos_state[2];
	PositionActualSet(&positionActual);

	VelocityActualData velocityActual;
	velocityActual.North = vel_state[0]
	velocityActual.East  = vel_state[1]
	velocityActual.Down  = vel_state[2]
	VelocityActualSet(&velocityActual);

	AttitudeActualData attitudeActual;
	attitudeActual.q1 = q_state[0];
	attitudeActual.q2 = q_state[1];
	attitudeActual.q3 = q_state[2];
	attitudeActual.q4 = q_state[3];
	Quaternion2RPY(&attitude.q1,&attitude.Roll);
	AttitudeActualSet(&attitudeActual);
}


 /**
  * @}
  */