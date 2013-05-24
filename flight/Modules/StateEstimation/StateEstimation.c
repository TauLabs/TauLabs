/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @file       stateestimation.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      State estimation module which calls to specific drivers
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

#include "pios.h"
#include "openpilot.h"
#include "physical_constants.h"
#include "filter_interface.h"

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
#include "CoordinateConversions.h"

// Private constants
#define STACK_SIZE_BYTES 2448
#define TASK_PRIORITY (tskIDLE_PRIORITY+3)
#define FAILSAFE_TIMEOUT_MS 10

// Private variables
static xTaskHandle attitudeTaskHandle;

// Private functions
static void StateEstimationTask(void *parameters);
static void settingsUpdatedCb(UAVObjEvent * objEv);

// Mapping from UAVO setting to filters
const static struct filter_driver filters[] = {
	[STATEESTIMATION_ATTITUDEFILTER_COMPLEMENTARY] = complementary_filter_driver,
	[STATEESTIMATION_ATTITUDEFILTER_INSINDOOR] = insindoor_filter_driver,
	[STATEESTIMATION_ATTITUDEFILTER_INSOUTDOOR] = insoutdoor_filter_driver,
};

// Wrapper for the types of filter classes
struct filter_class_infrastrcture {
	int32_t (*prepare)(struct filter_driver *);
	int32_t (*process)(struct filter_driver *, uintptr_t id, float dt);
}

// Set of functions for calling filter various filter classes
const static struct filter_class_infrastructure {
	[FILTER_CLASS_S3] = {
		.prepare = prepare_s3_infrastructure,
		.process = process_filter_s3
	},
	[FILTER_CLASS_GENERIC] = {
		.process = NULL,
		.process = process_filter_generic
	}
} infrastructures;

/**
 * Initialise the module.  Called before the start function
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t StateEstimationInitialize(void)
{
	// NOTE: we will have some practical issues not initializing new
	// objects (filter specific ones here) with telemetry.  They will
	// need a way to register themselves.

	AttitudeActualInitialize();
	AttitudeSettingsInitialize();
	SensorSettingsInitialize();
	NEDPositionInitialize();
	PositionActualInitialize();
	StateEstimationInitialize();
	VelocityActualInitialize();

	SensorSettingsConnectCallback(&settingsUpdatedCb);

	return 0;
}

/**
 * Start the task.  Expects all objects to be initialized by this point.
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t StateEstimationStart(void)
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

	// Start main task
	xTaskCreate(StateEstimationTask, (signed char *)"StateEstimation", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &attitudeTaskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_ATTITUDE, attitudeTaskHandle);
	PIOS_WDG_RegisterFlag(PIOS_WDG_ATTITUDE);

	return 0;
}

MODULE_INITCALL(StateEstimationInitialize, StateEstimationStart)

/**
 * Module thread, should not return.
 */
static void StateEstimationTask(void *parameters)
{
	AlarmsClear(SYSTEMALARMS_ALARM_ATTITUDE);

	// Force settings update to make sure rotation loaded
	settingsUpdatedCb(NULL);

	// Wait for all the sensors be to read
	vTaskDelay(100);

	struct filter_driver *current_filter = NULL;
	uintptr_t running_filter_id;

	// Get the driver for the selected filter
	uint8_t selected_filter;
	StateEstimationAttitudeFilterGet(&selected_filter);
	if (current_filter < NELEMENTS(filters))
		current_filter = filters[selected_filter];
	else
		goto FAIL;

	// Check this filter is safe to run
	if (!filter_validate(current_filter))
		goto FAIL;
	if (requested_filter->init(&running_filter_id) != 0)
		goto FAIL;
	if (requested_filter->reset(running_filter_id) != 0)
		goto FAIL;

	// Set up the filter class specific infrastructure
	struct filter_class_infrastrcture *infrastructure = &infrastructures[requested_filter->class];
	if (infrastructure->prepare)
		infrastructure->prepare(requested_filter);


	// Main task loop
	while (1) {
		infrastructure->process(requested_filter, running_filter_id);
		PIOS_WDG_UpdateFlag(PIOS_WDG_ATTITUDE);
	}

FAIL:
	AlarmsSet(SYSTEMALARMS_ALARM_ATTITUDE, SYSTEMALARMS_ALARM_CRITICAL);
	while(1) {
		vTaskDelay(100);
		PIOS_WDG_UpdateFlag(PIOS_WDG_ATTITUDE);
	}
}

/**
 * filter_validate Validate a filter has is safe to run and 
 * has a correct and matching driver
 * @param[in] filter    the filter to check
 * @param[in] id        the handle for the filter to process
 * @return true if safe, false if not
 */
static bool filter_validate(struct filter_driver *filter, uintptr_t id)
{
	if (filter == NULL)
		return false;

	switch (filter->class) {
	case FILTER_CLASS_S3:
		return filter->driver_s3.magic == FILTER_S3_MAGIC;
	case FILTER_CLASS_GENERIC
		return filter->driver_generic.magic = FILTER_GENERIC_MAGIC;
	default:
		return false;
	}
}

/************ code related to running generic filters *********/

/**
 * process_filter_generic Compute an update of a generic filter
 * @param[in] driver The generic filter driver
 * @param[in] dT the update time in seconds
 * @return 0 if succesfully updated or error code
 */
static int32_t process_filter_generic(struct filter_driver_generic *driver, uintptr_t id, float dT)
{
	driver->get_sensors(id);
	driver->update_filter(id, dT);
	driver->get_state(id);
}

/************ code related to running s3 filters **************/

static xQueueHandle gyroQueue;
static xQueueHandle accelQueue;
static xQueueHandle magQueue;
static xQueueHandle baroQueue;
static xQueueHandle gpsQueue;
static xQueueHandle gpsVelQueue;

//! Connect the queues used for S3 filters
static int32_t prepare_s3_infrastructure()
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
static int32_t process_filter_s3(struct filter_driver_s3 *driver, uintptr_t id)
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


/*********** miscellaneous code ****************/

/**
 * @brief Convert the GPS LLA position into NED coordinates
 * @note this method uses a taylor expansion around the home coordinates
 * to convert to NED which allows it to be done with all floating
 * calculations
 * @param[in] Current GPS coordinates
 * @param[out] NED frame coordinates
 * @returns 0 for success, -1 for failure
 */
static float T[3];
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
	if (ev == NULL || ev->obj == StateEstimationHandle())
		StateEstimationGet(&stateEstimation);
}

/**
 * @}
 * @}
 */