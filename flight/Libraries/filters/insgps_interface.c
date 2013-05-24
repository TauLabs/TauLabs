/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @file       filter_interface_insgps.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Interface from the SE(3)+ infrastructure to the INSGPS
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
#include "filter_infrastructure_s3.h"

static int32_t insgps_interface_init(uintptr_t id);
static int32_t insgps_interface_reset(uintptr_t id);
static int32_t insgps_interface_get_sensors(uintptr_t id, float gyros[3], float accels[3], 
		float mag[3], float pos[3], float vel[3], float baro[1],
		float airspeed[1]);
static int32_t insgps_interface_update_filter(uintptr_t id, float dt);
static int32_t insgps_interface_get_state(uintptr_t id, float pos[3], float vel[3],
		float attitude[4], float gyro_bias[3], float airspeed[1]);

struct filter_driver {
	.class = FILTER_CLASS_S3,

	// this will initialize the SE(3)+ infrastrcture too
	.init = insgps_interface_init,

	// connects the SE(3)+ queues
	.start = filter_infrastructure_s3_start,
	.reset = insgps_interface_reset,
	.process = filter_infrastructure_s3_process,
	.driver_s3 = {
		.get_sensors = insgps_interface_get_sensors,
		.update_filter = insgps_interface_update_filter,
		.get_state = insgps_interface_get_state,
		.magic = FILTER_S3_MAGIC,
	}
} insgps_filter_driver;

struct insgps_interface_data {
	struct filter_infrastructure_s3_data *s3_data;
	uint32_t magic;
};

static struct insgps_interface_data * insgps_interface_alloc()
{
	// TODO
}

/**
 * Initialize this INSGPS filter and the SE(3)+ infrastructure
 * @param[out]  id   the handle for this filter instance
 * @return 0 if successful, -1 if not
 */
static int32_t insgps_interface_init(uintptr_t *id)
{
	// Allocate the data structure
	struct insgps_interface_data * insgps_interface_data = insgps_interface_alloc();
	if (insgps_interface_data == NULL)
		return -1;

	// Initialize the infrastructure
	if (filter_infrastructure_s3_init(&insgps_interface_data->s3_data) != 0)
		return -2;
	
	// Return the handle
	(*id) = (uintptr_t) insgps_interface_data;

	return 0;
}


/********* formatting sensor data to the core math code goes below here *********/

/**
 * Reset the filter state to default
 * @param[in]  id        the filter handle to reset
 */
static int32_t insgps_interface_reset(uintptr_t id)
{
	
}

/**
 * get_sensors Get the sensor data from the core loop
 * @param[in]  id       the running filter handle
 * @params[in] gyros    new gyro data [deg/s] or NULL
 * @params[in] accels   new accel data [m/s^2] or NULL
 * @params[in] mag      new mag data [mGau] or NULL
 * @params[in] pos      new position measurement in NED [m] or NULL
 * @params[in] vel      new velocity meansurement in NED [m/s] or NULL
 * @params[in] baro     new baro data [m] or NULL
 * @params[in] airspeed new airspeed [m/s] or NULL
 * @returns 0 if sufficient data to run update
 */
static int32_t insgps_interface_get_sensors(uintptr_t id, float gyros[3], float accels[3], 
		float mag[3], float pos[3], float vel[3], float baro[1],
		float airspeed[1])
{

}

/**
 * update_filter Compute one time step of the filter
 * @param[in]  id        the running filter handle
 * @param[in]  dt        time step [s]
 * @return 0 if succesfully update, -1 if not
 */
static int32_t insgps_interface_update_filter(uintptr_t id, float dt)
{

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
static int32_t insgps_interface_get_state(uintptr_t id, float pos[3], float vel[3],
		float attitude[4], float gyro_bias[3], float airspeed[1])
{

}

 /**
  * @}
  */