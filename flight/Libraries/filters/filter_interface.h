/**
 ******************************************************************************
 * @file       filter_driver.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Provides a general way of defining state estimate filters
 * @addtogroup Filters
 * @{
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

enum filter_class {
	FILTER_CLASS_S3,      // uses infrastructure to ease S(3) filters
	FILTER_CLASS_GENERIC, // generic filter which does all its own work
};

//! Set of standard error codes for filters to allow mapping to system alarms
enum filter_return_codes {
	FILTER_RETURN_AWAITING_RESET,
	FILTER_RETURN_RUNNING,
	FILTER_RETURN_INSUFFICIENT_DATA,
	FILTER_RETURN_DIVERGED
};

/***** infrastructure for standard S(3) filter using our core sensors *****/

enum filter_s3_magic {
	FILTER_S3_MAGIC = 0x38583fbc,
};

//! Driver for an S3 filter using standard core sensors
struct filter_s3 {
	/**
	 * get_sensors Compute one time step of the filter
	 * @param[in] id        the running filter handle
	 * @param[in] gyros     new gyro data [deg/s] or NULL
	 * @param[in] accels    new accel data [m/s^2] or NULL
	 * @param[in] mag       new mag data [mGau] or NULL
	 * @param[in] pos       new position measurement in NED [m] or NULL
	 * @param[in] vel       new velocity meansurement in NED [m/s] or NULL
	 * @param[in] baro      new baro data [m] or NULL
	 * @param[in] airspeed  new airspeed [m/s] or NULL
	 * @param[in] dt        the time step
	 * @returns 0 if sufficient data to run update
	 */
	int32_t (*update_filter)(uintptr_t id, float gyros[3], float accels[3], 
		float mag[3], float pos[3], float vel[3], float baro[1],
		float airspeed[1], float dt);

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
	int32_t (*get_state)(uintptr_t id, float pos[3], float vel[3],
		float attitude[4], float gyro_bias[3], float airspeed[1]);

	//! To check the filter driver is valid, must be FILTER_S3_MAGIC
	int32_t magic;

};


/***** infrastructure for an entirely generic filter *****/

enum filter_generic_magic {
	FILTER_GENERIC_MAGIC = 0xbed387ab;
};

//! Driver for an S3 filter using standard core sensors
struct filter_generic {
	/**
	 * get_sensors Get sensor data directly from desired UAVOS or queues
	 * @param[in]  id        the running filter handle
	 * @returns 0 if sufficient data to run update
	 */
	int32_t (*get_sensors)(uintptr_t id);

	/**
	 * update_filter Compute one time step of the filter
	 * @param[in]  id        the running filter handle
	 * @param[in]  dt        time step [s]
	 * @return 0 if succesfully update, -1 if not
	 */
	int32_t (*update_filter)(uintptr_t id, float dt);

	/**
	 * get_state Sets the state directly into the UAVOs
	 * @param[in]  id        the running filter handle
	 * @return 0 if succesful, -1 if not
	 */
	int32_t (*get_state)(uintptr_t id);

	//! To check the filter driver is valid, must be FILTER_GENERIC_MAGIC
	int32_t magic;
};

/***** union of all the filter classes *****/

struct filter_driver {
	//! class must accurately match the following type
	enum filter_class class;

	/**
	 * Initialize the filter, allocate memory
	 * @param[out] id        the handle for this initialized filter instance
	 * @param[out] id handle for local data for this filter instance
	 */
	int32_t (*init)(uintptr_t *id);

	/**
	 * Start the filter once FreeRTOS running (all should exist)
	 * @param[in]  id        the filter handle to start
	 * @param[out] id handle for local data for this filter instance
	 */
	int32_t (*start)(uintptr_t id);

	/**
	 * Reset the filter state to default
	 * @param[in]  id        the filter handle to reset
	 */
	int32_t (*reset)(uintptr_t id);

	/**
	 * Process the filter forward a step
	 * @param[in]  id        the filter handle to reset
	 * @param[in]  dt        time step [s]
	 * @return 0 if successful or error code
	 */
	int32_t (*process)(uintptr_t id, float dt);

	//! The specific driver for this filter implementation class
	union driver {
		struct filter_s3       driver_s3;
		struct filter_generic  driver_generic;
	};
};

bool filter_interface_validate(struct filter_driver *filter, uintptr_t id);

/**
 * @}
 */