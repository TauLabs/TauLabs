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
	 * get_sensors Get the sensor data from the core loop
	 * @params[in] gyros    new gyro data [deg/s] or NULL
	 * @params[in] accels   new accel data [m/s^2] or NULL
	 * @params[in] mag      new mag data [mGau] or NULL
	 * @params[in] pos      new position measurement in NED [m] or NULL
	 * @params[in] vel      new velocity meansurement in NED [m/s] or NULL
	 * @params[in] baro     new baro data [m] or NULL
	 * @params[in] airspeed new airspeed [m/s] or NULL
	 * @returns 0 if sufficient data to run update
	 */
	int32_t (*get_sensors)(float gyros[3], float accels[3], float mag[3],
		float pos[3], float vel[3], float baro[1], float airspeed[1]);

	/**
	 * update_filter Compute one time step of the filter
	 * @param[in] dt       time step [s]
	 * @return 0 if succesfully update, -1 if not
	 */
	int32_t (*update_filter)(float dt);

	/**
	 * get_state Retrieve the state from the S(3) filter
	 * any param can be null indicating it is not being fetched
	 * @param[out] pos       the updated position in NED [m]
	 * @param[out] vel       the updated velocity in NED [m/s]
	 * @param[out] attitude  the updated attitude quaternion
	 * @param[out] gyro_bias the update gyro bias [deg/s]
	 * @param[out] airspeed  estimate of the airspeed
	 */
	int32_t (*get_state)(float pos[3], float vel[3], float attitude[4], float gyro_bias[3], float airspeed[1]);

	//! Working space for filter memory
	void * data;

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
	 * @returns 0 if sufficient data to run update
	 */
	int32_t (*get_sensors)();

	/**
	 * update_filter Compute one time step of the filter
	 * @param[in] dt       time step [s]
	 * @return 0 if succesfully update, -1 if not
	 */
	int32_t (*update_filter)(float dt);

	/**
	 * get_state Sets the state directly into the UAVOs
	 * @return 0 if succesful, -1 if not
	 */
	int32_t (*get_state)();

	/**
	 * Working space for filter memory.  A double pointer because the driver should
	 * be const but the working space should point to a pointer that can hold the
	 * real space.
	 */
	void ** data;

	//! To check the filter driver is valid, must be FILTER_GENERIC_MAGIC
	int32_t magic;
};

/***** union of all the filter classes *****/

// NOTE: we might want to move class into the first field of all the
// types to make it a 'flatter' structure but this would be less type
// safe, to the extent this is type safe at all
struct filter_driver {
	//! class must accurately match the following type
	enum filter_class class;

	//! Initialize the filter, allocate memory, and connect additional queues
	int32_t (*init)();

	//! Shut down the filter and disconnect all queues
	int32_t (*deinit)();

	//! Reset the filter state to default
	int32_t (*reset)();

	//! The specific driver for this filter implementation class
	union driver {
		struct filter_s3       driver_s3;
		struct filter_generic  driver_generic;
	};
};

// NOTE: we might have to switch to heap_4.c for deinit to make sense.  we can't even disconnect
// queues currently so you wouldn't be able to meaningfully disconnect a filter.  this is someting
// that we can bypass with a reboot for now

/**
 * @}
 */