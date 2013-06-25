/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @file       template_se3_interface.c
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

#error THIS SHOULD NOT BE USED WITHOUT MODIFICATION

/*********************************************************/
/* This is a minimal template of an SE(3) filter to help */
/* people get up and running quickly with new filters.   */
/* The file should be renamed appropriately and all      */
/* cases of "template" replaced with the filter name     */
/* Then your local data should be defined in the         */
/* template_interface_data structure and your algorithm  */
/* should be connected in template_interface_update      */
/*********************************************************/

#include "filter_interface.h"
#include "filter_infrastructure_se3.h"

#include "openpilot.h"

#include "physical_constants.h"
#include "coordinate_conversions.h"

#include "stdint.h"
#include "stdbool.h"

static int32_t template_interface_init(uintptr_t *id);
static int32_t template_interface_reset(uintptr_t id);
static int32_t template_interface_update(uintptr_t id, float gyros[3], float accels[3], 
		float mag[3], float pos[3], float vel[3], float baro[1],
		float airspeed[1], float dt);
static int32_t template_interface_get_state(uintptr_t id, float pos[3], float vel[3],
		float attitude[4], float gyro_bias[3], float airspeed[1]);

struct filter_driver cf_filter_driver = {
	.class = FILTER_CLASS_S3,

	// this will initialize the SE(3)+ infrastrcture too
	.init = template_interface_init,

	// connects the SE(3)+ queues
	.start = filter_infrastructure_se3_start,
	.reset = template_interface_reset,
	.process = filter_infrastructure_se3_process,
	.sub_driver = {
		.driver_s3 = {
			.update_filter = template_interface_update,
			.get_state = cf_interface_get_state,
			.magic = FILTER_S3_MAGIC,
		}
	}
};


enum template_interface_magic {
	TEMPLATE_INTERFACE_MAGIC = 0x81AEBE6C,
};

struct template_interface_data {
	struct filter_infrastructure_se3_data *s3_data;
	float     q[4];
	float     pos[3];
	float     vel[3];
	float     gyros_bias[3];

	enum      template_interface_magic magic;
};

/**
 * Allocate a new filter instance
 * @return pointer to device or null if failure
 */
static struct template_interface_data *template_interface_alloc()
{
	struct template_interface_data *h;

	h = pvPortMalloc(sizeof(*cf));
	if (h == NULL)
		return NULL;

	h->q[0]         = 1;
	h->q[1]         = 0;
	h->q[2]         = 0;
	h->q[3]         = 0;
	h->initialization  = false;
	h->magic        = TEMPLATE_INTERFACE_MAGIC;

	return h;
}

/**
 * Validate a filter handle
 * @return true if a valid interface
 */
static bool template_interface_validate(struct template_interface_data *dev)
{
	if (dev == NULL)
		return false;
	if (dev->magic != TEMPLATE_INTERFACE_MAGIC)
		return false;
	return true;
}


/**
 * Initialize the complimentary filter and the SE(3)+ infrastructure
 * @param[out]  id   the handle for this filter instance
 * @return 0 if successful, -1 if not
 */
static int32_t template_interface_init(uintptr_t *id)
{
	// Allocate the data structure
	struct template_interface_data *template_interface_data = template_interface_alloc();
	if (template_interface_data == NULL)
		return -1;

	// Initialize the infrastructure
	if (filter_infrastructure_se3_init(&template_interface_data->s3_data) != 0)
		return -2;

	// Reset to known starting condition	
	template_interface_reset((uintptr_t) cf_interface_data);

	// Return the handle
	(*id) = (uintptr_t) template_interface_data;

	return 0;
}


/********* formatting sensor data to the core math code goes below here *********/

/**
 * Reset the filter state to default
 * @param[in]  id        the filter handle to reset
 */
static int32_t template_interface_reset(uintptr_t id)
{
	struct template_interface_data *h = (struct template_interface_data *) id;
	if (!template_interface_validate(h))
		return -1;

	/* TODO: this method resets the filter.  This is typically called at powerup */

	h->q[0]               = 1;
	h->q[1]               = 0;
	h->q[2]               = 0;
	h->q[3]               = 0;
	h->pos[0]             = 0;
	h->pos[1]             = 0;
	h->pos[2]             = 0;
	h->vel[0]             = 0;
	h->vel[1]             = 0;
	h->vel[2]             = 0;
	h->gyros_bias[0]      = 0;
	h->gyros_bias[1]      = 0;
	h->gyros_bias[2]      = 0;

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
static int32_t template_interface_update(uintptr_t id, float gyros[3], float accels[3], 
		float mag[3], float pos[3], float vel[3], float baro[1],
		float airspeed[1], float dt)
{
	struct template_interface_data *h = (struct template_interface_data *) id;
	if (!template_interface_validate(h))
		return -1;

	// This architecture currently always provides both but sanity check this
	if (accels == NULL || gyros == NULL)
		return -1;

	/********************************************/
	/* TODO: perform a filter update here       */
	/*   The inputs that are not null indicate  */
	/*   new data as documented above           */
	/*                                          */
	/* local data should be stored in the h     */
	/*   structure which is defined above       */
	/********************************************/

	if (true)
		AlarmsSet(SYSTEMALARMS_ALARM_ATTITUDE, SYSTEMALARMS_ALARM_CRITICAL);
	else
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
static int32_t template_interface_get_state(uintptr_t id, float pos[3], float vel[3],
		float attitude[4], float gyro_bias[3], float airspeed[1])
{
	struct template_interface_data *h = (struct template_interface_data *) id;
	if (!template_interface_validate(h))
		return -1;

	// If attitude is requested return it
	if (attitude){
		attitude[0] = h->q[0];
		attitude[1] = h->q[1];
		attitude[2] = h->q[2];
		attitude[3] = h->q[3];
	}

	// If position is requested return it
	if (pos) {
		pos[0] = h->pos[0];
		pos[1] = h->pos[1];
		pos[2] = h->pos[2];
	}

	// If velocity is requested return it
	if (vel) {
		vel[0] = h->vel[0];
		vel[1] = h->vel[1];
		vel[2] = h->vel[2];
	}

	// If gyro bias is requested then return it
	if (gyro_bias) {
		gyro_bias[0] = h->gyros_bias[0];
		gyro_bias[1] = h->gyros_bias[1];
		gyro_bias[2] = h->gyros_bias[2];
	}

	if (airspeed)
		airspeed[0] = 0;

	return 0;
}


/**
 * @}
 */

