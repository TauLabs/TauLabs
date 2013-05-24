/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @file       filter_infrastrcture_s3.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Infrastructure for managing SE(3)+ filters
 *             because of the airspeed output this is slightly more than SE(3)
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

#if !defined(FILTER_INFRASTRUCTURE_S3)
#define FILTER_INFRASTRUCTURE_S3

// This should probably be opaque and the init should return uintptr_t
// for better API

//! Private data common for SE(3)+ filters
struct filter_infrastructure_s3_data {
	xQueueHandle gyroQueue;
	xQueueHandle accelQueue;
	xQueueHandle magQueue;
	xQueueHandle baroQueue;
	xQueueHandle gpsQueue;
	xQueueHandle gpsVelQueue;
};

// Initialize SE(3)+ UAVOs
int32_t filter_infrastructure_s3_init(struct filter_infrastructure_s3_data **data);

//! Connect  SE(3)+ queues
int32_t filter_infrastructure_s3_start(uintptr_t id);

//! Process an update for  SE(3)+
int32_t filter_infrastructure_s3_process(struct filter_driver_s3 *driver, uintptr_t id, float dt);

#endif /* FILTER_INFRASTRUCTURE_S3 */

 /**
  * @}
  */