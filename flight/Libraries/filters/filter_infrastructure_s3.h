/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @file       filter_infrastrcture_s3.h
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

 #if !defined(FILTER_INFRASTRUCTURE_S3)
 #define FILTER_INFRASTRUCTURE_S3

struct filter_infrastructure_s3_data {

};

// Initialize S(3) UAVOs
int32_t filter_infrastructure_s3_init(uintptr_t *id);

//! Connect S(3) queues
int32_t filter_infrastructure_s3_start(uintptr_t id);

//! Process an update for S(3)
int32_t filter_infrastructure_s3_process(struct filter_driver_s3 *driver, uintptr_t id, float dt);

 #endif /* FILTER_INFRASTRUCTURE_S3 */

 /**
  * @}
  */