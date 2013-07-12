/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @file       filter_inferface.c
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

#include "filter_interface.h"

/**
 * filter_validate Validate a filter has is safe to run and 
 * has a correct and matching driver
 * @param[in] filter    the filter to check
 * @return true if safe, false if not
 */
bool filter_interface_validate(struct filter_driver *filter)
{
	if (filter == NULL)
		return false;

	switch (filter->class) {
	case FILTER_CLASS_S3:
		return (filter->sub_driver.driver_s3.magic == FILTER_S3_MAGIC);
	case FILTER_CLASS_GENERIC:
		return (filter->sub_driver.driver_generic.magic == FILTER_GENERIC_MAGIC);
	}

	return false;
}
