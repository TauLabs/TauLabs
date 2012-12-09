/**
 ******************************************************************************
 * @file       path_calculation_simple.h
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @brief      A simple mapping from two waypoints to a path desired
 * @addtogroup PathCalculation Set of functions for computing PathDesired
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


#ifndef PATH_CALCULATION_SIMPLE_H
#define PATH_CALCULATION_SIMPLE_H

#include "openpilot.h"
#include "pathdesired.h"
#include "positionactual.h"
#include "waypoint.h"

//! Compute a path between two waypoints
int select_waypoint_simple(int32_t new_waypoint, int32_t previous_waypoint);

#endif /* PATH_CALCULATION_SIMPLE_H */

/**
 * @}
 */