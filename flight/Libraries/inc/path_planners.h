/**
 ******************************************************************************
 *
 * @file       path_planners.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @brief      Header for path manipulation library 
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

#ifndef PATHS_LIBRARY_H_
#define PATHS_LIBRARY_H_

#include "pios.h"
#include "openpilot.h"

enum path_planner_states {PATH_PLANNER_SUCCESS, PATH_PLANNER_PROCESSING, PATH_PLANNER_STUCK, PATH_PLANNER_INSUFFICIENT_MEMORY};

enum path_planner_states direct_path_planner(uint16_t numberOfWaypoints);
enum path_planner_states direct_path_planner_with_filleting(uint16_t numberOfWaypoints, float fillet_radius);

#endif // PATHS_LIBRARY_H_
