/**
 ******************************************************************************
 *
 * @file       paths.h
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
 * @brief      Header for path manager goal condition testing algorithms
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

#ifndef PATH_MANAGERS_H_
#define PATH_MANAGERS_H_

#include "pios.h"
#include "openpilot.h"

#include "pathsegmentdescriptor.h"


bool b_ball_goal_test(float position_NE[2], float swtiching_locus_NE[2], float threshold_distance);
bool half_plane_goal_test(float position_NE[2], float angular_distance_completed_D, float angular_distance_to_complete_D, float previous_locus[3], PathSegmentDescriptorData *pathSegmentDescriptor_current, PathSegmentDescriptorData *pathSegmentDescriptor_future, float advance_timing_ms, float nominal_groundspeed);

#endif // PATH_MANAGERS_H_
