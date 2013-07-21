/**
 ******************************************************************************
 * @file       b_ball.c
 * @author     Tau Labs, http://www.taulabs.org, Copyright (C) 2013
 * @addtogroup Path Followers
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

#include "path_managers.h"

/**
 * @brief b_ball_goal_test B-ball approach. This tests if the vehicle is within a threshold distance.
 * From R. Beard and T. McLain, "Small Unmanned Aircraft: Theory and Practice", 2011, Section 11.1.
 * @param position_NE Current position in North-East coordinates
 * @param swtiching_locus_NE switching locus in North-East coordinates
 * @param threshold_distance radius of switching ball
 * @return
 */
bool b_ball_goal_test(float position_NE[2], float swtiching_locus_NE[2], float threshold_distance)
{
	bool advanceSegment_flag = false;

	// This method is less robust to error than the half-plane. It is cheaper and simpler, but those are it's only two advantages
	float d[2] = {position_NE[0] - swtiching_locus_NE[0], position_NE[1] - swtiching_locus_NE[1]};

	if (sqrtf(powf(d[0], 2) + powf(d[1], 2)) < threshold_distance) {
		advanceSegment_flag = true;
	}

	return advanceSegment_flag;
}
