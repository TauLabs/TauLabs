/**
 ******************************************************************************
 * @file       half_plane.c
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
#include "misc_math.h"
#include "coordinate_conversions.h"

/**
 * @brief half_plane_goal_test From R. Beard and T. McLain, "Small Unmanned Aircraft: Theory and Practice", 2011, Section 11.1.
 * Note: The half-plane approach has difficulties when the plane and the two loci are close to colinear
 * and reversing in direction. That is to say, a plane at P is supposed to go to A and then B:
 *   B<----------P->-----A
 * @param position_NE Current vehicle position in North-East coordinates
 * @param angular_distance_completed_D If the vehicle is following an arc, then this is the current distance it has traveled along the arc
 * @param angular_distance_to_complete_D If the vehicle is following an arc, then this is the current distance it must travel along the arc
 * @param previous_locus_NED This is the past switching locus position in NED coordinates
 * @param pathSegmentDescriptor_current This is the current path descriptor
 * @param pathSegmentDescriptor_future This is the future path descriptor
 * @param advance_timing_ms This parameter controls how many milliseconds in the future the test is looking
 * @param nominal_groundspeed This is the vehicle's nominal ground speed. It does not take into account wind effects.
 * @return
 */
bool half_plane_goal_test(float position_NE[2], float angular_distance_completed_D, float angular_distance_to_complete_D, PathSegmentDescriptorData *pathSegmentDescriptor_past,
						  PathSegmentDescriptorData *pathSegmentDescriptor_current, PathSegmentDescriptorData *pathSegmentDescriptor_future,
						  float advance_timing_ms, float nominal_groundspeed)
{
	bool advanceSegment_flag = false;

	// Calculate vector from past to preset switching locus
	float *swl_past = pathSegmentDescriptor_past->SwitchingLocus;
	float *swl_current = pathSegmentDescriptor_current->SwitchingLocus;
	float q_current[3] = {swl_current[0] - swl_past[0], swl_current[1] - swl_past[1], 0};
	float q_current_mag = VectorMagnitude(q_current); //Normalize
	float q_future[3];
	float q_future_mag;

	// Line-line intersection. The halfplane frontier is the bisecting line between the arrival
	// and departure vector.
	if (pathSegmentDescriptor_current->PathCurvature == 0 && pathSegmentDescriptor_future->PathCurvature == 0) {
		float *swl_future  = pathSegmentDescriptor_future->SwitchingLocus;

		// Calculate vector from preset to future switching locus
		q_future [0] = swl_future[0] - swl_current[0];
		q_future [1] = swl_future[1] - swl_current[1];
		q_future [2] = 0;
		q_future_mag = VectorMagnitude(q_future); //Normalize

	}
	// "Small Unmanned Aircraft: Theory and Practice" provides no guidance for the perpendicular
	// intersection of a line and an arc. However, it seems reasonable to consider the halfplane
	// as occurring at the intersection between the vector and the arc, with the frontier defined
	// as the half-angle between the arriving vector and the departing arc tangent, similar to
	// the line-line case.
	//
	// The nice part about this approach is that it works equally well for a tangent curve, as the intersection
	// occurs at the tangent of the circle. In the case that due to numerical error the vector and arc do
	// not intersect, we will still test for crossing into the half plane defined of a line drawn between the arc's
	// center and the closest point on the vector to the arc.
	else if (pathSegmentDescriptor_current->PathCurvature == 0 && pathSegmentDescriptor_future->PathCurvature != 0)
	{
		// Calculate vector tangent to arc at preset switching locus. This comes from geometry that that tangent to a circle
		// is the perpendicular vector to the vector connecting the tangent point and the center of the circle. The vector R
		// is tangent_point - arc_center, so the perpendicular to R is <-lambda*Ry,lambda*Rx>, where lambda = +-1.

		bool clockwise = pathSegmentDescriptor_future->PathCurvature > 0;
		bool minor = pathSegmentDescriptor_future->ArcRank == PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
		int8_t lambda;

		if ((clockwise == true && minor == true) ||
				(clockwise == false && minor == false)) { //clockwise minor OR counterclockwise major
			lambda = 1;
		} else { //counterclockwise minor OR clockwise major
			lambda = -1;
		}

		enum arc_center_results arc_has_center;
		float arc_center_NE_future[2];
		arc_has_center = find_arc_center(pathSegmentDescriptor_current->SwitchingLocus,
										 pathSegmentDescriptor_future->SwitchingLocus,
										 1.0f/pathSegmentDescriptor_future->PathCurvature,
										 pathSegmentDescriptor_future->PathCurvature > 0,
										 pathSegmentDescriptor_future->ArcRank == PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR,
										 arc_center_NE_future);

		if (arc_has_center == ARC_CENTER_FOUND) {
			// Arc tangent vector at intersection of line and arc
			q_future[0] = -lambda*(swl_current[1] - arc_center_NE_future[1]);
			q_future[1] =  lambda*(swl_current[0] - arc_center_NE_future[0]);
			q_future[2] = 0;
			q_future_mag = VectorMagnitude(q_future); //Normalize

		}
		else { //---- This is bad, but we have to handle it.----///
			advanceSegment_flag = false; // Fixme: This means that the segment descriptor will never advance,
										 //unless it is changed by the timeout or overshoot scenarios

			return advanceSegment_flag;
		}
	}
	// "Small Unmanned Aircraft: Theory and Practice" provides no guidance for the perpendicular
	// intersection of an arc and a line. However, it seems reasonable to consider the halfplane
	// occurring at the intersection between the arc and the vector. The halfplane's frontier
	// is perpendicular to the tangent at the end of the arc.
	else if (pathSegmentDescriptor_current->PathCurvature != 0 && pathSegmentDescriptor_future->PathCurvature == 0)
	{
		// Cheat by remarking that the plane defined by the radius is perfectly defined by the angle made
		// between the center and the end of the trajectory. So if the vehicle has traveled further than
		// the required angular distance, it has crossed this
		if (SIGN(pathSegmentDescriptor_current->PathCurvature) * (angular_distance_completed_D - angular_distance_to_complete_D) >= 0)
			advanceSegment_flag = true;

		return advanceSegment_flag;
	}
	// "Small Unmanned Aircraft: Theory and Practice" provides no guidance for the perpendicular
	// intersection of two arcs. However, it seems reasonable to consider the halfplane
	// occurring at the intersection between the two arcs. The halfplane's frontier is defined
	// as the half-angle between the arriving arc tangent and the departing arc tangent, similar to
	// the line-line case.
	else if (pathSegmentDescriptor_current->PathCurvature != 0 && pathSegmentDescriptor_future->PathCurvature != 0)
	{
		// Cheat by remarking that the plane defined by the radius is perfectly defined by the angle made
		// between the center and the end of the trajectory. So if the vehicle has traveled further than
		// the required angular distance, it has crossed this
		if (SIGN(pathSegmentDescriptor_current->PathCurvature) * (angular_distance_completed_D - angular_distance_to_complete_D) >= 0)
			advanceSegment_flag = true;

		return advanceSegment_flag;
	}
	else {
		// Shouldn't be able to get here. Something has gone wrong.
		// TODO.
		AlarmsSet(SYSTEMALARMS_ALARM_PATHMANAGER, SYSTEMALARMS_ALARM_CRITICAL);
		return false;
	}

	// Compute the half-plane frontier as the line perpendicular to the sum of the approach and
	// departure vectors. See Fig 11.1 in reference.
	//
	// We're going to take a litle mathematical shortcut, by utilizing the fact that we don't need the actual
	// normalized normal vector, any normal vector will do. If a and b are vectors, then
	// a/|a|+b/|b| = 1/(|a||b|)*(a*|b| + b*|a|), which points in the same direction as (a*|b| + b*|a|)
	float half_plane[3] = {q_future[0]*q_current_mag + q_current[0]*q_future_mag,
					q_future[1]*q_current_mag + q_current[1]*q_future_mag,
					q_future[2]*q_current_mag + q_current[2]*q_future_mag};

	// Test if the UAV is in the half plane, H. This is easy by taking advantage of simple vector
	// calculus: a.b = |a||b|cos(theta), but since |a|,|b| >=0, then a.b > 0 if and only if
	// cos(theta) > 0, which means that the UAV is in I or IV quadrants, i.e. is in the half plane.
	float p[2] = {position_NE[0] - swl_current[0], position_NE[1] - swl_current[1]};

	// If we want to switch based on nominal time to locus, add the normalized q_current times the speed times the timing advace
	if (advance_timing_ms != 0) {
		for (int i=0; i<2; i++) {
				p[i] += q_current[i]/q_current_mag * nominal_groundspeed * (advance_timing_ms/1000.0f);
		}
	}

	// Finally test a.b > 0
	if (p[0]*half_plane[0] + p[1]*half_plane[1] > 0) {
		advanceSegment_flag = true;
	}

	return advanceSegment_flag;
}
