/**
 ******************************************************************************
 *
 * @file       direct_with_filleting.c
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
 * @brief      Library path manipulation 
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

#include "pios.h"
#include "uavobjectmanager.h"
#include "physical_constants.h"
#include "path_planners.h"

#include "pathmanagersettings.h"
#include "pathplannerstatus.h"
#include "pathsegmentdescriptor.h"
#include "positionactual.h"
#include "velocityactual.h"
#include "waypoint.h"

#include "CoordinateConversions.h"
#include "misc_math.h"

// Private functions
static uint8_t addNonCircleToSwitchingLoci(float position[3], float finalVelocity, float curvature, uint16_t index);
static uint8_t addCircleToSwitchingLoci(float position[3], float finalVelocity, float curvature, float number_of_orbits, float fillet_radius, uint16_t index);

/**
 * @brief direct_path_planner_with_filleting An upgrade from the "direct" path planner.
 * It connects waypoints together with straight lines, and fillets, so that the
 * vehicle dynamics, i.e. Dubin's cart constraints, are taken into account. However
 * the "direct with filleting" path planner still assumes that there are no obstacles 
 * along the path.
 * The general approach is that before adding a new segment, the
 * path planner looks ahead at the next waypoint, and adds in fillets that align the vehicle with
 * this next waypoint.
 * @param numberOfWaypoints The number of waypoints in the trajectory
 * @param fillet_radius The radius the path planner will attempt to use when connecting
 * together two waypoint segments. This is only a suggestion, as if there is not sufficient distance
 * between waypoints then the fillet radius will be decreased to fit the available space.
 * @return Path planner's finite state
 */

enum path_planner_states direct_path_planner_with_filleting(uint16_t numberOfWaypoints, float fillet_radius)
{
	// Check for memory before generating new path descriptors. This is a little harder
	// since we don't know how many switching loci we'll need ahead of time. However, a
	// rough guess is we'll need twice as many loci as we do waypoints
	if(1) //There is enough memory
	{
		// Generate the path segment descriptors
		for (int i=UAVObjGetNumInstances(PathSegmentDescriptorHandle()); i<UAVObjGetNumInstances(WaypointHandle())+1; i++) {
			//TODO: Ensure there is enough memory before generating
			PathSegmentDescriptorCreateInstance();
		}
	}
	else
		return PATH_PLANNER_INSUFFICIENT_MEMORY;

	PathSegmentDescriptorData pathSegmentDescriptor_first;

	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	pathSegmentDescriptor_first.SwitchingLocus[0] = positionActual.North;
	pathSegmentDescriptor_first.SwitchingLocus[1] = positionActual.East;
	pathSegmentDescriptor_first.SwitchingLocus[2] = positionActual.Down;
	pathSegmentDescriptor_first.FinalVelocity = 100;
	pathSegmentDescriptor_first.DesiredAcceleration = 0;
	pathSegmentDescriptor_first.NumberOfOrbits = 0;
	pathSegmentDescriptor_first.PathCurvature = 0;
	pathSegmentDescriptor_first.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
	PathSegmentDescriptorInstSet(0, &pathSegmentDescriptor_first);

	uint16_t offset = 1;

	for(int wptIdx=0; wptIdx<numberOfWaypoints; wptIdx++) {
		WaypointData waypoint;
		WaypointInstGet(wptIdx, &waypoint);

		// Velocity is independent of path shape
		float finalVelocity = waypoint.Velocity;

		// Determine if the path is a straight line or if it arcs
		bool path_is_circle = false;
		float curvature = 0;
		float number_of_orbits = 0;
		switch (waypoint.Mode) {
		case WAYPOINT_MODE_CIRCLEPOSITIONRIGHT:
			path_is_circle = true;
			number_of_orbits = 1e8; //TODO: Define this really large floating-point value as a magic number
		case WAYPOINT_MODE_FLYCIRCLERIGHT:
		case WAYPOINT_MODE_DRIVECIRCLERIGHT:
			curvature = 1.0f/waypoint.ModeParameters;
			break;
		case WAYPOINT_MODE_CIRCLEPOSITIONLEFT:
			path_is_circle = true;
			number_of_orbits = 1e8; //TODO: Define this really large floating-point value as a magic number
		case WAYPOINT_MODE_FLYCIRCLELEFT:
		case WAYPOINT_MODE_DRIVECIRCLELEFT:
			curvature = -1.0f/waypoint.ModeParameters;
			break;
		}

		// Only add fillets if the radius is greater than 0, and this is not the last waypoint
		if (fillet_radius>0 && wptIdx<numberOfWaypoints-1)
		{

			// Determine tangent direction of old and new segment.
			PathSegmentDescriptorData pathSegmentDescriptor_old;
			PathSegmentDescriptorInstGet(wptIdx-1+offset, &pathSegmentDescriptor_old);

			WaypointData waypoint_future;
			WaypointInstGet(wptIdx+1, &waypoint_future);
			bool future_path_is_circle = false;

			switch (waypoint_future.Mode) {
			case WAYPOINT_MODE_CIRCLEPOSITIONRIGHT:
			case WAYPOINT_MODE_CIRCLEPOSITIONLEFT:
				future_path_is_circle = true;
				break;
			}


			float *swl_past = pathSegmentDescriptor_old.SwitchingLocus;
			float *swl_current = waypoint.Position;
			float *swl_future  = waypoint_future.Position;
			float q_future[3];
			float q_future_mag = 0;
			float q_current[3];
			float q_current_mag = 0;

			// In the case of line-line intersection lines, this is simply the direction of
			// the old and new segments.
			if (curvature == 0 && 
				 (waypoint_future.ModeParameters == 0 || future_path_is_circle)) { // Fixme: waypoint_future.ModeParameters needs to be replaced by waypoint_future.Mode. FOr this, we probably need a new function to handle the switch(waypoint.Mode)
				// Vector from past to present switching locus
				q_current[0] = swl_current[0] - swl_past[0];
				q_current[1] = swl_current[1] - swl_past[1];
				q_current[2] = 0;
				q_current_mag = VectorMagnitude(q_current); //Normalize

				// Calculate vector from preset to future switching locus
				q_future[0] = swl_future[0] - swl_current[0];
				q_future[1] = swl_future[1] - swl_current[1];
				q_future[2] = 0;
				q_future_mag = VectorMagnitude(q_future); //Normalize

			}
			//In the case of line-arc intersections, calculate the tangent of the new section.
			else if (curvature == 0 && 
						(waypoint_future.ModeParameters != 0 && !future_path_is_circle)) { // Fixme: waypoint_future.ModeParameters needs to be replaced by waypoint_future.Mode. FOr this, we probably need a new function to handle the switch(waypoint.Mode)
				/**
				 * Old segment: straight line
				 */
				q_current[0] = swl_current[0] - swl_past[0];
				q_current[1] = swl_current[1] - swl_past[1];
				q_current[2] = 0;
				q_current_mag = VectorMagnitude(q_current); //Normalize

				/**
				 * New segment: Vector perpendicular to the vector from arc center to tangent point
				 */
				bool clockwise = curvature > 0;
				int8_t lambda;

				if ((clockwise == true)) { // clockwise
					lambda = 1;
				} else { // counterclockwise
					lambda = -1;
				}

				// Calculate circle center
				float arcCenter_NE[2];
				find_arc_center(swl_current, swl_future, 1.0f/curvature, curvature > 0, true, arcCenter_NE);

				// Vector perpendicular to the vector from arc center to tangent point
				q_future[0] = -lambda*(swl_current[1] - arcCenter_NE[1]);
				q_future[1] = lambda*(swl_current[0] - arcCenter_NE[0]);
				q_future[2] = 0;
				q_future_mag = VectorMagnitude(q_future); //Normalize
			}
			//In the case of arc-line intersections, calculate the tangent of the old section.
			else if (curvature != 0 && (waypoint_future.ModeParameters == 0 || future_path_is_circle)) { // Fixme: waypoint_future.ModeParameters needs to be replaced by waypoint_future.Mode. FOr this, we probably need a new function to handle the switch(waypoint.Mode)
				/**
				 * Old segment: Vector perpendicular to the vector from arc center to tangent point
				 */
				bool clockwise = pathSegmentDescriptor_old.PathCurvature > 0;
				bool minor = pathSegmentDescriptor_old.ArcRank == PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
				int8_t lambda;

				if ((clockwise == true && minor == true) ||
						(clockwise == false && minor == false)) { //clockwise minor OR counterclockwise major
					lambda = 1;
				} else { //counterclockwise minor OR clockwise major
					lambda = -1;
				}

				// Calculate old circle center
				float arcCenter_NE[2];
				find_arc_center(swl_past, swl_current,
						1.0f/pathSegmentDescriptor_old.PathCurvature, clockwise, minor, arcCenter_NE);

				// Vector perpendicular to the vector from arc center to tangent point
				q_current[0] = -lambda*(swl_current[1] - arcCenter_NE[1]);
				q_current[1] = lambda*(swl_current[0] - arcCenter_NE[0]);
				q_current[2] = 0;
				q_current_mag = VectorMagnitude(q_current); //Normalize


				/**
				 * New segment: straight line
				 */
				q_future [0] = swl_future[0] - swl_current[0];
				q_future [1] = swl_future[1] - swl_current[1];
				q_future [2] = 0;
				q_future_mag = VectorMagnitude(q_future); //Normalize
			}
			//In the case of arc-arc intersections, calculate the tangent of the old and new sections.
			else if (curvature != 0 && (waypoint_future.ModeParameters != 0 && !future_path_is_circle)) { // Fixme: waypoint_future.ModeParameters needs to be replaced by waypoint_future.Mode. FOr this, we probably need a new function to handle the switch(waypoint.Mode)
				/**
				 * Old segment: Vector perpendicular to the vector from arc center to tangent point
				 */
				bool clockwise = pathSegmentDescriptor_old.PathCurvature > 0;
				bool minor = pathSegmentDescriptor_old.ArcRank == PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
				int8_t lambda;

				if ((clockwise == true && minor == true) ||
						(clockwise == false && minor == false)) { //clockwise minor OR counterclockwise major
					lambda = 1;
				} else { //counterclockwise minor OR clockwise major
					lambda = -1;
				}

				// Calculate old arc center
				float arcCenter_NE[2];
				find_arc_center(swl_past, swl_current,
						1.0f/pathSegmentDescriptor_old.PathCurvature, clockwise, minor, arcCenter_NE);

				/**
				 * New segment: Vector perpendicular to the vector from arc center to tangent point
				 */
				q_current[0] = -lambda*(swl_past[1] - arcCenter_NE[1]);
				q_current[1] = lambda*(swl_past[0] - arcCenter_NE[0]);
				q_current[2] = 0;
				q_current_mag = VectorMagnitude(q_current); //Normalize

				if (curvature > 0) { // clockwise
					lambda = 1;
				} else { // counterclockwise
					lambda = -1;
				}

				// Calculate new arc center
				find_arc_center(swl_current, swl_future, 1.0f/curvature, curvature > 0, true, arcCenter_NE);

				// Vector perpendicular to the vector from arc center to tangent point
				q_future[0] = -lambda*(swl_current[1] - arcCenter_NE[1]);
				q_future[1] = lambda*(swl_current[0] - arcCenter_NE[0]);
				q_future[2] = 0;
				q_future_mag = VectorMagnitude(q_future); //Normalize
			}

			// Normalize q_current and q_future
			if (q_current_mag > 0) {
				for (int i=0; i<3; i++)
					q_current[i] = q_current[i]/q_current_mag;
			}
			if (q_future_mag > 0) {
				for (int i=0; i<3; i++)
					q_future[i] = q_future[i]/q_future_mag;
			}

			// Compute heading difference between current and future tangents.
			float theta = angle_between_2d_vectors(q_current, q_future);

			// Compute angle between current and future tangents.
			float rho = circular_modulus_rad(theta - PI);

			// Compute half angle
			float rho2 = rho/2.0f;


			// If the angle is so acute that the fillet would be further away than the radius of a circle
			// then instead of filleting the angle to the inside, circle around it to the outside
			if (fabsf(rho) < PI/3.0f) { // This is the simplification of R/(sinf(fabsf(rho2)))-R > R
				// Find minimum radius R that permits the three fillets to be completed before arriving at the next waypoint.
				// Fixme: The vehicle might not be able to follow this path so the path manager should indicate this.
				float R = fillet_radius; // TODO: Link airspeed to preferred radius
				if (q_current_mag>0 && q_current_mag< R*sqrtf(3))
					R = q_current_mag/sqrtf(3)-0.1f; // Remove 10cm to guarantee that no two points overlap.
				if (q_future_mag >0 && q_future_mag < R*sqrtf(3))
					R = q_future_mag /sqrtf(3)-0.1f; // Remove 10cm to guarantee that no two points overlap.

				// The sqrt(3) term comes from the fact that the triangle that connects the center of
				// the first/second arc with the center of the second/third arc is a 1-2-sqrt(3) triangle
				float f1[3] = {swl_current[0] - R*q_current[0]*sqrtf(3), swl_current[1] - R*q_current[1]*sqrtf(3), swl_current[2]};
				float f2[3] = {swl_current[0] + R*q_future[0]*sqrtf(3), swl_current[1] + R*q_future[1]*sqrtf(3), swl_current[2]};

				/**
				 * Add the waypoint segment
				 */
				// In the case of pure circles, the given waypoint is for a circle center
				// so we have to convert it into a pair of switching loci.
				if ( !path_is_circle  ) {
					uint8_t ret;
					ret = addNonCircleToSwitchingLoci(f1, finalVelocity, curvature, wptIdx+offset);
					offset += ret;
				}
				else {
					uint8_t ret;
					ret = addCircleToSwitchingLoci(f1, finalVelocity, curvature, number_of_orbits, R, wptIdx+offset);
					offset += ret;
				}


				/**
				 * Add the filleting segments in preparation for the next waypoint
				 */
				offset++;
				if (wptIdx+offset >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
					PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

				float gamma = atan2f(q_current[1], q_current[0]);

				// Compute eta, which is the angle between the horizontal and the center of the filleting arc f1 and
				// sigma, which is the angle between the horizontal and the center of the filleting arc f2.
				float eta;
				float sigma;
				if (theta > 0) {  // Change in direction is clockwise, so fillets are clockwise
					eta = gamma - PI/2.0f;
					sigma = gamma + theta - PI/2.0f;
				}
				else {
					eta = gamma + PI/2.0f;
					sigma = gamma + theta + PI/2.0f;
				}

				// The switching locus is the midpoint between the center of filleting arc f1 and the circle
				PathSegmentDescriptorData pathSegmentDescriptor;
				pathSegmentDescriptor.SwitchingLocus[0] = (waypoint.Position[0] + (f1[0] + R*cosf(eta)))/2;
				pathSegmentDescriptor.SwitchingLocus[1] = (waypoint.Position[1] + (f1[1] + R*sinf(eta)))/2;
				pathSegmentDescriptor.SwitchingLocus[2] = waypoint.Position[2];
				pathSegmentDescriptor.FinalVelocity = finalVelocity;
				pathSegmentDescriptor.PathCurvature = -SIGN(theta)*1.0f/R;
				pathSegmentDescriptor.NumberOfOrbits = 0;
				pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
				PathSegmentDescriptorInstSet(wptIdx+offset, &pathSegmentDescriptor);

				offset++;
				if (wptIdx+offset >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
					PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

				// The switching locus is the midpoint between the center of filleting arc f2 and the circle
				pathSegmentDescriptor.SwitchingLocus[0] = (waypoint.Position[0] + (f2[0] + R*cosf(sigma)))/2;
				pathSegmentDescriptor.SwitchingLocus[1] = (waypoint.Position[1] + (f2[1] + R*sinf(sigma)))/2;
				pathSegmentDescriptor.SwitchingLocus[2] = waypoint.Position[2];
				pathSegmentDescriptor.FinalVelocity = finalVelocity;
				pathSegmentDescriptor.PathCurvature = SIGN(theta)*1.0f/R;
				pathSegmentDescriptor.NumberOfOrbits = 0;
				pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MAJOR;
				PathSegmentDescriptorInstSet(wptIdx+offset, &pathSegmentDescriptor);

				offset++;
				if (wptIdx+offset >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
					PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

				pathSegmentDescriptor.SwitchingLocus[0] = f2[0];
				pathSegmentDescriptor.SwitchingLocus[1] = f2[1];
				pathSegmentDescriptor.SwitchingLocus[2] = waypoint.Position[2];
				pathSegmentDescriptor.FinalVelocity = finalVelocity;
				pathSegmentDescriptor.PathCurvature = -SIGN(theta)*1.0f/R;
				pathSegmentDescriptor.NumberOfOrbits = 0;
				pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
				PathSegmentDescriptorInstSet(wptIdx+offset, &pathSegmentDescriptor);
			}
			else if (theta != 0) { // The two tangents have different directions
				// Find minimum radius R that permits the fillet to be completed before arriving at the next waypoint.
				// In any case, do not allow R to be 0
				// Fixme: The vehicle might not be able to follow this path so the path manager should indicate this.
				float R = fillet_radius; // TODO: Link airspeed to preferred radius

				if (q_current_mag>0 && q_current_mag<fabsf(R/tanf(rho2)))
					R = MIN(R, q_current_mag*fabsf(tanf(rho2))-0.1f); // Remove 10cm to guarantee that no two points overlap. This would be better if we solved it by removing the next point instead.
				if (q_future_mag>0  && q_future_mag <fabsf(R/tanf(rho2)))
					R = MIN(R, q_future_mag* fabsf(tanf(rho2))-0.1f); // Remove 10cm to guarantee that no two points overlap. This would be better if we solved it by removing the next point instead.

				/**
				 * Add the waypoint segment
				 */
				float f1[3];
				f1[0] = waypoint.Position[0] - R/fabsf(tanf(rho2))*q_current[0];
				f1[1] = waypoint.Position[1] - R/fabsf(tanf(rho2))*q_current[1];
				f1[2] = waypoint.Position[2];

				// In the case of pure circles, the given waypoint is for a circle center
				// so we have to convert it into a pair of switching loci.
				if ( !path_is_circle ) {
					uint8_t ret;
					ret = addNonCircleToSwitchingLoci(f1, finalVelocity, curvature, wptIdx+offset);
					offset += ret;
				}
				else {
					uint8_t ret;
					ret = addCircleToSwitchingLoci(f1, finalVelocity, curvature, number_of_orbits, R, wptIdx+offset);
					offset += ret;
				}


				/**
				 * Add the filleting segment in preparation for the next waypoint
				 */
				offset++;
				if (wptIdx+offset >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
					PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

				PathSegmentDescriptorData pathSegmentDescriptor;
				pathSegmentDescriptor.SwitchingLocus[0] = waypoint.Position[0] + R/fabsf(tanf(rho2))*q_future[0];
				pathSegmentDescriptor.SwitchingLocus[1] = waypoint.Position[1] + R/fabsf(tanf(rho2))*q_future[1];
				pathSegmentDescriptor.SwitchingLocus[2] = waypoint.Position[2];
				pathSegmentDescriptor.FinalVelocity = finalVelocity;
				pathSegmentDescriptor.PathCurvature = SIGN(theta)*1.0f/R;
				pathSegmentDescriptor.NumberOfOrbits = 0;
				pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
				PathSegmentDescriptorInstSet(wptIdx+offset, &pathSegmentDescriptor);
			}
			else { // In this case, the two tangents are colinear
				if ( !path_is_circle ) {
					uint8_t ret;
					ret = addNonCircleToSwitchingLoci(waypoint.Position, finalVelocity, curvature, wptIdx+offset);
					offset += ret;
				}
				else {
					uint8_t ret;
					ret = addCircleToSwitchingLoci(waypoint.Position, finalVelocity, curvature, number_of_orbits, fillet_radius, wptIdx+offset);
					offset += ret;
				}

			}
		}
		else if (wptIdx==numberOfWaypoints-1) // This is the final waypoint
		{
			// In the case of pure circles, the given waypoint is for a circle center
			// so we have to convert it into a pair of switching loci.
			if ( !path_is_circle ) {
				uint8_t ret;
				ret = addNonCircleToSwitchingLoci(waypoint.Position, finalVelocity, curvature, wptIdx+offset);
				offset += ret;
			}
			else {
				uint8_t ret;
				ret = addCircleToSwitchingLoci(waypoint.Position, finalVelocity, curvature, number_of_orbits, fillet_radius, wptIdx+offset);
				offset += ret;
			}
		}
	}

	return PATH_PLANNER_SUCCESS;
}


/**
 * @brief addNonCircleToSwitchingLoci In the case of pure circles, the given waypoint is for a circle center,
 * so we have to convert it into a pair of switching loci.
 * @param position Switching locus
 * @param finalVelocity Final velocity to be attained along path
 * @param curvature Path curvature
 * @param index Current descriptor index
 * @return
 */
static uint8_t addNonCircleToSwitchingLoci(float position[3], float finalVelocity, 
														 float curvature, uint16_t index)
{

	PathSegmentDescriptorData pathSegmentDescriptor;

	pathSegmentDescriptor.FinalVelocity = finalVelocity;
	pathSegmentDescriptor.DesiredAcceleration = 0;
	pathSegmentDescriptor.NumberOfOrbits = 0;
	pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;


	if (index >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
		PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

	pathSegmentDescriptor.SwitchingLocus[0] = position[0];
	pathSegmentDescriptor.SwitchingLocus[1] = position[1];
	pathSegmentDescriptor.SwitchingLocus[2] = position[2];
	pathSegmentDescriptor.PathCurvature = curvature;
	PathSegmentDescriptorInstSet(index, &pathSegmentDescriptor);

	return 0;
}


/**
 * @brief addCircleToSwitchingLoci In the case of pure circles, the given waypoint is for a circle center,
 * so we have to convert it into a pair of switching loci.
 * @param circle_center Center of orbit in NED coordinates
 * @param finalVelocity Final velocity to be attained along path
 * @param curvature Path curvature
 * @param number_of_orbits Number of complete orbits to be made before continuing to next descriptor
 * @param fillet_radius Radius of fillet joining together two path segments
 * @param index Current descriptor index
 * @return
 */
static uint8_t addCircleToSwitchingLoci(float circle_center[3], float finalVelocity, 
													 float curvature, float number_of_orbits, 
													 float fillet_radius, uint16_t index)
{
	PathSegmentDescriptorData pathSegmentDescriptor_old;
	PathSegmentDescriptorInstGet(index-1, &pathSegmentDescriptor_old);

	PathSegmentDescriptorData pathSegmentDescriptor;
	pathSegmentDescriptor.FinalVelocity = finalVelocity;
	pathSegmentDescriptor.DesiredAcceleration = 0;

	PathManagerSettingsData pathManagerSettings;
	PathManagerSettingsGet(&pathManagerSettings);

	// Calculate orbit radius
	float radius = fabsf(1.0f/curvature);


	uint16_t offset = 0;

	// Calculate the approach angle from the previous switching locus to the waypoint
	float approachTheta_rad = atan2f(circle_center[1] - pathSegmentDescriptor_old.SwitchingLocus[1], circle_center[0] - pathSegmentDescriptor_old.SwitchingLocus[0]);

	// Calculate squared distance from previous switching locus to circle center.
	float d2 = powf(circle_center[0] - pathSegmentDescriptor_old.SwitchingLocus[0], 2) + powf(circle_center[1] - pathSegmentDescriptor_old.SwitchingLocus[1], 2);

	if (d2 > radius*radius) { // Outside the circle
		// Go straight toward circle center. Stop at beginning of fillet.
		float f1[3] = {circle_center[0] - cosf(approachTheta_rad)*(sqrtf(radius*(2*fillet_radius+radius))),
					   circle_center[1] - sinf(approachTheta_rad)*(sqrtf(radius*(2*fillet_radius+radius))),
					   circle_center[2]};

		// Add instances if necessary
		if (index >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
			PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

		pathSegmentDescriptor.SwitchingLocus[0] = f1[0];
		pathSegmentDescriptor.SwitchingLocus[1] = f1[1];
		pathSegmentDescriptor.SwitchingLocus[2] = f1[2];
		pathSegmentDescriptor.FinalVelocity = finalVelocity;
		pathSegmentDescriptor.PathCurvature = 0;
		pathSegmentDescriptor.NumberOfOrbits = 0;
		pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
		PathSegmentDescriptorInstSet(index, &pathSegmentDescriptor);

		// Add instances if necessary
		offset++;
		if (index+offset >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
			PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

		// Form fillet. See documentation http://XYZ
		pathSegmentDescriptor.SwitchingLocus[0] = (circle_center[0] + (f1[0] + SIGN(curvature)*fillet_radius*sinf(approachTheta_rad)))*radius/(fillet_radius + radius);
		pathSegmentDescriptor.SwitchingLocus[1] = (circle_center[1] + (f1[1] - SIGN(curvature)*fillet_radius*cosf(approachTheta_rad)))*radius/(fillet_radius + radius);
		pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
		pathSegmentDescriptor.FinalVelocity = finalVelocity;
		pathSegmentDescriptor.PathCurvature = -SIGN(curvature)/fillet_radius;
		pathSegmentDescriptor.NumberOfOrbits = 0;
		pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
		PathSegmentDescriptorInstSet(index+offset, &pathSegmentDescriptor);

		// Add instances if necessary
		offset++;
		if (index+offset >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
			PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

		// Orbit position. Choose a point 90 degrees later in the arc so that the minor arc is the correct one.
		pathSegmentDescriptor.SwitchingLocus[0] = circle_center[0] + SIGN(curvature)*sinf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[1] = circle_center[1] - SIGN(curvature)*cosf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
		pathSegmentDescriptor.FinalVelocity = finalVelocity;
		pathSegmentDescriptor.PathCurvature = curvature;
		pathSegmentDescriptor.NumberOfOrbits = number_of_orbits;
		pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
		PathSegmentDescriptorInstSet(index+offset, &pathSegmentDescriptor);
	}
	else {
		// Since index 0 is always the vehicle's location, then if the vehicle is already inside the circle
		// on the index 1, then we don't have any information to help determine from which way the vehicle
		// will be approaching. In that case, use the vehicle velocity
		if (index == 1){
			VelocityActualData velocityActual;
			VelocityActualGet(&velocityActual);

			approachTheta_rad = atan2f(velocityActual.East, velocityActual.North);
		}


		// Add instances if necessary
		if (index >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
			PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

		// Form fillet
		pathSegmentDescriptor.SwitchingLocus[0] = circle_center[0] - SIGN(curvature)*radius*sinf(approachTheta_rad);
		pathSegmentDescriptor.SwitchingLocus[1] = circle_center[1] + SIGN(curvature)*radius*cosf(approachTheta_rad);
		pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
		pathSegmentDescriptor.FinalVelocity = finalVelocity;
		pathSegmentDescriptor.PathCurvature = curvature*2.0f;
		pathSegmentDescriptor.NumberOfOrbits = 0;
		pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
		PathSegmentDescriptorInstSet(index, &pathSegmentDescriptor);

		// Add instances if necessary
		offset++;
		if (index+offset >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
			PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

		// Orbit position. Choose a point 90 degrees later in the arc so that the minor arc is the correct one.
		pathSegmentDescriptor.SwitchingLocus[0] = circle_center[0] - cosf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[1] = circle_center[1] - sinf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
		pathSegmentDescriptor.FinalVelocity = finalVelocity;
		pathSegmentDescriptor.PathCurvature = curvature;
		pathSegmentDescriptor.NumberOfOrbits = number_of_orbits;
		pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
		PathSegmentDescriptorInstSet(index+offset, &pathSegmentDescriptor);
	}

	return offset;
}
