/**
 ******************************************************************************
 *
 * @file       direct.c
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
#include "physical_constants.h"
#include "CoordinateConversions.h"
#include "misc_math.h"

#include "path_planners.h"

#include "pathmanagersettings.h"
#include "pathsegmentdescriptor.h"
#include "positionactual.h"
#include "waypoint.h"


// Private functions
static uint8_t addNonCircleToSwitchingLoci(float position[3], float finalVelocity, float curvature, uint16_t index);
static uint8_t addCircleToSwitchingLoci(float position[3], float finalVelocity, float curvature, float number_of_orbits, uint16_t index);

/**
 * @brief direct_path_planner Simplest of path planners. It connects waypoints together
 * with straight lines, regardless of vehicle dynamics or obstacles.
 * @param numberOfWaypoints The number of waypoints in the trajectory
 * @return Path planner's finite state
 */
enum path_planner_states direct_path_planner(uint16_t numberOfWaypoints)
{
	// Check for memory before generating new path descriptors
	if(1) //There is enough memory
	{
		// Generate the path segment descriptors
		for (uint16_t i=UAVObjGetNumInstances(PathSegmentDescriptorHandle()); i<UAVObjGetNumInstances(WaypointHandle())+1; i++) {
			//TODO: Ensure there is enough memory before generating
			PathSegmentDescriptorCreateInstance();
		}
	}
	else
		return PATH_PLANNER_INSUFFICIENT_MEMORY;

	PathSegmentDescriptorData pathSegmentDescriptor;

	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	pathSegmentDescriptor.SwitchingLocus[0] = positionActual.North;
	pathSegmentDescriptor.SwitchingLocus[1] = positionActual.East;
	pathSegmentDescriptor.SwitchingLocus[2] = positionActual.Down;
	pathSegmentDescriptor.FinalVelocity = 10;
	pathSegmentDescriptor.DesiredAcceleration = 0;
	pathSegmentDescriptor.NumberOfOrbits = 0;
	pathSegmentDescriptor.PathCurvature = 0;
	pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
	PathSegmentDescriptorInstSet(0, &pathSegmentDescriptor);

	uint16_t offset = 1;

	for(int wptIdx=0; wptIdx<numberOfWaypoints; wptIdx++) {
		WaypointData waypoint;
		WaypointInstGet(wptIdx, &waypoint);

		// Velocity is independent of path shape
		float final_velocity = waypoint.Velocity;

		// Determine if the path is a straight line or if it arcs
		float curvature = 0;
		bool path_is_circle = false;
		float number_of_orbits = 0;
		switch (waypoint.Mode)
		{
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

		// In the case of pure circles, the given waypoint is for a circle center
		// so we have to convert it into a pair of switching loci.
		if ( !path_is_circle ) {
			uint8_t ret;
			ret = addNonCircleToSwitchingLoci(waypoint.Position, final_velocity, curvature, wptIdx+offset);
			offset += ret;
		}
		else {
			uint8_t ret;
			ret = addCircleToSwitchingLoci(waypoint.Position, final_velocity, curvature, number_of_orbits, wptIdx+offset);
			offset += ret;
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
static uint8_t addNonCircleToSwitchingLoci(float position[3], float final_velocity, 
														 float curvature, uint16_t index)
{

	PathSegmentDescriptorData pathSegmentDescriptor;

	pathSegmentDescriptor.FinalVelocity = final_velocity;
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
 * @param index Current descriptor index
 * @return
 */
static uint8_t addCircleToSwitchingLoci(float circle_center[3], float finalVelocity, 
													 float curvature, float number_of_orbits, uint16_t index)
{
	PathSegmentDescriptorData pathSegmentDescriptor_old;
	PathSegmentDescriptorInstGet(index-1, &pathSegmentDescriptor_old);

	PathSegmentDescriptorData pathSegmentDescriptor;
	pathSegmentDescriptor.FinalVelocity = finalVelocity;
	pathSegmentDescriptor.DesiredAcceleration = 0;

	PathManagerSettingsData pathManagerSettings;
	PathManagerSettingsGet(&pathManagerSettings);

	float radius = fabsf(1.0f/curvature);

	// Calculate the approach angle from the previous switching locus to the waypoint
	float approachTheta_rad = atan2f(circle_center[1] - pathSegmentDescriptor_old.SwitchingLocus[1], circle_center[0] - pathSegmentDescriptor_old.SwitchingLocus[0]);

	// Calculate distance from previous waypoint to circle perimeter. (Distance to perimeter is distance to circle center minus radius)
	float d = sqrtf(powf(pathSegmentDescriptor.SwitchingLocus[0] - pathSegmentDescriptor_old.SwitchingLocus[0], 2) + powf(pathSegmentDescriptor.SwitchingLocus[1] - pathSegmentDescriptor_old.SwitchingLocus[1], 2)) - radius;

	if (d > pathManagerSettings.HalfPlaneAdvanceTiming*pathSegmentDescriptor.FinalVelocity) {
		if (index >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
			PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

		// Go straight toward circle center
		pathSegmentDescriptor.SwitchingLocus[0] = circle_center[0] - cosf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[1] = circle_center[1] - sinf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
		pathSegmentDescriptor.FinalVelocity = finalVelocity;
		pathSegmentDescriptor.PathCurvature = 0;
		pathSegmentDescriptor.NumberOfOrbits = 0;
		pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
		PathSegmentDescriptorInstSet(index, &pathSegmentDescriptor);

		// Add instances if necessary
		if (index+1 >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
			PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

		// Orbit position. Choose a point 90 degrees later in the arc so that the minor arc is the correct one.
		pathSegmentDescriptor.SwitchingLocus[0] = circle_center[0] + SIGN(curvature)*cosf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[1] = circle_center[1] - SIGN(curvature)*sinf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
		pathSegmentDescriptor.FinalVelocity = finalVelocity;
		pathSegmentDescriptor.PathCurvature = curvature;
		pathSegmentDescriptor.NumberOfOrbits = number_of_orbits;
		pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
		PathSegmentDescriptorInstSet(index+1, &pathSegmentDescriptor);
	}
	else {
		// Add instances if necessary
		if (index >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
			PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

		// Enter directly into circle
		pathSegmentDescriptor.SwitchingLocus[0] = circle_center[0] - cosf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[1] = circle_center[1] - sinf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
		pathSegmentDescriptor.FinalVelocity = finalVelocity;
		pathSegmentDescriptor.PathCurvature = 0;
		pathSegmentDescriptor.NumberOfOrbits = 0;
		pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
		PathSegmentDescriptorInstSet(index, &pathSegmentDescriptor);

		// Add instances if necessary
		if (index+1 >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
			PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

		// Orbit position. Choose a point 90 degrees later in the arc so that the minor arc is the correct one.
		pathSegmentDescriptor.SwitchingLocus[0] = circle_center[0] + SIGN(curvature)*cosf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[1] = circle_center[1] - SIGN(curvature)*sinf(approachTheta_rad)*radius;
		pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
		pathSegmentDescriptor.FinalVelocity = finalVelocity;
		pathSegmentDescriptor.PathCurvature = curvature;
		pathSegmentDescriptor.NumberOfOrbits = number_of_orbits;
		pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
		PathSegmentDescriptorInstSet(index+1, &pathSegmentDescriptor);
	}

	return 1;
}
