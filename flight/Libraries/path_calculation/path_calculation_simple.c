/**
 ******************************************************************************
 * @file       path_calculation_simple.c
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

#include "path_calculation_simple.h"

/**
 * Compute a path between two waypoints.  The new waypoint specifies how to get
 * there (path type).  This code should verify that a valid path exists.
 * @param[in] new_waypoint The waypoint to try and navigate too
 * @param[in] previous_waypoint The previous waypoint we are coming from.  -1
 *            indicates to use the current position
 * @returns The waypoint index selected if success or -1 if failure
 */
int32_t select_waypoint_simple(int32_t new_waypoint, int32_t previous_waypoint)
{
	WaypointData waypoint;

	if (new_waypoint >= UAVObjGetNumInstances(WaypointHandle())) {
		/* Attempting to access invalid waypoint.  Fall back to position hold at current location */
		return -1;
	}

	/* Get the activated waypoint */
	WaypointInstGet(new_waypoint, &waypoint);

	PathDesiredData pathDesired;

	pathDesired.End[PATHDESIRED_END_NORTH] = waypoint.Position[WAYPOINT_POSITION_NORTH];
	pathDesired.End[PATHDESIRED_END_EAST] = waypoint.Position[WAYPOINT_POSITION_EAST];
	pathDesired.End[PATHDESIRED_END_DOWN] = waypoint.Position[WAYPOINT_POSITION_DOWN];
	pathDesired.ModeParameters = waypoint.ModeParameters;

	/* Use this to ensure the cases match up (catastrophic if not) and to cover any cases */
	/* that don't make sense to come from the path planner                                */
	switch(waypoint.Mode) {
		case WAYPOINT_MODE_FLYVECTOR:
			pathDesired.Mode = PATHDESIRED_MODE_FLYVECTOR;
			break;
		case WAYPOINT_MODE_FLYENDPOINT:
			pathDesired.Mode = PATHDESIRED_MODE_FLYENDPOINT;
			break;
		case WAYPOINT_MODE_FLYCIRCLELEFT:
			pathDesired.Mode = PATHDESIRED_MODE_FLYCIRCLELEFT;
			break;
		case WAYPOINT_MODE_FLYCIRCLERIGHT:
			pathDesired.Mode = PATHDESIRED_MODE_FLYCIRCLERIGHT;
			break;
		default:
			return -1;
	}

	pathDesired.EndingVelocity = waypoint.Velocity;

	if(previous_waypoint < 0) {
		/* For first waypoint, get current position as start point */
		PositionActualData positionActual;
		PositionActualGet(&positionActual);
		
		pathDesired.Start[PATHDESIRED_START_NORTH] = positionActual.North;
		pathDesired.Start[PATHDESIRED_START_EAST] = positionActual.East;
		pathDesired.Start[PATHDESIRED_START_DOWN] = positionActual.Down - 1;
		pathDesired.StartingVelocity = waypoint.Velocity;
	} else {
		/* Get previous waypoint as start point */
		WaypointData waypointPrev;
		WaypointInstGet(previous_waypoint, &waypointPrev);
		
		pathDesired.Start[PATHDESIRED_END_NORTH] = waypointPrev.Position[WAYPOINT_POSITION_NORTH];
		pathDesired.Start[PATHDESIRED_END_EAST] = waypointPrev.Position[WAYPOINT_POSITION_EAST];
		pathDesired.Start[PATHDESIRED_END_DOWN] = waypointPrev.Position[WAYPOINT_POSITION_DOWN];
		pathDesired.StartingVelocity = waypointPrev.Velocity;
	}

	PathDesiredSet(&pathDesired);

	return new_waypoint;
}

/**
 * @}
 */