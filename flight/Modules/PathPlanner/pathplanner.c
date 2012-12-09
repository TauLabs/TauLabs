/**
 ******************************************************************************
 * @file       pathplanner.c
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @brief      Executes a series of waypoints
 * @addtogroup OpenPilotModules OpenPilot Modules
 * @{
 * @addtogroup PathPlanner Path Planner Module
 * @brief Executes a series of waypoints
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

#include "openpilot.h"
#include "paths.h"
#include "path_calculation_simple.h"

#include "flightstatus.h"
#include "pathdesired.h"
#include "pathplannersettings.h"
#include "pathstatus.h"
#include "positionactual.h"
#include "waypoint.h"
#include "waypointactive.h"

// Private constants
#define STACK_SIZE_BYTES 1024
#define TASK_PRIORITY (tskIDLE_PRIORITY+1)
#define MAX_QUEUE_SIZE 2
#define UPDATE_RATE_MS 20

// Private types

// Private variables
static xTaskHandle taskHandle;
static xQueueHandle queue;
static PathPlannerSettingsData pathPlannerSettings;
static WaypointActiveData waypointActive;

//! Flag to indicate the status for the @ref VtolPathFollower or @ref FixedWingFollower
//! has been updated but not processed (in the main task thread)
static bool path_status_updated;

//! Store which waypoint has actually been pushed into PathDesired
static int32_t active_waypoint = -1;

//! Store the previous waypoint which is used to determine the path trajectory
static int32_t previous_waypoint = -1;

//! Flag to track if the waypoints have changed since it was activated
bool waypoints_dirty = false;


// Private functions
static void advanceWaypoint();
static void checkTerminationCondition();
static void holdCurrentPosition();
static void pathPlannerTask(void *parameters);
static void settingsUpdated(UAVObjEvent * ev);
static void waypointsActiveUpdated(UAVObjEvent * ev);
static void waypointsUpdated(UAVObjEvent * ev);
static void pathStatusUpdated(UAVObjEvent * ev);
static void createPathBox();
static void createPathLogo();

/**
 * Module initialization
 */
int32_t PathPlannerStart()
{
	taskHandle = NULL;

	// Start VM thread
	xTaskCreate(pathPlannerTask, (signed char *)"PathPlanner", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_PATHPLANNER, taskHandle);

	return 0;
}

/**
 * Module initialization
 */
int32_t PathPlannerInitialize()
{
	taskHandle = NULL;

	PathPlannerSettingsInitialize();
	WaypointInitialize();
	WaypointActiveInitialize();
	
	// Create object queue
	queue = xQueueCreate(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));

	return 0;
}

MODULE_INITCALL(PathPlannerInitialize, PathPlannerStart)

/**
 * Module task
 */
static void pathPlannerTask(void *parameters)
{
	PathPlannerSettingsConnectCallback(settingsUpdated);
	settingsUpdated(PathPlannerSettingsHandle());

	WaypointConnectCallback(waypointsUpdated);
	WaypointActiveConnectCallback(waypointsActiveUpdated);

	PathStatusConnectCallback(pathStatusUpdated);

	// If the PathStatus isn't available no follower is running and we should abort
	if (PathStatusHandle() == NULL || !TaskMonitorQueryRunning(TASKINFO_RUNNING_PATHFOLLOWER)) {
		AlarmsSet(SYSTEMALARMS_ALARM_PATHPLANNER, SYSTEMALARMS_ALARM_CRITICAL);
		vTaskSuspend(taskHandle);
	}
	AlarmsClear(SYSTEMALARMS_ALARM_PATHPLANNER);

	FlightStatusData flightStatus;
	
	// Main thread loop
	bool pathplanner_active = false;
	path_status_updated = false;

	while (1)
	{

		vTaskDelay(UPDATE_RATE_MS / portTICK_RATE_MS);

		/* When not running the path planner short circuit and wait */
		FlightStatusGet(&flightStatus);
		pathplanner_active = flightStatus.FlightMode == FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER;
		
		if(pathplanner_active == false) {
			/* When the path planner is not active reset to a known initial state */

			WaypointActiveGet(&waypointActive);
			waypointActive.Index = 0;
			WaypointActiveSet(&waypointActive);

			/* Reset the state.  Active waypoint sholud be set to an invalid */
			/* value to force waypoint 0 to become activated when starting   */
			active_waypoint = -1;
			previous_waypoint = -1;
		} else {

			/* The logic for the path planner is mixed between callbacks (events)  */
			/* from the UAVOs and the code which runs here which might take longer */
			/* to execute.  path_status_updated indicates the PathFollower has     */
			/* update its information and we sholud determine whether to change    */
			/* the waypoint.  If that happens, the active_waypoint will change and */
			/* we should activate that waypoint (which in the future might involve */
			/* calculating a specific trajectory)                                  */

			if (path_status_updated)
				checkTerminationCondition();

			if (active_waypoint != waypointActive.Index || waypoints_dirty == true) {
				int32_t activated_waypoint = select_waypoint_simple(waypointActive.Index, previous_waypoint);
				if (activated_waypoint != waypointActive.Index) {
					/* If the path calculation does not make the desired waypoint active something went    */
					/* wrong.  Current solution is to fire an alarm and try to stay where we are currently */
					/* so the operator can deal with this.  Note that select_waypoint_simple will be       */
					/* repeatedly called */
					holdCurrentPosition();
					AlarmsSet(SYSTEMALARMS_ALARM_PATHPLANNER, SYSTEMALARMS_ALARM_ERROR);
				} else {
					active_waypoint = activated_waypoint;

					/* Invalidate any pending path status updates */
					path_status_updated = false;

					/* We are synced up to the waypoints */
					waypoints_dirty = false;
				}
			}
		}
	}
}

/**
 * When the active waypoint is changed refresh the object.  This allows
 * changing the active waypoint during flight from other places (e.g. GCS)
 */
static void waypointsActiveUpdated(UAVObjEvent * ev)
{
	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);
	if (flightStatus.FlightMode != FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER)
		return;
	
	WaypointActiveGet(&waypointActive);
}

/**
 * When the waypoints are changed mark them as dirty so the path can be
 * refreshed in flight
 */
static void waypointsUpdated(UAVObjEvent * ev)
{
	waypoints_dirty = true;
}

/**
 * When the PathStatus is updated indicate a new one is available to consume
 */
static void pathStatusUpdated(UAVObjEvent * ev)
{
	path_status_updated = true;
}

/**
 * This method checks the current position against the active waypoint
 * to determine if it has been reached
 */
static void checkTerminationCondition()
{
	PathStatusData pathStatus;
	PathStatusGet(&pathStatus);
	path_status_updated = false;

	if (pathStatus.Status == PATHSTATUS_STATUS_COMPLETED)
		advanceWaypoint();
}

/**
 * Initial position hold at current position.  This is used at the end
 * of a path or in the case of a problem.
 */
static void holdCurrentPosition()
{
	// TODO: Define a separate error condition method which can select RTH versus PH
	PositionActualData position;
	PositionActualGet(&position);

	PathDesiredData pathDesired;
	pathDesired.End[PATHDESIRED_END_NORTH] = position.North;
	pathDesired.End[PATHDESIRED_END_EAST] = position.East;
	pathDesired.End[PATHDESIRED_END_DOWN] = position.Down;
	pathDesired.Mode = PATHDESIRED_MODE_HOLDPOSITION;
	PathDesiredSet(&pathDesired);
}

/**
 * Increment the waypoint index which triggers the active waypoint method
 */
static void advanceWaypoint()
{
	WaypointActiveGet(&waypointActive);

	/* Store the currently active waypoint.  This is used in activeWaypoint to plot */
	/* a waypoint from this (previous) waypoint to the newly selected one           */
	previous_waypoint = waypointActive.Index;

	/* Default implementation simply jumps to the next possible waypoint.  Insert any     */
	/* conditional logic desired here.                                                    */
	/* Note: In the case of conditional logic it is the responsibilty of the implementer  */
	/* to ensure all possible paths are valid.                                            */
	waypointActive.Index++;

	if (waypointActive.Index >= UAVObjGetNumInstances(WaypointHandle())) {
		holdCurrentPosition();

		/* Do not reset path_status_updated here to avoid this method constantly being called */
		return;
	} else {
		WaypointActiveSet(&waypointActive);
	}

	/* Invalidate any pending path status updates */
	path_status_updated = false;
}

void settingsUpdated(UAVObjEvent * ev) {
	uint8_t preprogrammedPath = pathPlannerSettings.PreprogrammedPath;
	PathPlannerSettingsGet(&pathPlannerSettings);
	if (pathPlannerSettings.PreprogrammedPath != preprogrammedPath) {
		switch(pathPlannerSettings.PreprogrammedPath) {
			case PATHPLANNERSETTINGS_PREPROGRAMMEDPATH_NONE:
				break;
			case PATHPLANNERSETTINGS_PREPROGRAMMEDPATH_10M_BOX:
				createPathBox();
				break;
			case PATHPLANNERSETTINGS_PREPROGRAMMEDPATH_LOGO:
				createPathLogo();
				break;
				
		}
	}
}

static void createPathBox()
{
	WaypointCreateInstance();
	WaypointCreateInstance();
	WaypointCreateInstance();
	WaypointCreateInstance();
	WaypointCreateInstance();
	
	// Draw O
	WaypointData waypoint;
	waypoint.Velocity = 5; // Since for now this isn't directional just set a mag
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;

	waypoint.Position[0] = 0;
	waypoint.Position[1] = 0;
	waypoint.Position[2] = -10;
	WaypointInstSet(0, &waypoint);

	waypoint.Position[0] = 25;
	waypoint.Position[1] = 25;
	waypoint.Position[2] = -10;
	WaypointInstSet(1, &waypoint);

	waypoint.Position[0] = -25;
	waypoint.Position[1] = 25;
	waypoint.Mode = WAYPOINT_MODE_FLYCIRCLELEFT;
	//waypoint.Mode = WAYPOINT_MODE_FLYCIRCLERIGHT;
	waypoint.ModeParameters = 35;
	WaypointInstSet(2, &waypoint);

	waypoint.Position[0] = -25;
	waypoint.Position[1] = -25;
	WaypointInstSet(3, &waypoint);

	waypoint.Position[0] = 25;
	waypoint.Position[1] = -25;
	WaypointInstSet(4, &waypoint);

	waypoint.Position[0] = 25;
	waypoint.Position[1] = 25;
	WaypointInstSet(5, &waypoint);

	waypoint.Position[0] = 0;
	waypoint.Position[1] = 0;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointInstSet(6, &waypoint);
}

static void createPathLogo()
{
	float scale = 1;
	
	// Draw O
	WaypointData waypoint;
	waypoint.Velocity = 5; // Since for now this isn't directional just set a mag
	for(uint32_t i = 0; i < 20; i++) {
		waypoint.Position[1] = scale * 30 * cos(i / 19.0 * 2 * M_PI);
		waypoint.Position[0] = scale * 50 * sin(i / 19.0 * 2 * M_PI);
		waypoint.Position[2] = -50;
		waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
		WaypointCreateInstance();
	}
	
	// Draw P
	for(uint32_t i = 20; i < 35; i++) {
		waypoint.Position[1] = scale * (55 + 20 * cos(i / 10.0 * M_PI - M_PI / 2));
		waypoint.Position[0] = scale * (25 + 25 * sin(i / 10.0 * M_PI - M_PI / 2));
		waypoint.Position[2] = -50;
		waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
		WaypointCreateInstance();
	}
		
	waypoint.Position[1] = scale * 35;
	waypoint.Position[0] = scale * -50;
	waypoint.Position[2] = -50;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointCreateInstance();
	WaypointInstSet(35, &waypoint);
	
	// Draw Box
	waypoint.Position[1] = scale * 35;
	waypoint.Position[0] = scale * -60;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointCreateInstance();
	WaypointInstSet(36, &waypoint);

	waypoint.Position[1] = scale * 85;
	waypoint.Position[0] = scale * -60;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointCreateInstance();
	WaypointInstSet(37, &waypoint);

	waypoint.Position[1] = scale * 85;
	waypoint.Position[0] = scale * 60;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointCreateInstance();
	WaypointInstSet(38, &waypoint);
	
	waypoint.Position[1] = scale * -40;
	waypoint.Position[0] = scale * 60;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointCreateInstance();
	WaypointInstSet(39, &waypoint);

	waypoint.Position[1] = scale * -40;
	waypoint.Position[0] = scale * -60;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointCreateInstance();
	WaypointInstSet(40, &waypoint);

	waypoint.Position[1] = scale * 35;
	waypoint.Position[0] = scale * -60;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointCreateInstance();
	WaypointInstSet(41, &waypoint);

}

/**
 * @}
 * @}
 */
