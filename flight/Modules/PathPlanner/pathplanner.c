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

#include "flightstatus.h"
#include "pathdesired.h"
#include "pathplannersettings.h"
#include "pathstatus.h"
#include "positionactual.h"
#include "waypoint.h"
#include "waypointactive.h"
#include "modulesettings.h"

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
static WaypointData waypoint;
static bool path_status_updated;

// Private functions
static void advanceWaypoint();
static void checkTerminationCondition();
static void activateWaypoint();

static void pathPlannerTask(void *parameters);
static void settingsUpdated(UAVObjEvent * ev);
static void waypointsUpdated(UAVObjEvent * ev);
static void pathStatusUpdated(UAVObjEvent * ev);
static void createPathBox();
static void createPathLogo();

static bool module_enabled;

//! Store which waypoint has actually been pushed into PathDesired
static int32_t active_waypoint = -1;
//! Store the previous waypoint which is used to determine the path trajectory
static int32_t previous_waypoint = -1;
/**
 * Module initialization
 */
int32_t PathPlannerStart()
{
	if(module_enabled) {
		taskHandle = NULL;

		// Start VM thread
		xTaskCreate(pathPlannerTask, (signed char *)"PathPlanner", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_PATHPLANNER, taskHandle);
		return 0;
	}

	return -1;
}

/**
 * Module initialization
 */
int32_t PathPlannerInitialize()
{
	taskHandle = NULL;

#ifdef MODULE_PathPlanner_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_STATE_NUMELEM];
	ModuleSettingsStateGet(module_state);
	if (module_state[MODULESETTINGS_STATE_PATHPLANNER] == MODULESETTINGS_STATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if(module_enabled) {
		PathPlannerSettingsInitialize();
		WaypointInitialize();
		WaypointActiveInitialize();

		// Create object queue
		queue = xQueueCreate(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));

		return 0;
	}

	return -1;
}

MODULE_INITCALL(PathPlannerInitialize, PathPlannerStart)

/**
 * Module task
 */
static void pathPlannerTask(void *parameters)
{
	// If the PathStatus isn't available no follower is running and we should abort
	while (PathStatusHandle() == NULL || !TaskMonitorQueryRunning(TASKINFO_RUNNING_PATHFOLLOWER)) {
		AlarmsSet(SYSTEMALARMS_ALARM_PATHPLANNER, SYSTEMALARMS_ALARM_CRITICAL);
		vTaskDelay(1000);
	}
	AlarmsClear(SYSTEMALARMS_ALARM_PATHPLANNER);

	PathPlannerSettingsConnectCallback(settingsUpdated);
	settingsUpdated(PathPlannerSettingsHandle());

	WaypointConnectCallback(waypointsUpdated);
	WaypointActiveConnectCallback(waypointsUpdated);

	PathStatusConnectCallback(pathStatusUpdated);

	FlightStatusData flightStatus;

	// Main thread loop
	bool pathplanner_active = false;
	path_status_updated = false;

	while (1)
	{

		vTaskDelay(UPDATE_RATE_MS / portTICK_RATE_MS);

		// When not running the path planner short circuit and wait
		FlightStatusGet(&flightStatus);
		if (flightStatus.FlightMode != FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER) {
			pathplanner_active = false;
			continue;
		}

		if(pathplanner_active == false) {
			// This triggers callback to update variable
			WaypointActiveGet(&waypointActive);
			waypointActive.Index = 0;
			WaypointActiveSet(&waypointActive);

			// Reset the state.  Active waypoint sholud be set to an invalid
			// value to force waypoint 0 to become activated when starting
			active_waypoint = -1;
			previous_waypoint = -1;

			pathplanner_active = true;
			continue;
		}

		/* This method determines if we have achieved the goal of the active */
		/* waypoint */
		if (path_status_updated)
			checkTerminationCondition();

		/* If advance waypoint takes a long time to calculate then it should */
		/* be called from here when the active_waypoints does not equal the  */
		/* WaypointActive.Index                                              */
		/* if (active_waypoint != WaypointActive.Index)                      */
		/*     advanceWaypoint(WaypointActive.Index)                         */
	}
}

/**
 * On changed waypoints or active waypoint update position desired
 * if we are in charge
 */
static void waypointsUpdated(UAVObjEvent * ev)
{
	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);
	if (flightStatus.FlightMode != FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER)
		return;

	WaypointActiveGet(&waypointActive);
	if(active_waypoint != waypointActive.Index)
		activateWaypoint(waypointActive.Index);
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

	// Store the currently active waypoint.  This is used in activeWaypoint to plot
	// a waypoint from this (previous) waypoint to the newly selected one
	previous_waypoint = waypointActive.Index;

	// Default implementation simply jumps to the next possible waypoint.  Insert any
	// conditional logic desired here.
	// Note: In the case of conditional logic it is the responsibilty of the implementer
	// to ensure all possible paths are valid.
	waypointActive.Index++;

	if (waypointActive.Index >= UAVObjGetNumInstances(WaypointHandle())) {
		holdCurrentPosition();

		// Do not reset path_status_updated here to avoid this method constantly being called
		return;
	} else {
		WaypointActiveSet(&waypointActive);
	}

	// Invalidate any pending path status updates
	path_status_updated = false;
}

/**
 * This method is called when a new waypoint is activated
 *
 * Note: The way this is called, it runs in an object callback.  This is safe because
 * the execution time is extremely short.  If it starts to take a longer time then
 * the main task look should monitor a flag (such as the waypoint changing) and call
 * this method from the main task.
 */
static void activateWaypoint(int idx)
{
	active_waypoint = idx;

	if (idx >= UAVObjGetNumInstances(WaypointHandle())) {
		// Attempting to access invalid waypoint.  Fall back to position hold at current location
		AlarmsSet(SYSTEMALARMS_ALARM_PATHPLANNER, SYSTEMALARMS_ALARM_ERROR);
		holdCurrentPosition();
		return;
	}

	// Get the activated waypoint
	WaypointInstGet(idx, &waypoint);

	PathDesiredData pathDesired;

	pathDesired.End[PATHDESIRED_END_NORTH] = waypoint.Position[WAYPOINT_POSITION_NORTH];
	pathDesired.End[PATHDESIRED_END_EAST] = waypoint.Position[WAYPOINT_POSITION_EAST];
	pathDesired.End[PATHDESIRED_END_DOWN] = waypoint.Position[WAYPOINT_POSITION_DOWN];
	pathDesired.ModeParameters = waypoint.ModeParameters;

	// Use this to ensure the cases match up (catastrophic if not) and to cover any cases
	// that don't make sense to come from the path planner
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
			holdCurrentPosition();
			AlarmsSet(SYSTEMALARMS_ALARM_PATHPLANNER, SYSTEMALARMS_ALARM_ERROR);
			return;
	}

	pathDesired.EndingVelocity = waypoint.Velocity;

	if(previous_waypoint < 0) {
		// For first waypoint, get current position as start point
		PositionActualData positionActual;
		PositionActualGet(&positionActual);

		pathDesired.Start[PATHDESIRED_START_NORTH] = positionActual.North;
		pathDesired.Start[PATHDESIRED_START_EAST] = positionActual.East;
		pathDesired.Start[PATHDESIRED_START_DOWN] = positionActual.Down - 1;
		pathDesired.StartingVelocity = waypoint.Velocity;
	} else {
		// Get previous waypoint as start point
		WaypointData waypointPrev;
		WaypointInstGet(previous_waypoint, &waypointPrev);

		pathDesired.Start[PATHDESIRED_END_NORTH] = waypointPrev.Position[WAYPOINT_POSITION_NORTH];
		pathDesired.Start[PATHDESIRED_END_EAST] = waypointPrev.Position[WAYPOINT_POSITION_EAST];
		pathDesired.Start[PATHDESIRED_END_DOWN] = waypointPrev.Position[WAYPOINT_POSITION_DOWN];
		pathDesired.StartingVelocity = waypointPrev.Velocity;
	}

	PathDesiredSet(&pathDesired);

	// Invalidate any pending path status updates
	path_status_updated = false;

	AlarmsClear(SYSTEMALARMS_ALARM_PATHPLANNER);
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
