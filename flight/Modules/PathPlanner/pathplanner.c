/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup PathPlannerModule Path Planner Module
 * @{ 
 *
 * @file       pathplanner.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      Simple path planner which activates a sequence of waypoints
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
#include "physical_constants.h"
#include "paths.h"
#include "path_saving.h"

#include "flightstatus.h"
#include "pathdesired.h"
#include "pathplannersettings.h"
#include "pathstatus.h"
#include "positionactual.h"
#include "waypoint.h"
#include "waypointactive.h"
#include "modulesettings.h"
#include "pios_thread.h"
#include "pios_queue.h"

// Private constants
#define STACK_SIZE_BYTES 1024
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW
#define MAX_QUEUE_SIZE 2
#define UPDATE_RATE_MS 20

// Private types

// Private variables
static struct pios_thread *taskHandle;
static struct pios_queue *queue;
static PathPlannerSettingsData pathPlannerSettings;
static WaypointActiveData waypointActive;
static WaypointData waypoint;
static bool path_completed;

// Private functions
static void advanceWaypoint();
static void activateWaypoint(int idx);

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
		taskHandle = PIOS_Thread_Create(pathPlannerTask, "PathPlanner", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
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
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_PATHPLANNER] == MODULESETTINGS_ADMINSTATE_ENABLED) {
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
		queue = PIOS_Queue_Create(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));
		FlightStatusConnectQueue(queue);

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
		PIOS_Thread_Sleep(1000);
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

	pathStatusUpdated(NULL);
	path_completed = false;

	while (1)
	{

		// Make sure when flight mode toggles, to immediately update the path
		UAVObjEvent ev;
		PIOS_Queue_Receive(queue, &ev, UPDATE_RATE_MS);

		// When not running the path planner short circuit and wait
		FlightStatusGet(&flightStatus);
		if (flightStatus.FlightMode != FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER) {
			pathplanner_active = false;
			continue;
		}

		if(pathplanner_active == false) {
			// Reset the state.  Active waypoint should be set to an invalid
			// value to force waypoint 0 to become activated when starting
			// Note: this needs to be done before the callback is triggered!
			active_waypoint = -1;
			previous_waypoint = -1;

			// This triggers callback to update variable
			WaypointActiveGet(&waypointActive);
			waypointActive.Index = 0;
			WaypointActiveSet(&waypointActive);

			pathplanner_active = true;
			continue;
		}

		/* This method determines if we have achieved the goal of the active */
		/* waypoint */
		if (path_completed)
			advanceWaypoint();

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
	if(active_waypoint != waypointActive.Index) {
		active_waypoint = waypointActive.Index;

		activateWaypoint(waypointActive.Index);
	}
}

/**
 * When the PathStatus is updated indicate a new one is available to consume
 */
static void pathStatusUpdated(UAVObjEvent * ev)
{
	PathStatusData pathStatus;

	PathStatusGet(&pathStatus);

	if ((pathStatus.Status == PATHSTATUS_STATUS_COMPLETED) &&
			(pathStatus.Waypoint == active_waypoint)) {
		path_completed = true;
	}
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
	pathDesired.Start[PATHDESIRED_START_NORTH] = position.North;
	pathDesired.Start[PATHDESIRED_START_EAST] = position.East;
	pathDesired.Start[PATHDESIRED_START_DOWN] = position.Down;
	pathDesired.End[PATHDESIRED_END_NORTH] = position.North;
	pathDesired.End[PATHDESIRED_END_EAST] = position.East;
	pathDesired.End[PATHDESIRED_END_DOWN] = position.Down;
	pathDesired.Mode = PATHDESIRED_MODE_HOLDPOSITION;
	pathDesired.StartingVelocity = 5; // This will be the max velocity it uses to try and hold
	pathDesired.EndingVelocity = 5;
	pathDesired.ModeParameters = 0;
	pathDesired.Waypoint = -1;
	PathDesiredSet(&pathDesired);
}

/**
 * Initial position hold at current position.  This is used at the end
 * of a path or in the case of a problem.
 */
static void holdLastPosition()
{
	uint32_t idx = UAVObjGetNumInstances(WaypointHandle());

	// Get the activated waypoint
	WaypointData waypoint;
	WaypointInstGet(idx-1, &waypoint);

	PositionActualData position;
	PositionActualGet(&position);

	PathDesiredData pathDesired;
	pathDesired.Start[PATHDESIRED_START_NORTH] = position.North;
	pathDesired.Start[PATHDESIRED_START_EAST] = position.East;
	pathDesired.Start[PATHDESIRED_START_DOWN] = position.Down;
	pathDesired.End[PATHDESIRED_END_NORTH] = waypoint.Position[WAYPOINT_POSITION_NORTH];
	pathDesired.End[PATHDESIRED_END_EAST] = waypoint.Position[WAYPOINT_POSITION_EAST];
	pathDesired.End[PATHDESIRED_END_DOWN] = waypoint.Position[WAYPOINT_POSITION_DOWN];
	pathDesired.Mode = PATHDESIRED_MODE_HOLDPOSITION;
	pathDesired.StartingVelocity = 5; // This will be the max velocity it uses to try and hold
	pathDesired.EndingVelocity = 5;
	pathDesired.ModeParameters = 0;
	pathDesired.Waypoint = -1;
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
	waypointActive.Index = previous_waypoint+1;

	if (waypointActive.Index >= UAVObjGetNumInstances(WaypointHandle())) {
		holdLastPosition();	// This means last in path.
	} else {
		WaypointActiveSet(&waypointActive);
	}

	path_completed = false;
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
	if ((idx >= UAVObjGetNumInstances(WaypointHandle())) || (idx < 0)) {
		// Attempting to access invalid waypoint.  Fall back to position hold at current location
		AlarmsSet(SYSTEMALARMS_ALARM_PATHPLANNER, SYSTEMALARMS_ALARM_ERROR);
		holdCurrentPosition();
		return;
	}

	// Get the activated waypoint
	WaypointInstGet(idx, &waypoint);

	PathDesiredData pathDesired;

	pathDesired.Waypoint = idx;

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
		case WAYPOINT_MODE_LAND:
			pathDesired.Mode = PATHDESIRED_MODE_LAND;
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
	path_completed = false;

	AlarmsClear(SYSTEMALARMS_ALARM_PATHPLANNER);
}

void settingsUpdated(UAVObjEvent * ev) {
	uint8_t preprogrammedPath = pathPlannerSettings.PreprogrammedPath;
	int32_t retval = 0;
	bool    operation = false;

	PathPlannerSettingsGet(&pathPlannerSettings);
	switch (pathPlannerSettings.FlashOperation) {
	case PATHPLANNERSETTINGS_FLASHOPERATION_LOAD1:
		retval = pathplanner_load_path(1);
		operation = true;
		break;
	case PATHPLANNERSETTINGS_FLASHOPERATION_LOAD2:
		retval = pathplanner_load_path(2);
		operation = true;
		break;
	case PATHPLANNERSETTINGS_FLASHOPERATION_LOAD3:
		retval = pathplanner_load_path(3);
		operation = true;
		break;
	case PATHPLANNERSETTINGS_FLASHOPERATION_LOAD4:
		retval = pathplanner_load_path(4);
		operation = true;
		break;
	case PATHPLANNERSETTINGS_FLASHOPERATION_LOAD5:
		retval = pathplanner_load_path(5);
		operation = true;
		break;
	case PATHPLANNERSETTINGS_FLASHOPERATION_SAVE1:
		retval = pathplanner_save_path(1);
		operation = true;
		break;
	case PATHPLANNERSETTINGS_FLASHOPERATION_SAVE2:
		retval = pathplanner_save_path(2);
		operation = true;
		break;
	case PATHPLANNERSETTINGS_FLASHOPERATION_SAVE3:
		retval = pathplanner_save_path(3);
		operation = true;
		break;
	case PATHPLANNERSETTINGS_FLASHOPERATION_SAVE4:
		retval = pathplanner_save_path(4);
		operation = true;
		break;
	case PATHPLANNERSETTINGS_FLASHOPERATION_SAVE5:
		retval = pathplanner_save_path(5);
		operation = true;
		break;
	}

	if (pathPlannerSettings.PreprogrammedPath != preprogrammedPath &&
	    pathPlannerSettings.FlashOperation == PATHPLANNERSETTINGS_FLASHOPERATION_NONE) {
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

	if (operation && (retval == 0)) {
		pathPlannerSettings.FlashOperation = PATHPLANNERSETTINGS_FLASHOPERATION_COMPLETED;
		PathPlannerSettingsSet(&pathPlannerSettings);
	} else if (retval != 0) {
		pathPlannerSettings.FlashOperation = PATHPLANNERSETTINGS_FLASHOPERATION_FAILED;
		PathPlannerSettingsSet(&pathPlannerSettings);
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
	waypoint.Velocity = 2.5;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;

	waypoint.Position[0] = 0;
	waypoint.Position[1] = 0;
	waypoint.Position[2] = -10;
	WaypointInstSet(0, &waypoint);

	waypoint.Position[0] = 5;
	waypoint.Position[1] = 5;
	waypoint.Position[2] = -10;
	WaypointInstSet(1, &waypoint);

	waypoint.Position[0] = -5;
	waypoint.Position[1] = 5;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	//waypoint.Mode = WAYPOINT_MODE_FLYCIRCLERIGHT;
	waypoint.ModeParameters = 35;
	WaypointInstSet(2, &waypoint);

	waypoint.Position[0] = -5;
	waypoint.Position[1] = -5;
	WaypointInstSet(3, &waypoint);

	waypoint.Position[0] = 5;
	waypoint.Position[1] = -5;
	WaypointInstSet(4, &waypoint);

	waypoint.Position[0] = 5;
	waypoint.Position[1] = 5;
	WaypointInstSet(5, &waypoint);

	waypoint.Position[0] = 0;
	waypoint.Position[1] = 0;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointInstSet(6, &waypoint);
}

static void createPathLogo()
{
	// Draw O
	WaypointData waypoint;
	waypoint.Velocity = 5; // Since for now this isn't directional just set a mag
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	waypoint.ModeParameters = 0;
	waypoint.Position[2] = -20;

	waypoint.Position[0] = 6.49;
	waypoint.Position[1] = -9.52;
	WaypointInstSet(0, &waypoint);

	waypoint.Position[0] = 6.32;
	waypoint.Position[1] = -94.82;
	WaypointCreateInstance();
	WaypointInstSet(1, &waypoint);

	waypoint.Position[0] = 6.32;
	waypoint.Position[1] = -77.13;
	WaypointCreateInstance();
	WaypointInstSet(2, &waypoint);

	waypoint.Position[0] = -17.04;
	waypoint.Position[1] = -77.071;
	WaypointCreateInstance();
	WaypointInstSet(3, &waypoint);

	waypoint.Position[0] = -26.42;
	waypoint.Position[1] = -69.30;
	waypoint.Mode = WAYPOINT_MODE_FLYCIRCLELEFT;
	waypoint.ModeParameters = 10;
	WaypointCreateInstance();
	WaypointInstSet(4, &waypoint);

	waypoint.Position[0] = -27.06;
	waypoint.Position[1] = -59.58;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	waypoint.ModeParameters = 0;
	WaypointCreateInstance();
	WaypointInstSet(5, &waypoint);

	waypoint.Position[0] = -22.37;
	waypoint.Position[1] = -51.81;
	waypoint.Mode = WAYPOINT_MODE_FLYCIRCLELEFT;
	waypoint.ModeParameters = 8;
	WaypointCreateInstance();
	WaypointInstSet(6, &waypoint);

	waypoint.Position[0] = -4.25;
	waypoint.Position[1] = -38.64;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	waypoint.ModeParameters = 0;
	WaypointCreateInstance();
	WaypointInstSet(7, &waypoint);

	waypoint.Position[0] = 6.33;
	waypoint.Position[1] = -45.74;
	waypoint.Mode = WAYPOINT_MODE_FLYCIRCLELEFT;
	waypoint.ModeParameters = 10;
	WaypointCreateInstance();
	WaypointInstSet(8, &waypoint);

	waypoint.Position[0] = -5.11;
	waypoint.Position[1] = -52.46;
	waypoint.Mode = WAYPOINT_MODE_FLYCIRCLELEFT;
	waypoint.ModeParameters = 10;
	WaypointCreateInstance();
	WaypointInstSet(9, &waypoint);

	waypoint.Position[0] = -26.84;
	waypoint.Position[1] = -41.45;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	waypoint.ModeParameters = 0;
	WaypointCreateInstance();
	WaypointInstSet(10, &waypoint);

	waypoint.Position[0] = -18.11;
	waypoint.Position[1] = -34.11;
	waypoint.Mode = WAYPOINT_MODE_FLYCIRCLERIGHT;
	waypoint.ModeParameters = 10;
	WaypointCreateInstance();
	WaypointInstSet(11, &waypoint);

	waypoint.Position[0] = -10.65;
	waypoint.Position[1] = -3.45;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	waypoint.ModeParameters = 0;
	WaypointCreateInstance();
	WaypointInstSet(12, &waypoint);

}

/**
 * @}
 * @}
 */
