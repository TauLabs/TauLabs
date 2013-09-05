/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup PathPlannerModule Path Planner Module
 * @{ 
 *
 * @file       pathplanner.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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
#include "path_saving.h"
#include "path_planners.h"

#include "fixedwingairspeeds.h"
#include "flightstatus.h"
#include "pathmanagersettings.h"
#include "pathmanagerstatus.h"
#include "pathplannersettings.h"
#include "pathsegmentdescriptor.h"
#include "pathplannerstatus.h"
#include "positionactual.h"
#include "waypoint.h"
#include "waypointactive.h"
#include "modulesettings.h"

#include "CoordinateConversions.h"
#include "misc_math.h"

// Private constants
#define STACK_SIZE_BYTES 1024
#define TASK_PRIORITY (tskIDLE_PRIORITY+0)
#define MAX_QUEUE_SIZE 2
// It is difficult to cleanly define how often the path planner should run. Generally, it is an
// iterative process, and so we would like to continue to refine the solution as long as there
// are spare processor cycles available. The upshot of this is that the best strategy is to run
// the process and add 1ms delays in the structure of the algorithms. This provides a break for
// other processes of the same priority, so that they can have a chance to run.
#define UPDATE_RATE_MS 100 // Cannot be greater than 200
#define IDLE_UPDATE_RATE_MS (200-UPDATE_RATE_MS)

// Private types
enum guidanceTypes{NOMANAGER, RETURNHOME, HOLDPOSITION, PATHPLANNER};

// Private variables
static xTaskHandle taskHandle;
static xQueueHandle queue;
static PathPlannerSettingsData pathPlannerSettings;
static PathPlannerStatusData pathPlannerStatus;
static bool process_waypoints_flag;
static bool module_enabled;
static bool path_manager_status_updated;
static PathPlannerSettingsPlannerAlgorithmOptions plannerAlgorithm;
static uint8_t guidanceType = NOMANAGER;

// Private functions
static void pathPlannerTask(void *parameters);
static void settingsUpdated(UAVObjEvent * ev);
static void waypointsUpdated(UAVObjEvent * ev);
static void pathManagerStatusUpdated(UAVObjEvent * ev);
static void createPathBox();
static void createPathStar();
static void createPathLogo();
static void createPathHoldPosition();
static void createPathReturnToHome();
static enum path_planner_states processWaypoints(PathPlannerSettingsPlannerAlgorithmOptions plannerAlgorithm);


////! Store which waypoint has actually been pushed into PathDesired
//static int32_t active_waypoint = -1;
////! Store the previous waypoint which is used to determine the path trajectory
//static int32_t previous_waypoint = -1;
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
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_PATHPLANNER] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if(module_enabled) {
		PathManagerSettingsInitialize();
		PathManagerStatusInitialize();
		PathPlannerSettingsInitialize();
		PathPlannerStatusInitialize();
		PathSegmentDescriptorInitialize();
		WaypointInitialize();
		WaypointActiveInitialize();

		// Create object queue
		queue = xQueueCreate(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));

		// This variable must only be set during the initialization process. This
		// is due to the vast differences in RAM requirements between path planners
		PathPlannerSettingsGet(&pathPlannerSettings);
		plannerAlgorithm = pathPlannerSettings.PlannerAlgorithm;

		return 0;
	}

	return -1;
}

MODULE_INITCALL(PathPlannerInitialize, PathPlannerStart);

/**
 * Module task
 */
static void pathPlannerTask(void *parameters)
{
	// If the PathManagerStatus isn't available no manager is running and we should abort
	while (PathManagerStatusHandle() == NULL || !TaskMonitorQueryRunning(TASKINFO_RUNNING_PATHMANAGER)) {
		AlarmsSet(SYSTEMALARMS_ALARM_PATHPLANNER, SYSTEMALARMS_ALARM_CRITICAL);
		vTaskDelay(1000);
	}
	AlarmsClear(SYSTEMALARMS_ALARM_PATHPLANNER);

	PathPlannerSettingsConnectCallback(settingsUpdated);
	settingsUpdated(PathPlannerSettingsHandle());

//	WaypointConnectCallback(waypointsUpdated);
	WaypointActiveConnectCallback(waypointsUpdated);

	PathManagerStatusConnectCallback(pathManagerStatusUpdated);

	// Main thread loop
	path_manager_status_updated = false;

	while (1)
	{
		FlightStatusData flightStatus;
		FlightStatusGet(&flightStatus);

		switch (flightStatus.FlightMode) {
			case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:
				if (guidanceType != RETURNHOME) {
					createPathReturnToHome();

					pathPlannerStatus.PathAvailability = PATHPLANNERSTATUS_PATHAVAILABILITY_NONE;
					PathPlannerStatusSet(&pathPlannerStatus);
					guidanceType = RETURNHOME;
					process_waypoints_flag = true;
				}
				break;
			case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
				if (guidanceType != HOLDPOSITION) {
					createPathHoldPosition();

					pathPlannerStatus.PathAvailability = PATHPLANNERSTATUS_PATHAVAILABILITY_NONE;
					PathPlannerStatusSet(&pathPlannerStatus);
					guidanceType = HOLDPOSITION;
					process_waypoints_flag = true;
				}
				break;
			case FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER:
				if (guidanceType != PATHPLANNER) {
					PathPlannerSettingsGet(&pathPlannerSettings);

					switch(pathPlannerSettings.PreprogrammedPath) {
						case PATHPLANNERSETTINGS_PREPROGRAMMEDPATH_NONE:
							if (UAVObjGetNumInstances(WaypointHandle()) > 1) {
								pathPlannerStatus.PathAvailability = PATHPLANNERSTATUS_PATHAVAILABILITY_NONE;
								pathPlannerStatus.NumberOfWaypoints = UAVObjGetNumInstances(WaypointHandle()); //Fixme: This is dangerous, because waypoints, once created, cannot be destroyed. This means that a long program followed by a short one will lead to the wrong number of waypoints!
								PathPlannerStatusSet(&pathPlannerStatus);

								guidanceType = PATHPLANNER;
								process_waypoints_flag = true;
							}
							else {
								// No path? In that case, burn some time and loop back to beginning. This is something that should be fixed as this takes the final form.
								guidanceType = NOMANAGER;
								vTaskDelay(IDLE_UPDATE_RATE_MS * portTICK_RATE_MS);
							}
							break;
						case PATHPLANNERSETTINGS_PREPROGRAMMEDPATH_10M_BOX:
							createPathBox();

							pathPlannerStatus.PathAvailability = PATHPLANNERSTATUS_PATHAVAILABILITY_NONE;
							PathPlannerStatusSet(&pathPlannerStatus);
							guidanceType = PATHPLANNER;
							process_waypoints_flag = true;
							break;
						case PATHPLANNERSETTINGS_PREPROGRAMMEDPATH_STAR:
							createPathStar();
							
							pathPlannerStatus.PathAvailability = PATHPLANNERSTATUS_PATHAVAILABILITY_NONE;
							PathPlannerStatusSet(&pathPlannerStatus);
							guidanceType = PATHPLANNER;
							process_waypoints_flag = true;
							break;
						case PATHPLANNERSETTINGS_PREPROGRAMMEDPATH_LOGO:
							createPathLogo();

							pathPlannerStatus.PathAvailability = PATHPLANNERSTATUS_PATHAVAILABILITY_NONE;
							PathPlannerStatusSet(&pathPlannerStatus);
							guidanceType = PATHPLANNER;
							process_waypoints_flag = true;
							break;
					}
				}
				break;
			default:
				// When not running the path manager, short circuit and wait
				guidanceType = NOMANAGER;

				pathPlannerStatus.PathAvailability = PATHPLANNERSTATUS_PATHAVAILABILITY_NONE;
				PathPlannerStatusSet(&pathPlannerStatus);

				vTaskDelay(IDLE_UPDATE_RATE_MS * portTICK_RATE_MS);

				continue;
		}

		vTaskDelay(UPDATE_RATE_MS * portTICK_RATE_MS);

		if(process_waypoints_flag)
		{
			enum path_planner_states ret;
			ret = processWaypoints(plannerAlgorithm);
			switch (ret) {
				case PATH_PLANNER_SUCCESS:
					{
						process_waypoints_flag = false;

						pathPlannerStatus.PathAvailability = PATHPLANNERSTATUS_PATHAVAILABILITY_PATHREADY;
						PathPlannerStatusSet(&pathPlannerStatus);
					}
					break;
				case PATH_PLANNER_PROCESSING:
					pathPlannerStatus.PathAvailability = PATHPLANNERSTATUS_PATHAVAILABILITY_PROCESSING;
					break;
				case PATH_PLANNER_STUCK:
					pathPlannerStatus.PathAvailability = PATHPLANNERSTATUS_PATHAVAILABILITY_NOCONVERGENCE;
					process_waypoints_flag = false;
					// Need to inform the FlightDirector that the planner cannot find a solution to the path
					break;
				case PATH_PLANNER_INSUFFICIENT_MEMORY:
					pathPlannerStatus.PathAvailability = PATHPLANNERSTATUS_PATHAVAILABILITY_OUTOFMEMORY;
					process_waypoints_flag = false;
					// Need to inform the FlightDirector that there isn't enough memory to continue. This could be because of refinement of the path, or because of too many waypoints
					break;
			}
		}
	}
}


enum path_planner_states processWaypoints(PathPlannerSettingsPlannerAlgorithmOptions algorithm)
{
	enum path_planner_states ret;
	
	switch(algorithm)
	{
		case PATHPLANNERSETTINGS_PLANNERALGORITHM_DIRECT:
			ret = direct_path_planner(pathPlannerStatus.NumberOfWaypoints);
			break;
		case PATHPLANNERSETTINGS_PLANNERALGORITHM_DIRECTWITHFILLETING:
			ret = direct_path_planner_with_filleting(pathPlannerStatus.NumberOfWaypoints, pathPlannerSettings.PreferredRadius);
			break;
		default:
			// TODO: Some kind of error here
			ret = PATH_PLANNER_PROCESSING;
			break;
	}

	return ret;
}




/**
 * On changed waypoints, replan the path
 */
static void waypointsUpdated(UAVObjEvent * ev)
{
	process_waypoints_flag = true;
}

/**
 * When the PathManagerStatus is updated indicate a new one is available to consume
 */
static void pathManagerStatusUpdated(UAVObjEvent * ev)
{
	path_manager_status_updated = true;
}

void settingsUpdated(UAVObjEvent * ev)
{
	uint8_t preprogrammedPath = pathPlannerSettings.PreprogrammedPath;
	int32_t retval = 0;
	bool    operation = false;

	PathPlannerSettingsGet(&pathPlannerSettings);
	if (pathPlannerSettings.PreprogrammedPath != preprogrammedPath) {
		switch(pathPlannerSettings.PreprogrammedPath) {
			case PATHPLANNERSETTINGS_PREPROGRAMMEDPATH_NONE:
				break;
			case PATHPLANNERSETTINGS_PREPROGRAMMEDPATH_10M_BOX:
				createPathBox();
				break;
			case PATHPLANNERSETTINGS_PREPROGRAMMEDPATH_STAR:
				createPathStar();
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


/******************
 ******************
 ******************/

static void createPathReturnToHome()
{
	WaypointData waypoint;

	float airspeedDesired;
	FixedWingAirspeedsBestClimbRateSpeedGet(&airspeedDesired);
	float radius = airspeedDesired*airspeedDesired/(GRAVITY*tanf(15*DEG2RAD)); // 15 degree average bank for staying on circle

	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	waypoint.Position[0] = 0;
	waypoint.Position[1] = 0;
	waypoint.Position[2] = positionActual.Down - 10;
	waypoint.Velocity = airspeedDesired;
	waypoint.Mode = WAYPOINT_MODE_CIRCLEPOSITIONRIGHT;
	waypoint.ModeParameters = radius;
	WaypointInstSet(0, &waypoint);

	pathPlannerStatus.NumberOfWaypoints = 1;
	PathPlannerStatusSet(&pathPlannerStatus);
}

static void createPathHoldPosition()
{
	WaypointData waypoint;

	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	float airspeedDesired;
	FixedWingAirspeedsBestClimbRateSpeedGet(&airspeedDesired);
	float radius = airspeedDesired*airspeedDesired/(GRAVITY*tanf(15*DEG2RAD)); // 15 degree average bank for staying on circle

	waypoint.Position[0] = positionActual.North;
	waypoint.Position[1] = positionActual.East;
	waypoint.Position[2] = positionActual.Down - 10;
	waypoint.Velocity = airspeedDesired;
	waypoint.Mode = WAYPOINT_MODE_CIRCLEPOSITIONLEFT;
	waypoint.ModeParameters = radius;
	WaypointInstSet(0, &waypoint);

	pathPlannerStatus.NumberOfWaypoints = 1;
	PathPlannerStatusSet(&pathPlannerStatus);
}

static void createPathBox()
{
	float airspeedDesired;
	FixedWingAirspeedsBestClimbRateSpeedGet(&airspeedDesired);
	float scale = 8.0f * airspeedDesired/12.0f;

	pathPlannerStatus.NumberOfWaypoints = 7;
	PathPlannerStatusSet(&pathPlannerStatus);

	for (int i=UAVObjGetNumInstances(WaypointHandle()); i<pathPlannerStatus.NumberOfWaypoints; i++) {
		WaypointCreateInstance();
	}

	WaypointData waypoint;
	waypoint.Velocity = airspeedDesired;

	waypoint.Position[0] = 0;
	waypoint.Position[1] = 0;
	waypoint.Position[2] = -10*250;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	waypoint.ModeParameters = 0;
	WaypointInstSet(0, &waypoint);

	waypoint.Position[0] = 5*scale;
	waypoint.Position[1] = 5*scale;
	waypoint.Position[2] = -10;
	WaypointInstSet(1, &waypoint);

	waypoint.Position[0] = -5*scale;
	waypoint.Position[1] = 5*scale;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	//waypoint.Mode = WAYPOINT_MODE_FLYCIRCLERIGHT;
	waypoint.ModeParameters = 35;
	WaypointInstSet(2, &waypoint);

	waypoint.Position[0] = -5*scale;
	waypoint.Position[1] = -5*scale;
	WaypointInstSet(3, &waypoint);

	waypoint.Position[0] = 5*scale;
	waypoint.Position[1] = -5*scale;
	WaypointInstSet(4, &waypoint);

	waypoint.Position[0] = 5*scale;
	waypoint.Position[1] = 5*scale;
	WaypointInstSet(5, &waypoint);

	waypoint.Position[0] = 0;
	waypoint.Position[1] = 0;
	waypoint.Mode = WAYPOINT_MODE_CIRCLEPOSITIONLEFT;
	waypoint.ModeParameters = 25*scale/4; // Quarter the size of the box
	WaypointInstSet(6, &waypoint);
}


static void createPathStar()
{
	float airspeedDesired;
	FixedWingAirspeedsBestClimbRateSpeedGet(&airspeedDesired);
	float scale = 12.0f * airspeedDesired/12.0f;
	
	float theta = 0;
	float step = 72*2*DEG2RAD; //This is the angular distance required to advance by two sides of a pentagram
	
	pathPlannerStatus.NumberOfWaypoints = 8;
	PathPlannerStatusSet(&pathPlannerStatus);
	
	for (int i=UAVObjGetNumInstances(WaypointHandle()); i<pathPlannerStatus.NumberOfWaypoints; i++) {
		WaypointCreateInstance();
	}
	
	WaypointData waypoint;
	waypoint.Velocity = airspeedDesired; // Since for now this isn't directional just set a mag
	
	// Start at home
	waypoint.Position[0] = 0;
	waypoint.Position[1] = 0;
	waypoint.Position[2] = -50;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	waypoint.ModeParameters = 0;
	WaypointInstSet(0, &waypoint);
	
	// Make five sides of star, plus one extra path to get to the start of the star
	for (int i=1; i<7; i++) {
		waypoint.Position[0] = 35*scale*cosf(theta);
		waypoint.Position[1] = 35*scale*sinf(theta);
		waypoint.Position[2] = -50;
		waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
		waypoint.ModeParameters = 0;
		WaypointInstSet(i, &waypoint);
		
		theta += step;
	}
	
	// Finish at home
	waypoint.Position[0] = 0;
	waypoint.Position[1] = 0;
	waypoint.Mode = WAYPOINT_MODE_CIRCLEPOSITIONRIGHT;
	waypoint.ModeParameters = 35*scale/2.0f; // Half the size of the box
	WaypointInstSet(7, &waypoint);
}


static void createPathLogo()
{
	float scale = 1;

	pathPlannerStatus.NumberOfWaypoints = 42;
	PathPlannerStatusSet(&pathPlannerStatus);

	for (int i=UAVObjGetNumInstances(WaypointHandle()); i<pathPlannerStatus.NumberOfWaypoints; i++) {
		WaypointCreateInstance();
	}


	WaypointData waypoint;
	waypoint.Velocity = 5; // Since for now this isn't directional just set a mag
	waypoint.ModeParameters = 0;

	// Draw O
	for(uint32_t i = 0; i < 20; i++) {
		waypoint.Position[0] = scale * 50 * sinf(i / 19.0 * 2 * PI);
		waypoint.Position[1] = scale * 30 * cosf(i / 19.0 * 2 * PI);
		waypoint.Position[2] = -50;
		waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
		WaypointInstSet(i, &waypoint);
	}

	// Draw P
	for(uint32_t i = 20; i < 35; i++) {
		waypoint.Position[0] = scale * (25 + 25 * sinf(i / 10.0 * PI - PI / 2));
		waypoint.Position[1] = scale * (55 + 20 * cosf(i / 10.0 * PI - PI / 2));
		waypoint.Position[2] = -50;
		waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
		WaypointInstSet(i, &waypoint);
	}

	waypoint.Position[0] = scale * -50;
	waypoint.Position[1] = scale * 35;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointInstSet(35, &waypoint);

	// Draw Box
	waypoint.Position[0] = scale * -60;
	waypoint.Position[1] = scale * 35;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointInstSet(36, &waypoint);

	waypoint.Position[0] = scale * -60;
	waypoint.Position[1] = scale * 85;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointInstSet(37, &waypoint);

	waypoint.Position[0] = scale * 60;
	waypoint.Position[1] = scale * 85;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointInstSet(38, &waypoint);

	waypoint.Position[0] = scale * 60;
	waypoint.Position[1] = scale * -40;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointInstSet(39, &waypoint);

	waypoint.Position[0] = scale * -60;
	waypoint.Position[1] = scale * -40;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointInstSet(40, &waypoint);

	waypoint.Position[0] = scale * -60;
	waypoint.Position[1] = scale * 35;
	waypoint.Position[2] = -30;
	waypoint.Mode = WAYPOINT_MODE_FLYVECTOR;
	WaypointInstSet(41, &waypoint);
}

/**
 * @}
 * @}
 */
