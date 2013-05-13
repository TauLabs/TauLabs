/**
 ******************************************************************************
 * @file       pathmanager.c
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
 * @brief      Executes a series of paths
 * @addtogroup Modules
 * @{
 * @addtogroup PathManager Path Manager Module
 * @brief The path manager switches between motion descriptors in order to maneuver
 * along the path
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
#include "physical_constants.h"
#include "misc_math.h"

#include "flightstatus.h"
#include "positionactual.h"
#include "waypoint.h"
#include "waypointactive.h"
#include "modulesettings.h"

#include "fixedwingairspeeds.h"

#include "pathmanagerstatus.h"
#include "pathmanagersettings.h"
#include "pathplannerstatus.h"
#include "pathsegmentdescriptor.h"
#include "paths_library.h"

#include "CoordinateConversions.h"

// Private constants
#define STACK_SIZE_BYTES 700
#define TASK_PRIORITY (tskIDLE_PRIORITY+1)
#define MAX_QUEUE_SIZE 2
#define UPDATE_RATE_MS 100 // Cannot be greater than 200
#define IDLE_UPDATE_RATE_MS (200-UPDATE_RATE_MS)
#define OVERSHOOT_TIMER_MS 1000
#define ANGULAR_PROXIMITY_THRESHOLD 30

// Private types
enum guidanceTypes{NOMANAGER, RETURNHOME, HOLDPOSITION, PATHPLANNER};
static struct PreviousLocus {
	float Position[3];

	float Velocity;
} *previousLocus;

// Private variables
static bool module_enabled;
static xTaskHandle taskHandle;
static xQueueHandle queue;
static FixedWingAirspeedsData fixedWingAirspeeds;
static PathManagerSettingsData pathManagerSettings;
static PathManagerStatusData pathManagerStatus;
static PathSegmentDescriptorData pathSegmentDescriptor_current;
static portTickType segmentTimer;
static float angularDistanceToComplete_D;
static float angularDistanceCompleted_D;
static float oldPosition_NE[2];
static float arcCenter_NE[2];
static uint8_t guidanceType = NOMANAGER;
static enum arc_center_results arc_has_center = INSUFFICIENT_RADIUS;

// Private functions
static bool checkGoalCondition();
static void checkOvershoot();
static void pathManagerTask(void *parameters);
static void settingsUpdated(UAVObjEvent * ev);
static void advanceSegment();

/**
 * Module initialization
 */
int32_t PathManagerStart()
{
	if (module_enabled) {
		taskHandle = NULL;

		// Start VM thread
		xTaskCreate(pathManagerTask, (signed char *)"PathManager", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_PATHMANAGER, taskHandle);
		return 0;
	}

	return -1;
}

/**
 * Module initialization
 */
int32_t PathManagerInitialize()
{
	taskHandle = NULL;

#ifdef MODULE_PathManager_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_PATHMANAGER] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (module_enabled) {
		PathManagerStatusInitialize();
		PathManagerSettingsInitialize();
		PathSegmentDescriptorInitialize();

		FixedWingAirspeedsInitialize(); //TODO: This shouldn't really be here, as it's airframe specific

		// Create object queue
		queue = xQueueCreate(MAX_QUEUE_SIZE, sizeof(UAVObjEvent)); //TODO: Is this even necessary?

		// Allocate memory
		previousLocus = (struct PreviousLocus *) pvPortMalloc(sizeof(struct PreviousLocus));
		memset(previousLocus, 0, sizeof(struct PreviousLocus));
		return 0;
	}

	return -1;
}

MODULE_INITCALL(PathManagerInitialize, PathManagerStart);

/**
 * Module task
 */
static void pathManagerTask(void *parameters)
{
	// If no follower is running then we cannot continue
	while (!TaskMonitorQueryRunning(TASKINFO_RUNNING_PATHFOLLOWER)) {
		AlarmsSet(SYSTEMALARMS_ALARM_PATHMANAGER, SYSTEMALARMS_ALARM_CRITICAL);
		vTaskDelay(1000);
	}
	AlarmsClear(SYSTEMALARMS_ALARM_PATHMANAGER);

	// Connect callbacks
	PathManagerSettingsConnectCallback(settingsUpdated);
	FixedWingAirspeedsConnectCallback(settingsUpdated);

	// Force reload all settings
	settingsUpdated(NULL);

	// Initialize all main loop variables
	bool pathplanner_active = false;
	static portTickType lastSysTime;
	static portTickType overshootTimer;
	lastSysTime = xTaskGetTickCount();
	overshootTimer = xTaskGetTickCount();
	uint8_t theta_roundoff_trim_count = 0;

	// Main thread loop
	while (1)
	{
		// Wait
		vTaskDelayUntil(&lastSysTime, UPDATE_RATE_MS * portTICK_RATE_MS);

#if !defined PATH_PLANNER // If there is no path planner, it's probably because memory is too scarce, such as on CC/CC3D. In that case, provide a return to home and a position hold
		// Check flight mode
		FlightStatusData flightStatus;
		FlightStatusGet(&flightStatus);

		switch (flightStatus.FlightMode) {
			case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:

				if (guidanceType != RETURNHOME) {
					guidanceType = RETURNHOME;
					pathplanner_active = false;

					// Load pregenerated return to home program
					simple_return_to_home();
				}
				break;
			case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
				if (guidanceType != HOLDPOSITION) {
					guidanceType = HOLDPOSITION;
					pathplanner_active = false;

					// Load pregenerated hold-position program
					simple_hold_position();
				}
				break;
			case FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER:
				if (guidanceType != PATHPLANNER) {
					guidanceType = PATHPLANNER;
					pathplanner_active = false;

					// Load pregenerated example program
					example_program();
				}
				break;
			default:
				// When not running the path manager, short circuit and wait
				pathplanner_active = false;
				guidanceType = NOMANAGER;
				vTaskDelay(IDLE_UPDATE_RATE_MS * portTICK_RATE_MS);

				continue;
		}
#else
		PathPlannerStatusData pathPlannerStatus;
		PathPlannerStatusGet(&pathPlannerStatus);

		if (pathPlannerStatus.PathAvailability == PATHPLANNERSTATUS_PATHAVAILABILITY_PATHREADY)
		{
			if (guidanceType != PATHPLANNER) {
				guidanceType = PATHPLANNER;
				pathplanner_active = false;
			}
		}
		else{
			pathplanner_active = false;
			guidanceType = NOMANAGER;
			vTaskDelay(IDLE_UPDATE_RATE_MS * portTICK_RATE_MS);
			continue;
		}
#endif //PATH_PLANNER

		bool advanceSegment_flag = false;

		// Update arc measure traveled
		if (pathSegmentDescriptor_current.PathCurvature != 0) {
			PositionActualData positionActual;
			PositionActualGet(&positionActual);
			float newPosition_NE[2] = {positionActual.North, positionActual.East};
			if (arc_has_center == CENTER_FOUND) {
				angularDistanceCompleted_D  += measure_arc_rad(oldPosition_NE, newPosition_NE, arcCenter_NE) * RAD2DEG;

				oldPosition_NE[0] = newPosition_NE[0];
				oldPosition_NE[1] = newPosition_NE[1];

				// Every 128 samples, correct for roundoff error. Error doesn't accumulate too quickly, so
				// this trigger value can safely be made much higher, with the condition that the type of
				// theta_roundoff_trim_count be changed from uint8_t;
				if ((theta_roundoff_trim_count++ & 0x8F) == 0) {
					theta_roundoff_trim_count = 0;

					float referenceTheta_D = measure_arc_rad(previousLocus->Position, newPosition_NE, arcCenter_NE) * RAD2DEG;
					float error_D = circular_modulus_deg(referenceTheta_D-angularDistanceCompleted_D);

					angularDistanceCompleted_D += error_D;
				}
			}

		}

		// If the vehicle is sufficiently close to the goal, check if it has achieved the goal
		// of the active path segment. Sufficiently close is chosen to be an arbitrary angular
		// distance, as this is robust and sufficient to describe all paths, including infinite
		// straight lines and infinite number of orbits about a point.
		if (SIGN(pathSegmentDescriptor_current.PathCurvature) * (angularDistanceToComplete_D - angularDistanceCompleted_D) < ANGULAR_PROXIMITY_THRESHOLD)
			advanceSegment_flag = checkGoalCondition();


		// Check if the path_manager was just activated
		if (pathplanner_active == false) {
			// Update path manager
			pathManagerStatus.ActiveSegment = 0; // This will get immediately incremented to 1 by advanceSegment()
			pathManagerStatus.PathCounter++; // Incrementing this tells the path follower that there is a new path
			pathManagerStatus.Status = PATHMANAGERSTATUS_STATUS_INPROGRESS;
			PathManagerStatusSet(&pathManagerStatus);

			advanceSegment_flag = true;
			pathplanner_active = true;

			// Reset timer
			segmentTimer = xTaskGetTickCount();
		}


		// Advance segment
		if (advanceSegment_flag) {
			advanceSegment();
		}
		else if (lastSysTime-segmentTimer > pathManagerStatus.Timeout*1000*portTICK_RATE_MS)
		{	// Check if we have timed out
			// TODO: Handle the buffer overflow in xTaskGetTickCount
			pathManagerStatus.Status = PATHMANAGERSTATUS_STATUS_TIMEDOUT;
			PathManagerStatusSet(&pathManagerStatus);
		}
		else if (lastSysTime-overshootTimer > OVERSHOOT_TIMER_MS*portTICK_RATE_MS)
		{	// Once every second or so, check for higher-level path planner failure
			checkOvershoot();
			overshootTimer = lastSysTime;
		}
	}
}


/**
 * @brief advanceSegment FIXME: Currently, this will read as many PathSegmentDescriptor instances as there are, not as many
 * as exist in the program. This condition occurs when a long program is replaced by a short one.
 */
static void advanceSegment()
{
	PathSegmentDescriptorData pathSegmentDescriptor_past;
	PathSegmentDescriptorInstGet(pathManagerStatus.ActiveSegment, &pathSegmentDescriptor_past);

	previousLocus->Position[0] = pathSegmentDescriptor_past.SwitchingLocus[0];
	previousLocus->Position[1] = pathSegmentDescriptor_past.SwitchingLocus[1];
	previousLocus->Position[2] = pathSegmentDescriptor_past.SwitchingLocus[2];
	previousLocus->Velocity = pathSegmentDescriptor_past.FinalVelocity;

	// Advance segment
	pathManagerStatus.ActiveSegment++;
	pathManagerStatus.Status = PATHMANAGERSTATUS_STATUS_INPROGRESS;
	PathManagerStatusSet(&pathManagerStatus);

	// Load current segment into global memory.
	PathSegmentDescriptorInstGet(pathManagerStatus.ActiveSegment, &pathSegmentDescriptor_current);  // TODO: Check that an instance is successfully returned

	// Reset angular distance
	angularDistanceCompleted_D = 0;

	// If the path is an arc, find the center and angular distance along arc
	if (pathSegmentDescriptor_current.PathCurvature != 0 ) {
		// Determine if the arc has a center, and if so assign it to arcCenter_NE
		arc_has_center = find_arc_center(previousLocus->Position, pathSegmentDescriptor_current.SwitchingLocus,
										 1.0f/pathSegmentDescriptor_current.PathCurvature,
										 pathSegmentDescriptor_current.PathCurvature > 0,
										 pathSegmentDescriptor_current.ArcRank == PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR,
										 arcCenter_NE);

		// If the arc has a center, then set the initial position as the beginning of the arc, and calculate the angular
		// distance to be traveled along the arc
		switch (arc_has_center) {
			case CENTER_FOUND: 
			{
				oldPosition_NE[0] = previousLocus->Position[0];
				oldPosition_NE[1] = previousLocus->Position[1];

				float tmpAngle_D = measure_arc_rad(previousLocus->Position, pathSegmentDescriptor_current.SwitchingLocus, arcCenter_NE) * RAD2DEG;
				if (SIGN(pathSegmentDescriptor_current.PathCurvature) * tmpAngle_D < 0)
				{
					tmpAngle_D = tmpAngle_D	+ 360*SIGN(pathSegmentDescriptor_current.PathCurvature);
				}
				angularDistanceToComplete_D = SIGN(pathSegmentDescriptor_current.PathCurvature) * pathSegmentDescriptor_current.NumberOfOrbits*360 + tmpAngle_D;
			}
				break;
			default:
				// This is really bad, and is only possible if the path planner screws up, but we need to handle these cases nonetheless because
				// the alternative might be to crash. The simplest way tof fix the problem is to increase the radius, but we can't do this
				// because it is forbidden for two modules to write one UAVO.
				angularDistanceToComplete_D = 0;
				break;
		}
	}
	else{
		angularDistanceToComplete_D = 0;
	}

	// Calculate timout. This is where winds aloft should be taken into account
	float s;
	if (pathSegmentDescriptor_current.PathCurvature == 0) { // Straight line
		s = sqrtf(powf(pathSegmentDescriptor_current.SwitchingLocus[0] - pathSegmentDescriptor_past.SwitchingLocus[0],2) +
				powf(pathSegmentDescriptor_current.SwitchingLocus[1] - pathSegmentDescriptor_past.SwitchingLocus[1],2));
	}
	else // Arc
		s = angularDistanceToComplete_D * DEG2RAD / pathSegmentDescriptor_current.PathCurvature;

	if (pathSegmentDescriptor_current.FinalVelocity > 0)
		pathManagerStatus.Timeout = bound_min_max(ceilf(fabsf(s)/((float)pathSegmentDescriptor_current.FinalVelocity)), 0, 65535);
	else
		pathManagerStatus.Timeout = 65535; // Set this to maximum possible value for variable type


	PathManagerStatusSet(&pathManagerStatus);

	// Reset timer
	segmentTimer = xTaskGetTickCount();
}

// This is not a strict end to the segment, as some amount of error will always
// creep in. Instead, come within either a preset distance or a preset time of
// the goal condition.
static bool checkGoalCondition()
{
	bool advanceSegment_flag = false;

	// Half-plane approach
	// From R. Beard and T. McLain, "Small Unmanned Aircraft: Theory and Practice", 2011, Section 11.1.
	// Note: The half-plane approach has difficulties when the plane and the two loci are close to colinear
	// and reversing in direction. That is to say, a plane at P is supposed to go to A and then B:
	//    B----------P------A
	switch (pathManagerSettings.SwitchingStrategy) {
		case PATHMANAGERSETTINGS_SWITCHINGSTRATEGY_HALFPLANE:
		{
			// Check if there is a switching locus after the present one
			if (pathManagerStatus.ActiveSegment + 1 < UAVObjGetNumInstances(PathSegmentDescriptorHandle())) {
				// Calculate vector from past to preset switching locus
				float *swl_past = previousLocus->Position;
				float *swl_current = pathSegmentDescriptor_current.SwitchingLocus;
				float q_current[3] = {swl_current[0] - swl_past[0], swl_current[1] - swl_past[1], 0};
				float q_current_mag = VectorMagnitude(q_current); //Normalize
				float q_future[3];
				float q_future_mag;

				PathSegmentDescriptorData pathSegmentDescriptor_future;
				PathSegmentDescriptorInstGet(pathManagerStatus.ActiveSegment+1, &pathSegmentDescriptor_future);  // TODO: Check that an instance is successfully returned

				// Line-line intersection. The halfplane frontier is the bisecting line between the arrival
				// and departure vector.
				if (pathSegmentDescriptor_current.PathCurvature == 0 && pathSegmentDescriptor_future.PathCurvature == 0) {
					float *swl_future  = pathSegmentDescriptor_future.SwitchingLocus;

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
				else if (pathSegmentDescriptor_current.PathCurvature == 0 && pathSegmentDescriptor_future.PathCurvature != 0)
				{
					// Calculate vector tangent to arc at preset switching locus. This comes from geometry that that tangent to a circle
					// is the perpendicular vector to the vector connecting the tangent point and the center of the circle. The vector R
					// is tangent_point - arc_center, so the perpendicular to R is <-lambda*Ry,lambda*Rx>, where lambda = +-1.

					bool clockwise = pathSegmentDescriptor_future.PathCurvature > 0;
					bool minor = pathSegmentDescriptor_future.ArcRank == PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
					int8_t lambda;

					if ((clockwise == true && minor == true) ||
							(clockwise == false && minor == false)) { //clockwise minor OR counterclockwise major
						lambda = 1;
					} else { //counterclockwise minor OR clockwise major
						lambda = -1;
					}

					// Vector perpendicular to the vector from arc center to tangent point
					q_future[0] = -lambda*(swl_current[1] - arcCenter_NE[1]);
					q_future[1] = lambda*(swl_current[0] - arcCenter_NE[0]);
					q_future[2] = 0;
					q_future_mag = VectorMagnitude(q_future); //Normalize
				}
				// "Small Unmanned Aircraft: Theory and Practice" provides no guidance for the perpendicular
				// intersection of an arc and a line. However, it seems reasonable to consider the halfplane
				// occurring at the intersection between the arc and the vector. The halfplane's frontier
				// is perpendicular to the tangent at the end of the arc.
				else if (pathSegmentDescriptor_current.PathCurvature != 0 && pathSegmentDescriptor_future.PathCurvature == 0)
				{
						// Cheat by remarking that the plane defined by the radius is perfectly defined by the angle made
						// between the center and the end of the trajectory. So if the vehicle has traveled further than
						// the required angular distance, it has crossed this
						if (SIGN(pathSegmentDescriptor_current.PathCurvature) * (angularDistanceCompleted_D - angularDistanceToComplete_D) >= 0)
							advanceSegment_flag = true;

						return advanceSegment_flag;
					}
					// "Small Unmanned Aircraft: Theory and Practice" provides no guidance for the perpendicular
					// intersection of two arcs. However, it seems reasonable to consider the halfplane
					// occurring at the intersection between the two arcs. The halfplane's frontier is defined
					// as the half-angle between the arriving arc tangent and the departing arc tangent, similar to
					// the line-line case.
					else if (pathSegmentDescriptor_current.PathCurvature != 0 && pathSegmentDescriptor_future.PathCurvature != 0)
					{
						// Cheat by remarking that the plane defined by the radius is perfectly defined by the angle made
						// between the center and the end of the trajectory. So if the vehicle has traveled further than
						// the required angular distance, it has crossed this
						if (SIGN(pathSegmentDescriptor_current.PathCurvature) * (angularDistanceCompleted_D - angularDistanceToComplete_D) >= 0)
							advanceSegment_flag = true;

						return advanceSegment_flag;
				}
				else{
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
				float halfPlane[3] = {q_future[0]*q_current_mag + q_current[0]*q_future_mag,
								q_future[1]*q_current_mag + q_current[1]*q_future_mag,
								q_future[2]*q_current_mag + q_current[2]*q_future_mag};

				// Test if the UAV is in the half plane, H. This is easy by taking advantage of simple vector
				// calculus: a.b = |a||b|cos(theta), but since |a|,|b| >=0, then a.b > 0 if and only if
				// cos(theta) > 0, which means that the UAV is in I or IV quadrants, i.e. is in the half plane.
				PositionActualData positionActual;
				PositionActualGet(&positionActual);
				float p[2] = {positionActual.North - swl_current[0], positionActual.East - swl_current[1]};

				// If we want to switch based on nominal time to locus, add the normalized q_current times the speed times the timing advace
				if (pathManagerSettings.HalfPlaneAdvanceTiming != 0) {
					for (int i=0; i<2; i++) {
							p[i] += q_current[i]/q_current_mag * fixedWingAirspeeds.BestClimbRateSpeed * (pathManagerSettings.HalfPlaneAdvanceTiming/1000.0f);
					}
				}

				// Finally test a.b > 0
				if (p[0]*halfPlane[0] + p[1]*halfPlane[1] > 0) {
					advanceSegment_flag = true;
				}
			}
			else{ // Since there are no further switching loci, this must be the waypoint.
				//Do nothing.
			}
		}
		break;
		// B-ball approach. This tests if the vehicle is within a threshold distance.
		// From R. Beard and T. McLain, "Small Unmanned Aircraft: Theory and Practice", 2011, Section 11.1.
		case PATHMANAGERSETTINGS_SWITCHINGSTRATEGY_BBALL:
		{
			// This method is less robust to error than the half-plane. It is cheaper and simpler, but those are it's only two advantages
			PositionActualData positionActual;
			PositionActualGet(&positionActual);
			float d[3] = {positionActual.North - pathSegmentDescriptor_current.SwitchingLocus[0], positionActual.East - pathSegmentDescriptor_current.SwitchingLocus[1], 0};

			if (VectorMagnitude(d) < pathManagerSettings.BBallThresholdDistance)
			{
				advanceSegment_flag = true;
			}
		}
		break;
	default:
		// TODO: This is bad to get here. Make sure it's not possible.
		break;
	}

	return advanceSegment_flag;
}

//Check to see if we've seriously overflown our destination. Since the path follower is simply following a
//motion descriptor,it has no concept of where the path ends. It will simply keep following it to infinity
//if we don't stop it.
//So while we don't know why the navigation manager failed, we know we don't want the plane flying off.
static void checkOvershoot()
{
	// TODO: Check for overshoot with non-infinite arcs, too.
	if (pathSegmentDescriptor_current.PathCurvature == 0) {
		PositionActualData positionActual;
		PositionActualGet(&positionActual);

		float p[2] = {positionActual.North, positionActual.East};
		float c[2] = {pathSegmentDescriptor_current.SwitchingLocus[0], pathSegmentDescriptor_current.SwitchingLocus[1]};
		float *r = previousLocus->Position;

		// Calculate vector from initial to final point
		float q[3] = {c[0] - r[0], c[1] - r[1], 0};
		float q_mag = VectorMagnitude(q); //Normalize

		// Add in a distance equal to 5s of flight time for good measure, in to make sure we don't have any jitter.
		for (int i=0; i < 2; i++)
			c[i] += q[i]/q_mag * fixedWingAirspeeds.BestClimbRateSpeed * 5;

		// Perform a quick vector dot product to test if we've gone past the waypoint.
		if ((p[0]-c[0])*q[0] + (p[1]-c[1])*q[1] > 0) {
			//Whoops, we've really overflown our destination point, and haven't received any instructions.

			//Inform the FSM
			pathManagerStatus.Status = PATHMANAGERSTATUS_STATUS_OVERSHOOT;
			PathManagerStatusSet(&pathManagerStatus);

			//TODO: Declare an alarm
			AlarmsSet(SYSTEMALARMS_ALARM_PATHMANAGER, SYSTEMALARMS_ALARM_CRITICAL);
			//TODO: Start circling
		}
	}
}



static void settingsUpdated(UAVObjEvent * ev) {
	if (ev == NULL || ev->obj == PathManagerSettingsHandle()) {
		PathManagerSettingsGet(&pathManagerSettings);
	}
	if (ev == NULL || ev->obj == FixedWingAirspeedsHandle()) {
		FixedWingAirspeedsGet(&fixedWingAirspeeds);
	}
}


/**
 * @}
 * @}
 */
