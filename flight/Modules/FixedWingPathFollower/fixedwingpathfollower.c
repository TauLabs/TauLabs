/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup FixedWingPathFollower Fixed wing path follower module
 * @{
 *
 * @file       fixedwingpathfollower.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      This module compared @ref PositionActual to @ref PathDesired
 * and sets @ref StabilizationDesired.  It only does this when the FlightMode field
 * of @ref ManualControlCommand is Auto.
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
#include "openpilot.h"
#include "physical_constants.h"
#include "paths.h"
#include "misc_math.h"
#include "atmospheric_math.h"
#include "path_followers.h"

#include "airspeedactual.h"
#include "attitudeactual.h"
#include "fixedwingairspeeds.h"
#include "fixedwingpathfollowersettings.h"
#include "flightstatus.h"
#include "homelocation.h"
#include "modulesettings.h"
#include "positionactual.h"
#include "stabilizationdesired.h" // object that will be updated by the module
#include "stabilizationsettings.h"
#include "systemsettings.h"
#include "velocityactual.h"

#include "coordinate_conversions.h"

#include "pathsegmentdescriptor.h"
#include "pathfollowerstatus.h"
#include "pathmanagerstatus.h"


// Private constants
#define MAX_QUEUE_SIZE 4
#define STACK_SIZE_BYTES 850
#define TASK_PRIORITY (tskIDLE_PRIORITY+2)

// Private types
struct ControllerOutput {
	float roll;
	float pitch;
	float yaw;
	float throttle;
};

struct ErrorIntegral {
	float total_energy_error;
	float calibrated_airspeed_error;
	float line_error;
	float arc_error;
};

// Private variables
static bool module_enabled = false;
static xTaskHandle pathfollowerTaskHandle;
static bool flightStatusUpdate = false;
static uint8_t flightMode = FLIGHTSTATUS_FLIGHTMODE_MANUAL;

static PathDesiredData *pathDesired;
static PathSegmentDescriptorData *pathSegmentDescriptor;
static FixedWingPathFollowerSettingsData fixedwingpathfollowerSettings;
static FixedWingAirspeedsData fixedWingAirspeeds;
static struct ErrorIntegral *integral;
static uint16_t activeSegment;
static uint8_t pathCounter;
static float arc_radius;
static xQueueHandle pathManagerStatusQueue;

// Private functions
static void pathfollowerTask(void *parameters);
static void FlightStatusUpdatedCb(UAVObjEvent * ev);

static void SettingsUpdatedCb(UAVObjEvent * ev);
static void updateDestination(void);
static int8_t updateFixedWingDesiredStabilization(void);

static void airspeedController(struct ControllerOutput *airspeedControl, float calibrated_airspeed_error, float altitudeError, float dT);
static void totalEnergyController(struct ControllerOutput *energyControl, float true_airspeed_desired,
						   float true_airspeed_actual, float altitude_desired_NED, float altitude_actual_NED, float dT);
static void simple_heading_controller(struct ControllerOutput *headingControl, PositionActualData *positionActual, float curvature, float courseActual_R, float true_airspeed_desired, float dT);
static void roll_constrained_heading_controller(struct ControllerOutput *headingControl, float headingActual_R, PositionActualData *positionActual, VelocityActualData *velocityActual, float curvature, float true_airspeed, float true_airspeed_desired);


/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t FixedWingPathFollowerStart()
{
	if (module_enabled) {
		// Start main task
		xTaskCreate(pathfollowerTask, (signed char *)"PathFollower", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &pathfollowerTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_PATHFOLLOWER, pathfollowerTaskHandle);
	}

	return 0;
}

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t FixedWingPathFollowerInitialize()
{
#ifdef MODULE_FixedWingPathFollower_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_FIXEDWINGPATHFOLLOWER] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (!module_enabled)
		return false;

	// Initialize UAVOs necessary for all pathfinders
	PathFollowerStatusInitialize();

	// Initialize UAVO for fixed-wing flight
	FixedWingPathFollowerSettingsInitialize();
	FixedWingAirspeedsInitialize();


#if defined(PATHDESIRED_DIAGNOSTICS)
	PathDesiredInitialize();
#endif


	// Register callbacks
	FlightStatusConnectCallback(FlightStatusUpdatedCb);
	FixedWingAirspeedsConnectCallback(SettingsUpdatedCb);
	FixedWingPathFollowerSettingsConnectCallback(SettingsUpdatedCb);

	// Register queues
	pathManagerStatusQueue = xQueueCreate(1, sizeof(UAVObjEvent));
	PathManagerStatusConnectQueue(pathManagerStatusQueue);

	// Allocate memory
	integral = (struct ErrorIntegral *) pvPortMalloc(sizeof(struct ErrorIntegral));
	memset(integral, 0, sizeof(struct ErrorIntegral));

	pathSegmentDescriptor = (PathSegmentDescriptorData *) pvPortMalloc(sizeof(PathSegmentDescriptorData));
	memset(pathSegmentDescriptor, 0, sizeof(PathSegmentDescriptorData));

	pathDesired = (PathDesiredData *) pvPortMalloc(sizeof(PathDesiredData));
	memset(pathDesired, 0, sizeof(PathDesiredData));

	// Load all settings
	SettingsUpdatedCb((UAVObjEvent *)NULL);

	// Zero the integrals
	integral->total_energy_error = 0;
	integral->calibrated_airspeed_error = 0;
	integral->line_error = 0;
	integral->arc_error = 0;


	return 0;
}
MODULE_INITCALL(FixedWingPathFollowerInitialize, FixedWingPathFollowerStart);


/**
 * Module thread, should not return.
 */
static void pathfollowerTask(void *parameters)
{
	portTickType lastUpdateTime;
	PathFollowerStatusData pathFollowerStatus;
	uint16_t update_peroid_ms;

	FixedWingPathFollowerSettingsConnectCallback(SettingsUpdatedCb);
	FixedWingAirspeedsConnectCallback(SettingsUpdatedCb);
	PathDesiredConnectCallback(SettingsUpdatedCb);

	// Force update of all the settings
	SettingsUpdatedCb(NULL);

	FixedWingPathFollowerSettingsUpdatePeriodGet(&update_peroid_ms);

	// Main task loop
	lastUpdateTime = xTaskGetTickCount();
	while (1) {
		if (flightStatusUpdate)
			FlightStatusFlightModeGet(&flightMode);

		if(flightMode != FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME &&
				flightMode != FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD &&
				flightMode != FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER) {

			// Be clean and reset integrals
			integral->total_energy_error = 0;
			integral->calibrated_airspeed_error = 0;
			integral->line_error = 0;
			integral->arc_error = 0;

			PathFollowerStatusGet(&pathFollowerStatus);
			pathFollowerStatus.Status = PATHFOLLOWERSTATUS_STATUS_IDLE;
			PathFollowerStatusSet(&pathFollowerStatus);

			// Wait 100ms before continuing
			vTaskDelay(100 * portTICK_RATE_MS);
			continue;
		} else if (pathFollowerStatus.Status == PATHFOLLOWERSTATUS_STATUS_IDLE) {
			PathFollowerStatusGet(&pathFollowerStatus);
			pathFollowerStatus.Status = PATHFOLLOWERSTATUS_STATUS_SUCCEEDING;
			PathFollowerStatusSet(&pathFollowerStatus);
		}

		vTaskDelayUntil(&lastUpdateTime, update_peroid_ms * portTICK_RATE_MS);
		updateFixedWingDesiredStabilization();
	}
}


/**
 * @brief
 */
int8_t updateFixedWingDesiredStabilization(void)
{
	// Check if the path manager has updated.
	UAVObjEvent ev;
	if (xQueueReceive(pathManagerStatusQueue, &ev, 0) == pdTRUE)
	{
		PathManagerStatusData pathManagerStatusData;
		PathManagerStatusGet(&pathManagerStatusData);

		// Fixme: This isn't a very elegant check. Since the manager can update it's state with new loci, but
		// still have the original ActiveSegment, the pathcounter was introduced. It only changes when the
		// path manager gets a new path. Since this pathcounter variable doesn't do anything else, it's a bit
		// of a waste of space right now. Logically, the path planner should set this variable but since we
		// can't be sure a path planner is running, it works better on the level of the path manager.
		if (activeSegment != pathManagerStatusData.ActiveSegment || pathCounter != pathManagerStatusData.PathCounter)
		{
			activeSegment = pathManagerStatusData.ActiveSegment;
			pathCounter = pathManagerStatusData.PathCounter;

			updateDestination();
		}
	}

	float dT = fixedwingpathfollowerSettings.UpdatePeriod / 1000.0f; //Convert from [ms] to [s]

	VelocityActualData velocityActual;
	StabilizationDesiredData stabilizationDesired;
	float true_airspeed;
	float calibrated_airspeed;

	float true_airspeed_desired;
	float calibrated_airspeed_desired;
	float calibrated_airspeed_error;

	float altitudeDesired_NED;
	float altitudeError_NED;

	VelocityActualGet(&velocityActual);
	StabilizationDesiredGet(&stabilizationDesired);

	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	// Current airspeed
	AirspeedActualTrueAirspeedGet(&true_airspeed);
	AirspeedActualCalibratedAirspeedGet(&calibrated_airspeed);

	// Current course
	float courseActual_R = atan2f(velocityActual.East, velocityActual.North);

	// Current heading
	float headingActual_D;
	AttitudeActualYawGet(&headingActual_D);
	float headingActual_R = headingActual_D * DEG2RAD;

	/**
	 * Compute setpoints.
	 */
	/**
	 * TODO: These setpoints only need to be set once per locus update
	 */
	// Set the desired altitude
	altitudeDesired_NED = pathDesired->End[2];

	// Set desired calibrated airspeed, bounded by airframe limits
	calibrated_airspeed_desired = bound_min_max(pathDesired->EndingVelocity, fixedWingAirspeeds.StallSpeedDirty, fixedWingAirspeeds.AirSpeedMax);

	// Set the desired true airspeed, assuming STP atmospheric conditions. This isn't ideal, but we don't have a reliable source of temperature or pressure
	struct AirParameters air_STP = initialize_air_structure();
	true_airspeed_desired = cas2tas(calibrated_airspeed_desired, -positionActual.Down, &air_STP);

	/**
	 * Compute setpoint errors
	 */
	// Airspeed error
	calibrated_airspeed_error = calibrated_airspeed_desired - calibrated_airspeed;

	// Altitude error
	altitudeError_NED = altitudeDesired_NED - positionActual.Down;

	/**
	 * Compute controls
	 */
	// Compute airspeed control
	struct ControllerOutput airspeedControl;
	airspeedController(&airspeedControl, calibrated_airspeed_error, altitudeError_NED, dT);

	// Compute altitude control
	struct ControllerOutput totalEnergyControl;
	totalEnergyController(&totalEnergyControl, true_airspeed_desired, true_airspeed, altitudeDesired_NED, positionActual.Down, dT);

	// Compute heading control
	struct ControllerOutput headingControl;
	switch (fixedwingpathfollowerSettings.FollowerAlgorithm) {
	case FIXEDWINGPATHFOLLOWERSETTINGS_FOLLOWERALGORITHM_ROLLLIMITED:
		roll_constrained_heading_controller(&headingControl, headingActual_R, &positionActual, &velocityActual, pathDesired->Curvature, true_airspeed, true_airspeed_desired);
		break;
	case FIXEDWINGPATHFOLLOWERSETTINGS_FOLLOWERALGORITHM_SIMPLE:
	default:
		simple_heading_controller(&headingControl, &positionActual, pathDesired->Curvature, courseActual_R, true_airspeed_desired, dT);
		break;
	}


	// Sum all controllers
	stabilizationDesired.Throttle = bound_min_max(headingControl.throttle + airspeedControl.throttle + totalEnergyControl.throttle,
												  fixedwingpathfollowerSettings.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLELIMIT_MIN],
												  fixedwingpathfollowerSettings.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLELIMIT_MAX]);
	stabilizationDesired.Roll     = bound_sym(headingControl.roll + airspeedControl.roll + totalEnergyControl.roll,
												  fixedwingpathfollowerSettings.RollLimit);
	stabilizationDesired.Pitch    = bound_min_max(headingControl.pitch + airspeedControl.pitch + totalEnergyControl.pitch,
												  fixedwingpathfollowerSettings.PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGS_PITCHLIMIT_MIN],
												  fixedwingpathfollowerSettings.PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGS_PITCHLIMIT_MAX]);
	stabilizationDesired.Yaw      = headingControl.yaw + airspeedControl.yaw + totalEnergyControl.yaw; // Coordinated flight control only works when stabilizationDesired.Yaw == 0

	// Set stabilization modes
	stabilizationDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE; //This needs to be EnhancedAttitude control
	stabilizationDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE; //This needs to be EnhancedAttitude control
	stabilizationDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_COORDINATEDFLIGHT;

	StabilizationDesiredSet(&stabilizationDesired);

	return 0;
}


static void airspeedController(struct ControllerOutput *airspeedControl, float calibrated_airspeed_error, float altitudeError_NED, float dT)
{
	// This is the throttle value required for level flight at the given airspeed
	float feedForwardThrottle = fixedwingpathfollowerSettings.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLELIMIT_NEUTRAL];


	/**
	 * Compute desired pitch command
	 */

#define AIRSPEED_KP      fixedwingpathfollowerSettings.AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_AIRSPEEDPI_KP]
#define AIRSPEED_KI      fixedwingpathfollowerSettings.AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_AIRSPEEDPI_KI]
#define AIRSPEED_ILIMIT	 fixedwingpathfollowerSettings.AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_AIRSPEEDPI_ILIMIT]

	if (AIRSPEED_KI > 0.0f) {
		//Integrate with saturation
		integral->calibrated_airspeed_error=bound_sym(integral->calibrated_airspeed_error + calibrated_airspeed_error * dT,
													  AIRSPEED_ILIMIT/AIRSPEED_KI);
	}

	//Compute the cross feed from altitude to pitch, with saturation
#define PITCHCROSSFEED_KP fixedwingpathfollowerSettings.AltitudeErrorToPitchCrossFeed[FIXEDWINGPATHFOLLOWERSETTINGS_ALTITUDEERRORTOPITCHCROSSFEED_KP]
#define PITCHCROSSFEED_MIN	fixedwingpathfollowerSettings.AltitudeErrorToPitchCrossFeed[FIXEDWINGPATHFOLLOWERSETTINGS_ALTITUDEERRORTOPITCHCROSSFEED_KP]
#define PITCHCROSSFEED_MAX fixedwingpathfollowerSettings.AltitudeErrorToPitchCrossFeed[FIXEDWINGPATHFOLLOWERSETTINGS_ALTITUDEERRORTOPITCHCROSSFEED_KP]
	float altitudeErrorToPitchCommandComponent=bound_min_max(altitudeError_NED* PITCHCROSSFEED_KP, -PITCHCROSSFEED_MIN, PITCHCROSSFEED_MAX);

	//Saturate pitch command
#define PITCHLIMIT_NEUTRAL  fixedwingpathfollowerSettings.PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGS_PITCHLIMIT_NEUTRAL]

	// Assign airspeed controller outputs
	airspeedControl->throttle = feedForwardThrottle;
	airspeedControl->roll = 0;
	airspeedControl->pitch = -(calibrated_airspeed_error*AIRSPEED_KP + integral->calibrated_airspeed_error*AIRSPEED_KI) + altitudeErrorToPitchCommandComponent + PITCHLIMIT_NEUTRAL; //TODO: This needs to be taken out once EnhancedAttitude is merged
	airspeedControl->yaw = 0;
}


/**
 * @brief totalEnergyController This controller uses a PID to stabilize the total kinetic and potential energy.
 * @param energyControl Output structure
 * @param true_airspeed_desired True airspeed setpoint (Use true airspeed for kinetic energy)
 * @param true_airspeed_actual True airspeed estimation
 * @param altitude_desired_NED Down altitude desired (-height)
 * @param altitude_actual_NED Current Down estimation
 * @param dT
 */
static void totalEnergyController(struct ControllerOutput *energyControl, float true_airspeed_desired, float true_airspeed_actual, float altitude_desired_NED, float altitude_actual_NED, float dT)
{
	//Proxy because instead of m*(1/2*v^2+g*h), it's v^2+2*gh. This saves processing power
	float totalEnergyProxySetpoint=powf(true_airspeed_desired, 2.0f) - 2.0f*9.8f*altitude_desired_NED;
	float totalEnergyProxyActual=powf(true_airspeed_actual, 2.0f) - 2.0f*9.8f*altitude_actual_NED;
	float errorTotalEnergy= totalEnergyProxySetpoint - totalEnergyProxyActual;

#define THROTTLE_KP fixedwingpathfollowerSettings.ThrottlePI[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLEPI_KP]
#define THROTTLE_KI fixedwingpathfollowerSettings.ThrottlePI[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLEPI_KI]
#define THROTTLE_ILIMIT fixedwingpathfollowerSettings.ThrottlePI[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLEPI_ILIMIT]

	//Integrate with bound. Make integral leaky for better performance. Approximately 30s time constant.
	if (THROTTLE_KI > 0.0f) {
		integral->total_energy_error=bound_sym(integral->total_energy_error+errorTotalEnergy*dT,
											   THROTTLE_ILIMIT/THROTTLE_KI)*(1.0f-1.0f/(1.0f+30.0f/dT));
	}

	// Assign altitude controller outputs
	energyControl->throttle = errorTotalEnergy*THROTTLE_KP + integral->total_energy_error*THROTTLE_KI;
	energyControl->roll = 0;
	energyControl->pitch = 0;
	energyControl->yaw = 0;
}


/**
 * @brief simple_heading_controller This calculates the heading setpoint as a function of a vector field directed onto a path. This is
 * based off of research into Lyanpov controllers, performed by R. Beard at Bringham Young University, Utah, USA.
 * @param[out] headingControl Output structure
 * @param[in] positionActual
 * @param[in] curvature
 * @param[in] courseActual_R
 * @param[in] true_airspeed_desired
 * @param[in] dT
 */
static void simple_heading_controller(struct ControllerOutput *headingControl, PositionActualData *positionActual, float curvature, float courseActual_R, float true_airspeed_desired, float dT)
{
	float k_psi_int = fixedwingpathfollowerSettings.FollowerIntegralGain;

	float courseDesired_R;

	if (curvature == 0) { // Straight line has no curvature
		//========================================
		//SHOULD NOT BE HARD CODED

		float chi_inf = PI/4.0f; //Fixme: This should be a function of how long the path is

		//Saturate chi_inf, i.e., never allow an approach path steeper angle than 90 degrees
		chi_inf = fabsf(chi_inf) > PI/2.0f ? PI/2.0f : fabsf(chi_inf);
		//========================================

		float k_path  = fixedwingpathfollowerSettings.VectorFollowingGain / true_airspeed_desired; //Divide gain by airspeed so that the vector field scales with airspeed
		courseDesired_R = simple_line_follower(positionActual, pathDesired, chi_inf, k_path, k_psi_int, &(integral->line_error), dT);
	} else {
		float k_orbit = fixedwingpathfollowerSettings.OrbitFollowingGain/true_airspeed_desired;  //Divide gain by airspeed so that the vector field scales with airspeed
		courseDesired_R = simple_arc_follower(positionActual, pathDesired->End, arc_radius, SIGN(curvature), k_orbit, k_psi_int, &(integral->arc_error), dT);
	}

	// Course error, wrapped to [-pi,pi]
	float courseError_R = circular_modulus_rad(courseDesired_R - courseActual_R);

	// Assign heading controller outputs
	headingControl->throttle = 0;
	headingControl->roll = (courseError_R * fixedwingpathfollowerSettings.HeadingPI[FIXEDWINGPATHFOLLOWERSETTINGS_HEADINGPI_KP]) * RAD2DEG;
	headingControl->pitch = 0;
	headingControl->yaw = 0;
}


/**
 * @brief roll_constrained_heading_controller This calculates the heading setpoint as a function of a vector field directed
 * onto a path, and taking into account the UAVs roll limits. This is based off of research into Lyanpov controllers,
 * performed by R. Beard at Bringham Young University, Utah, USA.
 * @param[out] headingControl Output structure
 * @param[in] headingActual_R
 * @param[in] positionActual
 * @param[in] velocityActual
 * @param[in] curvature
 * @param[in] true_airspeed
 * @param[in] true_airspeed_desired
 */
static void roll_constrained_heading_controller(struct ControllerOutput *headingControl, float headingActual_R,
								 PositionActualData *positionActual, VelocityActualData *velocityActual,
								 float curvature, float true_airspeed, float true_airspeed_desired)
{
	float phi_max = fixedwingpathfollowerSettings.RollLimit * DEG2RAD;
	float gamma_max = fixedwingpathfollowerSettings.PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGS_PITCHLIMIT_MAX] * DEG2RAD; // Flight path angle is in fact only loosely coupled to pitch

	// Calculate roll command
	float roll_c_R;

	if (curvature == 0) { // Straight line has no curvature
		roll_c_R = roll_limited_line_follower(positionActual, velocityActual, pathDesired, true_airspeed,
											  true_airspeed_desired, headingActual_R, gamma_max, phi_max);
	} else { // Curve following
		roll_c_R = roll_limited_arc_follower(positionActual, velocityActual, pathDesired->End, SIGN(pathDesired->Curvature), arc_radius,
										true_airspeed, true_airspeed_desired, headingActual_R, gamma_max, phi_max);
	}

	// Assign heading controller outputs
	headingControl->throttle = 0;
	headingControl->roll = roll_c_R * RAD2DEG;
	headingControl->pitch = 0;
	headingControl->yaw = 0;
}


/**
 * @brief SettingsUpdatedCb Updates settings when relevant UAVO are written
 * @param[in] ev UAVObject event
 */
static void SettingsUpdatedCb(UAVObjEvent * ev)
{
	if (ev == NULL || ev->obj == FixedWingPathFollowerSettingsHandle())
		FixedWingPathFollowerSettingsGet(&fixedwingpathfollowerSettings);
	if (ev == NULL || ev->obj == FixedWingAirspeedsHandle())
		FixedWingAirspeedsGet(&fixedWingAirspeeds);
	if (ev == NULL || ev->obj == PathManagerStatusHandle())
	{

	}
}


/**
 * @brief updateDestination Takes path segment descriptors and writes the path to the PathDesired UAVO
 */
static void updateDestination(void)
{
	PathSegmentDescriptorData pathSegmentDescriptor_old;

	int8_t ret;
	ret = PathSegmentDescriptorInstGet(activeSegment-1, &pathSegmentDescriptor_old);
	if(ret != 0) {
			if (activeSegment == 0) { // This means we're going to the first switching locus.
				PositionActualData positionActual;
				PositionActualGet(&positionActual);

				pathDesired->Start[0]=positionActual.North;
				pathDesired->Start[1]=positionActual.East;
				pathDesired->Start[2]=positionActual.Down;

				// TODO: Figure out if this can't happen in normal behavior. Consider adding a warning if so.
			} else {
			//TODO: Set off a warning

			return;
			}
	} else {
		pathDesired->Start[0]=pathSegmentDescriptor_old.SwitchingLocus[0];
		pathDesired->Start[1]=pathSegmentDescriptor_old.SwitchingLocus[1];
		pathDesired->Start[2]=pathSegmentDescriptor_old.SwitchingLocus[2];
	}

	ret = PathSegmentDescriptorInstGet(activeSegment, pathSegmentDescriptor);
	if(ret != 0) {
			//TODO: Set off a warning

			return;
	}

	// For a straight line use the switching locus as the vector endpoint...
	if(pathSegmentDescriptor->PathCurvature == 0) {
		pathDesired->End[0]=pathSegmentDescriptor->SwitchingLocus[0];
		pathDesired->End[1]=pathSegmentDescriptor->SwitchingLocus[1];
		pathDesired->End[2]=pathSegmentDescriptor->SwitchingLocus[2];

		pathDesired->Curvature = 0;
	} else { // ...but for an arc, use the switching loci to calculate the arc center
		float *oldPosition_NE = pathDesired->Start;
		float *newPosition_NE = pathSegmentDescriptor->SwitchingLocus;
		float arcCenter_NE[2];
		enum arc_center_results ret;

		ret = find_arc_center(oldPosition_NE, newPosition_NE, 1.0f/pathSegmentDescriptor->PathCurvature, pathSegmentDescriptor->PathCurvature > 0, pathSegmentDescriptor->ArcRank == PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR, arcCenter_NE);

		if (ret == ARC_CENTER_FOUND) {
			pathDesired->End[0]=arcCenter_NE[0];
			pathDesired->End[1]=arcCenter_NE[1];
			pathDesired->End[2]=pathSegmentDescriptor->SwitchingLocus[2];
		} else { //---- This is bad, but we have to handle it.----///
			// The path manager should catch this and handle it, but in case it doesn't we'll circle around the midpoint. This
			// way we still maintain positive control, and will satisfy the path requirements, making sure we don't get stuck
			pathDesired->End[0]=(oldPosition_NE[0] + newPosition_NE[0])/2.0f;
			pathDesired->End[1]=(oldPosition_NE[1] + newPosition_NE[1])/2.0f;
			pathDesired->End[2]=pathSegmentDescriptor->SwitchingLocus[2];

			// TODO: Set alarm warning
			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER, SYSTEMALARMS_ALARM_WARNING);
		}

		uint8_t max_avg_roll_D;
		FixedWingPathFollowerSettingsRollLimitGet(&max_avg_roll_D);

		//Calculate arc_radius using r*omega=v and omega = g/V_g*tan(max_avg_roll_D)
		float min_radius = powf(pathSegmentDescriptor->FinalVelocity, 2)/(9.805f*tanf(max_avg_roll_D*DEG2RAD));
		arc_radius = fabsf(1.0f/pathSegmentDescriptor->PathCurvature) > min_radius ? fabsf(1.0f/pathSegmentDescriptor->PathCurvature) : min_radius;
		pathDesired->Curvature = SIGN(pathSegmentDescriptor->PathCurvature) / arc_radius;
	}

	//-------------------------------------------------//
	//FIXME: Inspect pathDesired values for NaN or Inf.//
	//-------------------------------------------------//

	pathDesired->EndingVelocity = pathSegmentDescriptor->FinalVelocity;

#if defined(PATHDESIRED_DIAGNOSTICS)
	PathDesiredSet(pathDesired);
#endif
}

// Triggered by changes in FlightStatus
static void FlightStatusUpdatedCb(UAVObjEvent * ev)
{
	flightStatusUpdate = true;
}


/**
 * @}
 * @}
 */
