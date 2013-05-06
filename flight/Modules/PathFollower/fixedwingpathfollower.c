/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup PathFollowerModule Path Follower Module
 * @{ 
 *
 * @file       fixedwingpathfollower.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      This module compared @ref PositionActual to @ref ActiveWaypoint 
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

/**
 * Input object: @ref PositionActual and @ref PathDesired
 * Output object: @ref StabilizationDesired
 *
 * Computes the @ref StabilizationDesired that gets the UAV back on or keeps
 * the UAV on the requested path
 */

#include "openpilot.h"
#include "physical_constants.h"
#include "paths.h"
#include "misc_math.h"
#include "atmospheric_math.h"

#include "airspeedactual.h"
#include "attitudeactual.h"
#include "fixedwingairspeeds.h"
#include "fixedwingpathfollowersettings.h"
#include "homelocation.h"
#include "manualcontrol.h"
#include "modulesettings.h"
#include "positionactual.h"
#include "stabilizationdesired.h" // object that will be updated by the module
#include "stabilizationsettings.h"
#include "systemsettings.h"
#include "velocityactual.h"

#include "coordinate_conversions.h"

#include "fixedwingpathfollower.h"
#include "pathsegmentdescriptor.h"
#include "pathfollowerstatus.h"
#include "pathmanagerstatus.h"

// Private constants

// Private types
static struct Integral {
	float total_energy_error;
	float calibrated_airspeed_error;

	float line_error;
	float circle_error;
} *integral;


struct ControllerOutput {
	float roll;
	float pitch;
	float yaw;
	float throttle;
};

// Private variables
static PathDesiredData pathDesired;
static PathSegmentDescriptorData *pathSegmentDescriptor;
static FixedWingPathFollowerSettingsData fixedwingpathfollowerSettings;
static FixedWingAirspeedsData fixedWingAirspeeds;
static uint16_t activeSegment;
static uint8_t pathCounter;
static float arc_radius;
static xQueueHandle pathManagerStatusQueue;
static uint8_t max_roll_D = 55;

// Private functions
static void SettingsUpdatedCb(UAVObjEvent * ev);
static void updateDestination();
static float followStraightLine(float r[3], float q[3], float p[3], float chi_inf, float k_path, float k_psi_int, float delT);
static float followOrbit(float c[3], float rho, bool direction, float p[3], float k_orbit, float k_psi_int, float delT);

void airspeedController(struct ControllerOutput *airspeedControl, float calibrated_airspeed_error, float altitudeError, float dT);
void totalEnergyController(struct ControllerOutput *energyControl, float true_airspeed_desired,
						   float true_airspeed_actual, float altitude_desired_NED, float altitude_actual_NED, float dT);
void heading_controller_simple(struct ControllerOutput *headingControl, float headingError_R);
void heading_controller_advanced(struct ControllerOutput *headingControl, float headingActual_R, PositionActualData *positionActual, VelocityActualData *velocityActual, float curvature, float true_airspeed, float true_airspeed_desired, float dT);

float desiredTrackingHeading(PositionActualData *positionActual, float curvature, float true_airspeed_desired, float dT);

void initializeFixedWingPathFollower()
{
	// Initialize UAVOs
	FixedWingPathFollowerSettingsInitialize();
	FixedWingAirspeedsInitialize();
	AirspeedActualInitialize();

	// Register callbacks
	FixedWingAirspeedsConnectCallback(SettingsUpdatedCb);
	FixedWingPathFollowerSettingsConnectCallback(SettingsUpdatedCb);

	// Register queues
	pathManagerStatusQueue = xQueueCreate(1, sizeof(UAVObjEvent));
	PathManagerStatusConnectQueue(pathManagerStatusQueue);

	// Allocate memory
	integral = (struct Integral *) pvPortMalloc(sizeof(struct Integral));
	memset(integral, 0, sizeof(struct Integral));

	pathSegmentDescriptor = (PathSegmentDescriptorData *) pvPortMalloc(sizeof(PathSegmentDescriptorData));
	memset(pathSegmentDescriptor, 0, sizeof(PathSegmentDescriptorData));


	// Load all settings
	SettingsUpdatedCb((UAVObjEvent *)NULL);
}

void zeroGuidanceIntegral(){
	integral->total_energy_error = 0;
	integral->calibrated_airspeed_error = 0;
	integral->line_error = 0;
	integral->circle_error = 0;
}


/**
 * @brief
 */
int8_t updateFixedWingDesiredStabilization()
{
	// Check if the path manager has updated.
	UAVObjEvent ev;
	if (xQueueReceive(pathManagerStatusQueue, &ev, 0) == pdTRUE)
	{
		PathManagerStatusData pathManagerStatusData;
		PathManagerStatusGet(&pathManagerStatusData);

		// Fixme: This isn't a very elegant check. Since the manager can update it's state with new loci, but
		// still have the original ActiveSegment, the pathcounter was introduced, which only resets when the
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

	float altitudeError_NED;
	float headingError_R;


	float altitudeDesired_NED;
	float headingDesired_R;

	VelocityActualGet(&velocityActual);
	StabilizationDesiredGet(&stabilizationDesired);

	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	// Current airspeed
	AirspeedActualTrueAirspeedGet(&true_airspeed);
	AirspeedActualCalibratedAirspeedGet(&calibrated_airspeed);

	// Current heading
	float headingActual_R = atan2f(velocityActual.East, velocityActual.North);

	/**
	 * Compute setpoints.
	 */
	/**
	 * TODO: These setpoints only need to be calibrated once per locus update
	 */
	// Set desired calibrated airspeed, bounded by airframe limits
	calibrated_airspeed_desired = bound_min_max(pathDesired.EndingVelocity, fixedWingAirspeeds.StallSpeedDirty, fixedWingAirspeeds.AirSpeedMax);

	// Set the desired true airspeed, assuming STP atmospheric conditions. This isn't ideal, but we don't have a reliable source of temperature or pressure
	AirParameters air_STP = initialize_air_structure();
	true_airspeed_desired = cas2tas(calibrated_airspeed_desired, -positionActual.Down, &air_STP);

	// Set the desired altitude
	altitudeDesired_NED = pathDesired.End[2];

	// Set the desired heading.
	headingDesired_R = desiredTrackingHeading(&positionActual, pathDesired.Curvature, true_airspeed_desired, dT);

	/**
	 * Compute setpoint errors
	 */
	// Airspeed error
	calibrated_airspeed_error = calibrated_airspeed_desired - calibrated_airspeed;

	// Altitude error
	altitudeError_NED = altitudeDesired_NED - positionActual.Down;

	// Heading error, wrapped to [-pi,pi]
	headingError_R = circular_modulus_rad(headingDesired_R - headingActual_R);

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
	if (0)
		heading_controller_simple(&headingControl, headingError_R);
	else
	{
		heading_controller_advanced(&headingControl, headingActual_R, &positionActual, &velocityActual, pathDesired.Curvature, true_airspeed, true_airspeed_desired, dT);
	}


	// Sum all controllers
	stabilizationDesired.Throttle = bound_min_max(headingControl.throttle + airspeedControl.throttle + totalEnergyControl.throttle,
												  fixedwingpathfollowerSettings.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLELIMIT_MIN],
												  fixedwingpathfollowerSettings.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLELIMIT_MAX]);
	stabilizationDesired.Roll     = bound_min_max(headingControl.roll + airspeedControl.roll + totalEnergyControl.roll,
												  fixedwingpathfollowerSettings.RollLimit[FIXEDWINGPATHFOLLOWERSETTINGS_ROLLLIMIT_MIN],
												  fixedwingpathfollowerSettings.RollLimit[FIXEDWINGPATHFOLLOWERSETTINGS_ROLLLIMIT_MAX]);
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

/**
 * Calculate command for following simple vector based line. Taken from R. Beard at BYU.
 */
static float followStraightLine(float r[3], float q[3], float p[3], float chi_inf, float k_path, float k_psi_int, float delT){
	float chi_q=atan2f(q[1], q[0]);

	float err_xt=-sinf(chi_q)*(p[0]-r[0])+cosf(chi_q)*(p[1]-r[1]); // Compute cross-track error
	integral->line_error+=delT*err_xt;
	float psi_command = chi_q-chi_inf*2.0f/PI*atanf(k_path*err_xt)-k_psi_int*integral->line_error;

	return psi_command;
}


/**
 * Calculate command for following simple vector based orbit. Taken from R. Beard at BYU.
 */
static float followOrbit(float c[3], float rho, bool direction, float p[3], float k_orbit, float k_psi_int, float delT){
	float pncn=p[0]-c[0];
	float pece=p[1]-c[1];
	float d=sqrtf(pncn*pncn + pece*pece);

	float err_orbit=d-rho;
	integral->circle_error+=err_orbit*delT;

	float phi=atan2f(pece, pncn);

	float psi_command = (direction==true) ?
		phi + (PI/2.0f + atanf(k_orbit*err_orbit) + k_psi_int*integral->circle_error): // Turn clockwise
		phi - (PI/2.0f + atanf(k_orbit*err_orbit) + k_psi_int*integral->circle_error); // Turn counter-clockwise

	return psi_command;
}


/**
 * @brief updateDestination Takes path segment descriptors and writes the path to the PathDesired UAVO
 */
static void updateDestination(){
	PathSegmentDescriptorData pathSegmentDescriptor_old;

	int8_t ret;
	ret = PathSegmentDescriptorInstGet(activeSegment-1, &pathSegmentDescriptor_old);
	if(ret != 0){
			if (activeSegment == 0) { // This means we're going to the first switching locus.
				PositionActualData positionActual;
				PositionActualGet(&positionActual);

				pathDesired.Start[0]=positionActual.North;
				pathDesired.Start[1]=positionActual.East;
				pathDesired.Start[2]=positionActual.Down;

				// TODO: Figure out if this can't happen in normal behavior. Consider adding a warning if so.
			}
			else{
			//TODO: Set off a warning

			return;
			}
	}
	else{
		pathDesired.Start[0]=pathSegmentDescriptor_old.SwitchingLocus[0];
		pathDesired.Start[1]=pathSegmentDescriptor_old.SwitchingLocus[1];
		pathDesired.Start[2]=pathSegmentDescriptor_old.SwitchingLocus[2];
	}

	ret = PathSegmentDescriptorInstGet(activeSegment, pathSegmentDescriptor);
	if(ret != 0){
			//TODO: Set off a warning

			return;
	}

	// For a straight line use the switching locus as the vector endpoint...
	if(pathSegmentDescriptor->PathCurvature == 0){
		pathDesired.End[0]=pathSegmentDescriptor->SwitchingLocus[0];
		pathDesired.End[1]=pathSegmentDescriptor->SwitchingLocus[1];
		pathDesired.End[2]=pathSegmentDescriptor->SwitchingLocus[2];

		pathDesired.Curvature = 0;
	}
	else{ // ...but for an arc, use the switching loci to calculate the arc center
		float *oldPosition_NE = pathDesired.Start;
		float *newPosition_NE = pathSegmentDescriptor->SwitchingLocus;
		float arcCenter_XY[2];
		bool ret;

		ret = find_arc_center(oldPosition_NE, newPosition_NE, 1.0f/pathSegmentDescriptor->PathCurvature, arcCenter_XY, pathSegmentDescriptor->PathCurvature > 0, pathSegmentDescriptor->ArcRank == PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR);

		if (ret == CENTER_FOUND){
			pathDesired.End[0]=arcCenter_XY[0];
			pathDesired.End[1]=arcCenter_XY[1];
			pathDesired.End[2]=pathSegmentDescriptor->SwitchingLocus[2];
		}
		else { //---- This is bad, but we have to handle it.----///
			// The path manager should catch this and handle it, but in case it doesn't we'll circle around the midpoint. This
			// way we still maintain positive control, and will satisfy the path requirements, making sure we don't get stuck
			pathDesired.End[0]=(oldPosition_NE[0] + newPosition_NE[0])/2.0f;
			pathDesired.End[1]=(oldPosition_NE[1] + newPosition_NE[1])/2.0f;
			pathDesired.End[2]=pathSegmentDescriptor->SwitchingLocus[2];

			// TODO: Set alarm warning
			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER, SYSTEMALARMS_ALARM_WARNING);
		}

		// Set max average roll as half the vehicle's maximum roll
		StabilizationSettingsRollMaxGet(&max_roll_D);

		//Calculate arc_radius using r*omega=v and omega = g/V_g * tan(max_roll_D)
		float min_radius = powf(pathSegmentDescriptor->FinalVelocity,2)/(9.805f*tanf(fabsf(max_roll_D/2.0f*DEG2RAD)));
		arc_radius = fabsf(1.0f/pathSegmentDescriptor->PathCurvature) > min_radius ? fabsf(1.0f/pathSegmentDescriptor->PathCurvature) : min_radius;
		pathDesired.Curvature = sign(pathSegmentDescriptor->PathCurvature) / arc_radius;
	}

	//-------------------------------------------------//
	//FIXME: Inspect pathDesired values for NaN or Inf.//
	//-------------------------------------------------//

	pathDesired.EndingVelocity=pathSegmentDescriptor->FinalVelocity;

#if defined(PATHDESIRED_DIAGNOSTICS)
	PathDesiredSet(&pathDesired);
#endif
}


void airspeedController(struct ControllerOutput *airspeedControl, float calibrated_airspeed_error, float altitudeError_NED, float dT)
{
	// This is the throttle value required for level flight at the given airspeed
	float feedForwardThrottle = fixedwingpathfollowerSettings.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLELIMIT_NEUTRAL];


	/**
	 * Compute desired pitch command
	 */

#define AIRSPEED_KP      fixedwingpathfollowerSettings.AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_AIRSPEEDPI_KP]
#define AIRSPEED_KI      fixedwingpathfollowerSettings.AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_AIRSPEEDPI_KI]
#define AIRSPEED_ILIMIT	 fixedwingpathfollowerSettings.AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_AIRSPEEDPI_ILIMIT]

	if (AIRSPEED_KI > 0.0f){
		//Integrate with saturation
		integral->calibrated_airspeed_error=bound_min_max(integral->calibrated_airspeed_error + calibrated_airspeed_error * dT,
									  -AIRSPEED_ILIMIT/AIRSPEED_KI,
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
void totalEnergyController(struct ControllerOutput *energyControl, float true_airspeed_desired, float true_airspeed_actual, float altitude_desired_NED, float altitude_actual_NED, float dT)
{
	//Proxy because instead of m*(1/2*v^2+g*h), it's v^2+2*gh. This saves processing power
	float totalEnergyProxySetpoint=powf(true_airspeed_desired, 2.0f) - 2.0f*9.8f*altitude_desired_NED;
	float totalEnergyProxyActual=powf(true_airspeed_actual, 2.0f) - 2.0f*9.8f*altitude_actual_NED;
	float errorTotalEnergy= totalEnergyProxySetpoint - totalEnergyProxyActual;

#define THROTTLE_KP fixedwingpathfollowerSettings.ThrottlePI[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLEPI_KP]
#define THROTTLE_KI fixedwingpathfollowerSettings.ThrottlePI[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLEPI_KI]
#define THROTTLE_ILIMIT fixedwingpathfollowerSettings.ThrottlePI[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLEPI_ILIMIT]

	//Integrate with bound. Make integral leaky for better performance. Approximately 30s time constant.
	if (THROTTLE_KI > 0.0f){
		integral->total_energy_error=bound_min_max(integral->total_energy_error+errorTotalEnergy*dT,
										 -THROTTLE_ILIMIT/THROTTLE_KI,
										 THROTTLE_ILIMIT/THROTTLE_KI)*(1.0f-1.0f/(1.0f+30.0f/dT));
	}

	// Assign altitude controller outputs
	energyControl->throttle = errorTotalEnergy*THROTTLE_KP + integral->total_energy_error*THROTTLE_KI;
	energyControl->roll = 0;
	energyControl->pitch = 0;
	energyControl->yaw = 0;
}


/**
 * @brief desiredTrackingHeading This calculates the heading setpoint as a function of a vector field going onto a path. This is
 * based off of research into Lyanpov controllers, performed by R. Beard at Bringham Young University, Utah, USA.
 * @param pathSegmentDescriptor The type of path descriptor the vehicle will be following
 * @param positionActual Current NED position
 * @param true_airspeed_desired Used to adapt gains so that the trajectory across the vectorfield has similar turning rates no matter the airspeed
 * @param dT Time step used for error integrator
 * @return
 */
float desiredTrackingHeading(PositionActualData *positionActual, float curvature, float true_airspeed_desired, float dT)
{
	float p[3]={positionActual->North, positionActual->East, positionActual->Down};
	float *c = pathDesired.End;
	float *r = pathDesired.Start;
	float q[3] = {c[0]-r[0], c[1]-r[1], c[2]-r[2]};

	float k_psi_int = fixedwingpathfollowerSettings.FollowerIntegralGain;
	//========================================
	//SHOULD NOT BE HARD CODED

	float chi_inf = PI/4.0f; //THIS NEEDS TO BE A FUNCTION OF HOW LONG OUR PATH IS.

	//Saturate chi_inf. I.e., never approach the path at a steeper angle than 45 degrees
	chi_inf = chi_inf > PI/4.0f ? PI/4.0f : chi_inf;
	//========================================

	float headingDesired_R;

	if (curvature == 0) { // Straight line has no curvature
		float k_path  = fixedwingpathfollowerSettings.VectorFollowingGain/true_airspeed_desired; //Divide gain by airspeed so that the vector field scales with airspeed
		headingDesired_R=followStraightLine(r, q, p, chi_inf, k_path, k_psi_int, dT);
	}
	else {
		float k_orbit = fixedwingpathfollowerSettings.OrbitFollowingGain/true_airspeed_desired;  //Divide gain by airspeed so that the vector field scales with airspeed
		if(curvature > 0) // Turn clockwise
			headingDesired_R=followOrbit(c, arc_radius, true, p, k_orbit, k_psi_int, dT);
		else // Turn counter-clockwise
			headingDesired_R=followOrbit(c, arc_radius, false, p, k_orbit, k_psi_int, dT);
	}

	return headingDesired_R;
}


/**
 * This simplified heading controller only computes a roll command based on heading error
 */
void heading_controller_simple(struct ControllerOutput *headingControl, float headingError_R)
{

	// Assign heading controller outputs
	headingControl->throttle = 0;
	headingControl->roll = (headingError_R * fixedwingpathfollowerSettings.HeadingPI[FIXEDWINGPATHFOLLOWERSETTINGS_HEADINGPI_KP]) * RAD2DEG;
	headingControl->pitch = 0;
	headingControl->yaw = 0;
}

/**
 * This advanced heading controller computes a roll command and yaw command based on
 */
void heading_controller_advanced(struct ControllerOutput *headingControl, float headingActual_R,
								 PositionActualData *positionActual, VelocityActualData *velocityActual,
								 float curvature, float true_airspeed, float true_airspeed_desired, float dT)
{
	float gamma;
	float err_xt;
	float err_xt_dot;

	float phi_max = fixedwingpathfollowerSettings.RollLimit[FIXEDWINGPATHFOLLOWERSETTINGS_ROLLLIMIT_MAX] * DEG2RAD;
	float psi = headingActual_R;
	float psi_tilde_thresh = PI/4; // Beyond 45 degrees of course, full roll is applied
	float gamma_max = fixedwingpathfollowerSettings.PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGS_PITCHLIMIT_MAX] * DEG2RAD; // Maximum glide path angle

	float p[3]={positionActual->North, positionActual->East, positionActual->Down};
	float *c = pathDesired.End;
	float *r = pathDesired.Start;
	float q[3] = {c[0]-r[0], c[1]-r[1], c[2]-r[2]};

	float V = true_airspeed;

	gamma = atan2f(velocityActual->Down, sqrtf(powf(velocityActual->North, 2) + powf(velocityActual->East, 2)));


	// Roll command
	float roll_c_R;

	if (curvature == 0) {
		float k1 = 3.9/true_airspeed_desired; // Dividing by airspeed ensures that roll rates stay constant with increasing scale
		float k2 = 1;
		float k3 = 13/true_airspeed_desired;

		float chi_q=atan2f(q[1], q[0]);
		float psi_tilde = circular_modulus_rad(psi - chi_q); // This is the difference between the vehicle heading and the path heading

		err_xt = -sinf(chi_q)*(p[0] - r[0]) + cosf(chi_q)*(p[1] - r[1]); // Compute cross-track error
		err_xt_dot = V * sinf(psi_tilde) * cosf(gamma);// + wind_y //TODO: add wind estimate in the local reference frame

		if (psi_tilde < -psi_tilde_thresh)
			roll_c_R = phi_max;
		else if (psi_tilde > psi_tilde_thresh)
			roll_c_R = -phi_max;
		else
		{
			float M1 = tanf(phi_max);
			float M2 = GRAVITY/2.0f * M1 * cosf(psi_tilde_thresh) * cosf(gamma_max);
			roll_c_R = -bound_sym((k1*err_xt_dot + bound_sym(k2*(k1*err_xt + k3*err_xt_dot), M2))/(GRAVITY*cosf(psi_tilde)*cosf(gamma)), M1);
		}
	}
	else {
		float k4 = 4.5/true_airspeed_desired;
		float k5 = 0.4;
		float k6 = 13/true_airspeed_desired;


		int8_t lambda;
		if (curvature < 0)
			lambda = -1;
		else
			lambda = 1;

		float W = 0;
		float psi_wind;
		/* W = sqrtf(powf(windActual->North, 2) + powf(windActual->East, 2));*/
		if (W == 0)
			psi_wind = 0;
/*		else
			psi_wind = atan2f(windActual->East, windActual->North);
*/
		float Phi = atan2f(p[1]-c[1], p[0]-c[0]);
		float psi_d = Phi + lambda * PI/2.0f;
		float psi_tilde = circular_modulus_rad(psi - psi_d); // This is the difference between the vehicle heading and the path heading

		float d = sqrtf(powf(p[0]-c[0], 2) + powf(p[1]-c[1], 2));
		float d_min = bound_min_max(arc_radius * 0.50f, V*(V + W)/(GRAVITY * tanf(phi_max)), arc_radius); // Try for 50% of the arc radius

		float d_tilde = d - arc_radius;
		float d_tilde_dot = -lambda*V*sinf(psi_tilde)*cosf(gamma) + W*cosf(psi-psi_wind);

		float M4;
		if (d_min > 1e3)
			M4 = tanf(phi_max) - V*V/(d_min*GRAVITY)*cosf(gamma_max)*cosf(psi_tilde_thresh);
		else
			M4 = tanf(phi_max)*(1 - cosf(gamma_max)*cosf(psi_tilde_thresh));

		float M5 = 1/2.0f * M4*GRAVITY*fabsf(cosf(psi_tilde_thresh)*cosf(gamma_max) - W/V);

		float z2 = k4*d_tilde + k6*d_tilde_dot;
		float zeta = k5*z2;

		// Test for navigation errors that come from limits being exceeded.
		if (W < V * cosf(psi_tilde_thresh) * cosf(gamma_max))
			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER, SYSTEMALARMS_ALARM_WARNING);
		else if	(d_min/arc_radius < 0.90f) // Check if the d_min is greater than 90% of arc_radius
			AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER, SYSTEMALARMS_ALARM_WARNING);


		if (d < d_min)
			roll_c_R = 0;
		else if (lambda*psi_tilde <= -psi_tilde_thresh)
			roll_c_R =  lambda * phi_max;
		else if (lambda*psi_tilde >=  psi_tilde_thresh)
			roll_c_R = -lambda * phi_max;
		else {
			roll_c_R = atanf(lambda*V*V/(GRAVITY*d)*cosf(gamma)*cosf(psi_tilde)+
							 bound_sym((k4*d_tilde_dot+bound_sym(zeta, M5))/(lambda*GRAVITY*cosf(psi_tilde)*cosf(gamma)+
																		 GRAVITY*W/V*sinf(psi-psi_wind)), M4));
		}
	}

	// Assign heading controller outputs
	headingControl->throttle = 0;
	headingControl->roll = roll_c_R * RAD2DEG;
	headingControl->pitch = 0;
	headingControl->yaw = 0;
}

void SettingsUpdatedCb(UAVObjEvent * ev)
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
 * @}
 * @}
 */
