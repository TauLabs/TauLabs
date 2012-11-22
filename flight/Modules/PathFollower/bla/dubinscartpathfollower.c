/**
 ******************************************************************************
 *
 * @file       dubinscartpathfollowe.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @brief      This module compared @ref PositionActuatl to @ref ActiveWaypoint 
 * and sets @ref AttitudeDesired.  It only does this when the FlightMode field
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
 * Input object: ???
 * Output object: AttitudeDesired
 *
 * This module will periodically update the value of the AttitudeDesired object.
 *
 * The module executes in its own thread in this example.
 *
 * Modules have no API, all communication to other modules is done through UAVObjects.
 * However modules may use the API exposed by shared libraries.
 * See the OpenPilot wiki for more details.
 * http://www.openpilot.org/OpenPilot_Application_Architecture
 *
 */

#include "openpilot.h"
#include "fixedwingpathfollower.h"
#include "fixedwingairspeeds.h"
#include "hwsettings.h"
#include "attitudeactual.h"
#include "positionactual.h"
#include "velocityactual.h"
#include "manualcontrol.h"
#include "flightstatus.h"
#include "airspeedactual.h"
#include "homelocation.h"
#include "stabilizationdesired.h"	// object that will be updated by the module
#include "pathdesired.h"	// object that will be updated by the module
#include "systemsettings.h"

#include "CoordinateConversions.h"

// Private constants
#define MAX_QUEUE_SIZE 4
#define STACK_SIZE_BYTES 750
#define TASK_PRIORITY (tskIDLE_PRIORITY+2)
#define F_PI 3.14159265358979323846f
#define RAD2DEG (180.0f/F_PI)
#define DEG2RAD (F_PI/180.0f)
#define GEE 9.805f
#define CRITICAL_ERROR_THRESHOLD_MS 5000	//Time in [ms] before an error becomes a critical error

// Private types
static struct Integral {
	float totalEnergyError;
	float groundspeedError;

	float lineError;
	float circleError;
} *integral;

// Private variables
extern bool flightStatusUpdate;
static bool homeOrbit = true;

// Private functions
static uint8_t waypointFollowing(uint8_t flightMode,
				 FixedWingPathFollowerSettingsData
				 fixedwingpathfollowerSettings);
//static void FixedWingPathFollowerParamsUpdatedCb(UAVObjEvent * ev);
//static void updateSteadyStateAttitude();
static float bound(float val, float min, float max);
static float followStraightLine(float r[3], float q[3], float p[3],
				float heading, float chi_inf, float k_path,
				float k_psi_int, float delT);
static float followOrbit(float c[3], float rho, bool direction, float p[3],
			 float phi, float k_orbit, float k_psi_int, float delT);

/**
 * @brief Initialize dubins cart path following variables
 */
void initializeDubinsCartPathFollower()
{
	integral = (struct Integral *)pvPortMalloc(sizeof(struct Integral));
	memset(integral, 0, sizeof(struct Integral));
}

uint8_t updateDubinsCartDesiredStabilization(uint8_t flightMode,
					     FixedWingPathFollowerSettingsData
					     fixedwingpathfollowerSettings)
{
	// Compute path follower commands
	switch (flightMode) {
	case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:
	case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
		waypointFollowing(flightMode, fixedwingpathfollowerSettings);
		break;
	default:
		// Be cleaner and reset integrals
		integral->totalEnergyError = 0;
		integral->groundspeedError = 0;
		integral->lineError = 0;
		integral->circleError = 0;

		break;
	}

	return 0;
}

/**
 * Compute desired attitude from the desired velocity
 *
 * Takes in @ref NedActual which has the acceleration in the 
 * NED frame as the feedback term and then compares the 
 * @ref VelocityActual against the @ref VelocityDesired
 */
static uint8_t waypointFollowing(uint8_t flightMode, FixedWingPathFollowerSettingsCCData fixedwingpathfollowerSettingsCC)
{
	float dT = fixedwingpathfollowerSettings.UpdatePeriod / 1000.0f;	//Convert from [ms] to [s]

	VelocityActualData velocityActual;
	StabilizationDesiredData stabDesired;
	float groundspeed;

	float groundspeedDesired;
	float groundspeedError;

	float throttleCommand;

	float headingError_R;
	float yawCommand;

	VelocityActualGet(&velocityActual);
	StabilizationDesiredGet(&stabDesired);
	AirspeedActualTrueAirspeedGet(&groundspeed);

	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	PathDesiredData pathDesired;
	PathDesiredGet(&pathDesired);

	if (flightStatusUpdate) {
		if (flightMode == FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME) {
			// Simple Return To Home mode: climb 10 meters and fly to home position

			pathDesired.Start[PATHDESIRED_START_NORTH] =
			    positionActual.North;
			pathDesired.Start[PATHDESIRED_START_EAST] =
			    positionActual.East;
			pathDesired.Start[PATHDESIRED_START_DOWN] =
			    positionActual.Down;
			pathDesired.End[PATHDESIRED_END_NORTH] = 0;
			pathDesired.End[PATHDESIRED_END_EAST] = 0;
			pathDesired.End[PATHDESIRED_END_DOWN] =
			    positionActual.Down - 10;
			pathDesired.StartingVelocity =
			    fixedwingpathfollowerSettings.BestClimbRateSpeed;
			pathDesired.EndingVelocity =
			    fixedwingpathfollowerSettings.BestClimbRateSpeed;
			pathDesired.Mode = PATHDESIRED_MODE_FLYVECTOR;

			homeOrbit = false;
		} else if (flightMode == FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD) {
			// Simple position hold: stay at present altitude and position

			pathDesired.Start[PATHDESIRED_START_NORTH] = positionActual.North - 1;	//Offset by one so that the start and end points don't perfectly coincide
			pathDesired.Start[PATHDESIRED_START_EAST] =
			    positionActual.East;
			pathDesired.Start[PATHDESIRED_START_DOWN] =
			    positionActual.Down;
			pathDesired.End[PATHDESIRED_END_NORTH] =
			    positionActual.North;
			pathDesired.End[PATHDESIRED_END_EAST] =
			    positionActual.East;
			pathDesired.End[PATHDESIRED_END_DOWN] =
			    positionActual.Down;
			pathDesired.StartingVelocity =
			    fixedwingpathfollowerSettings.BestClimbRateSpeed;
			pathDesired.EndingVelocity =
			    fixedwingpathfollowerSettings.BestClimbRateSpeed;
			pathDesired.Mode = PATHDESIRED_MODE_FLYVECTOR;
		}
		PathDesiredSet(&pathDesired);

		flightStatusUpdate = false;
	}

	/**
	 * Compute speed error (required for throttle and pitch)
	 */

	// Current heading
	float headingActual_R =
	    atan2f(velocityActual.East, velocityActual.North);

	// Desired groundspeed
	groundspeedDesired = pathDesired.EndingVelocity;

	// groundspeed error
	groundspeedError = groundspeedDesired - groundspeed;

	/**
	 * Compute desired throttle command
	 */
    float airspeed_kp       = fixedwingpathfollowerSettingsCC.AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGSCC_AIRSPEEDPI_KP];
    float airspeed_ki       = fixedwingpathfollowerSettingsCC.AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGSCC_AIRSPEEDPI_KI];
    float airspeed_ilimit	  = fixedwingpathfollowerSettingsCC.AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGSCC_AIRSPEEDPI_ILIMIT];

	if (airspeed_ki > 0.0f) {
		//Integrate with saturation
		integral->groundspeedError =
		    bound(integral->groundspeedError + groundspeedError * dT,
			  -airspeed_ilimit / airspeed_ki,
			  airspeed_ilimit / airspeed_ki);
	}
	//Compute the throttle command as err*Kp + errInt*Ki.
	throttleCommand = -(groundspeedError * airspeed_kp
			    + integral->groundspeedError * airspeed_ki);

    float throttlelimit_neutral  = fixedwingpathfollowerSettingsCC.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_THROTTLELIMIT_NEUTRAL];
    float throttlelimit_min      = fixedwingpathfollowerSettingsCC.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_THROTTLELIMIT_MIN];
    float throttlelimit_max      = fixedwingpathfollowerSettingsCC.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_THROTTLELIMIT_MAX];

	// set throttle with saturation
	stabDesired.Throttle = bound(throttleCommand + throttlelimit_neutral,
				     throttlelimit_min, throttlelimit_max);

	/**
	 * Compute desired roll command
	 */

	float p[3] =
	    { positionActual.North, positionActual.East, positionActual.Down };
	float *c = pathDesired.End;
	float *r = pathDesired.Start;
	float q[3] =
	    { pathDesired.End[0] - pathDesired.Start[0],
	   pathDesired.End[1] - pathDesired.Start[1],
	   pathDesired.End[2] - pathDesired.Start[2] };

	float k_path = fixedwingpathfollowerSettings.VectorFollowingGain / 
            pathDesired.EndingVelocity;	//Divide gain by groundspeed so that 
                                        //the turn rate is independent of groundspeed
	float k_orbit = fixedwingpathfollowerSettings.OrbitFollowingGain / 
            pathDesired.EndingVelocity;	//Divide gain by groundspeed so that 
                                        //the turn rate is independent of groundspeed
	float k_psi_int = fixedwingpathfollowerSettings.FollowerIntegralGain;
//========================================
	//SHOULD NOT BE HARD CODED

	bool direction;

	float chi_inf = F_PI / 4.0f;	//THIS NEEDS TO BE A FUNCTION OF HOW LONG OUR PATH IS.

	//Saturate chi_inf. I.e., never approach the path at a steeper angle than 45 degrees
	chi_inf = chi_inf < F_PI / 4.0f ? F_PI / 4.0f : chi_inf;
//========================================      

	float rho;
	float headingCommand_R;

	float pncn = p[0] - c[0];
	float pece = p[1] - c[1];
	float d = sqrtf(pncn * pncn + pece * pece);

//Assume that we want a lateral acceleration of 1.0m/s2. This should yield a nice, non-agressive turn
#define LATERAL_ACCEL_FOR_HOLDING_CIRCLE 1.0f	
	//Calculate radius, rho, using r*omega=v and omega = g/V_g * tan(phi)
	//THIS SHOULD ONLY BE CALCULATED ONCE, INSTEAD OF EVERY TIME
	rho =
	    powf(pathDesired.EndingVelocity,
		 2) / (LATERAL_ACCEL_FOR_HOLDING_CIRCLE);

	typedef enum {
		LINE,
		ORBIT
	} pathTypes_t;

	pathTypes_t pathType;

	//Determine if we should fly on a line or orbit path.
	switch (flightMode) {
	case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:
		if (d < rho + 5.0f * pathDesired.EndingVelocity || homeOrbit == true) {	
            //When approx five seconds from the circle, start integrating into it
			pathType = ORBIT;
			homeOrbit = true;
		} else {
			pathType = LINE;
		}
		break;
	case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
		pathType = ORBIT;
		break;
	default:
		pathType = LINE;
		break;
	}

	//Check to see if we've gone past our destination. Since the path follower 
    //is simply following a vector, it has no concept of where the vector ends. 
    //It will simply keep following it to infinity if we don't stop it. So 
    //while we don't know why the commutation to the next point failed, we 
    //don't know we don't want the plane flying off.
	if (pathType == LINE) {

		//Compute the norm squared of the horizontal path length
        /** @todo IT WOULD BE NICE TO ONLY DO THIS ONCE PER WAYPOINT UPDATE, INSTEAD OF EVERY LOOP */
		float pathLength2 = q[0] * q[0] + q[1] * q[1];	

		//Perform a quick vector math operation, |a| < a.b/|a| = |b|cos(theta), 
        //to test if we've gone past the waypoint. Add in a distance equal to 
        //5s of flight time, for good measure to make sure we don't add jitter.
		if (((p[0] - r[0]) * q[0] + (p[1] - r[1]) * q[1]) >
		    pathLength2 + 5.0f * pathDesired.EndingVelocity) {
			//Whoops, we've really overflown our destination point, and haven't 
            //received any instructions. Start circling
			flightMode = FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD;	
            //flightMode will reset the next time a waypoint changes, so 
            //there's no danger of it getting stuck in orbit mode.
			pathType = ORBIT;
		}
	}

	switch (pathType) {
	case ORBIT:
		if (pathDesired.Mode == PATHDESIRED_MODE_FLYCIRCLELEFT) {
			direction = false;
		} else {
			//In the case where the direction is undefined, always fly in a 
            //clockwise fashion
			direction = true;
		}

		headingCommand_R =
		    followOrbit(c, rho, direction, p, headingActual_R, k_orbit,
				k_psi_int, dT);
		break;
	case LINE:
		headingCommand_R =
		    followStraightLine(r, q, p, headingActual_R, chi_inf,
				       k_path, k_psi_int, dT);
		break;
	}

	//Calculate heading error
	headingError_R = headingCommand_R - headingActual_R;

	//Wrap heading error around circle
	if (headingError_R < -F_PI)
		headingError_R += 2.0f * F_PI;
	if (headingError_R > F_PI)
		headingError_R -= 2.0f * F_PI;

	//GET RID OF THE RAD2DEG. IT CAN BE FACTORED INTO HeadingPI
    float yawlimit_neutral   = fixedwingpathfollowerSettings.RollLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_ROLLLIMIT_NEUTRAL];
    float yawlimit_min       = fixedwingpathfollowerSettings.RollLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_ROLLLIMIT_MIN];
    float yawlimit_max       = fixedwingpathfollowerSettings.RollLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_ROLLLIMIT_MAX];
    float headingpi_kp  = fixedwingpathfollowerSettings.HeadingPI[FIXEDWINGPATHFOLLOWERSETTINGSCC_HEADINGPI_KP];
	yawCommand = (headingError_R * headingpi_kp) * RAD2DEG;

	//Turn heading 

	stabDesired.Yaw = bound(yawlimit_neutral +
				yawCommand, yawlimit_min, yawlimit_max);

#ifdef SIM_OSX
	fprintf(stderr, " headingError_R: %f, rollCommand: %f\n",
		headingError_R, rollCommand);
#endif

	/**
	 * Reset desired roll and pitch commands
	 */
	stabDesired.Roll = 0;
	stabDesired.Pitch = 0;

	stabDesired.
	    StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] =
	    STABILIZATIONDESIRED_STABILIZATIONMODE_NONE;
	stabDesired.
	    StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] =
	    STABILIZATIONDESIRED_STABILIZATIONMODE_NONE;
	stabDesired.
	    StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] =
	    STABILIZATIONDESIRED_STABILIZATIONMODE_RATE;

	StabilizationDesiredSet(&stabDesired);

	//Stuff some debug variables into PathDesired, because right now these 
    //fields aren't being used.
	pathDesired.ModeParameters[0] = yawCommand;
	pathDesired.ModeParameters[1] = groundspeedError;
	pathDesired.ModeParameters[2] = integral->groundspeedError;
//      pathDesired.ModeParameters[3]=alitudeError;
//      pathDesired.UID=errorTotalEnergy;

	PathDesiredSet(&pathDesired);

	return 0;
}

/**
 * Calculate command for following simple vector based line. Taken from R. 
 * Beard at BYU.
 */
float followStraightLine(float r[3], float q[3], float p[3], float psi,
			 float chi_inf, float k_path, float k_psi_int,
			 float delT)
{
	float chi_q = atan2f(q[1], q[0]);
	while (chi_q - psi < -F_PI) {
		chi_q += 2.0f * F_PI;
	}
	while (chi_q - psi > F_PI) {
		chi_q -= 2.0f * F_PI;
	}

	float err_p =
	    -sinf(chi_q) * (p[0] - r[0]) + cosf(chi_q) * (p[1] - r[1]);
	integral->lineError += delT * err_p;
	float psi_command =
	    chi_q - chi_inf * 2.0f / F_PI * atanf(k_path * err_p) -
	    k_psi_int * integral->lineError;

	return psi_command;
}

/**
 * Calculate command for following simple vector based orbit. Taken from R. 
 * Beard at BYU.
 */
float followOrbit(float c[3], float rho, bool direction, float p[3], float psi,
		  float k_orbit, float k_psi_int, float delT)
{
	float pncn = p[0] - c[0];
	float pece = p[1] - c[1];
	float d = sqrtf(pncn * pncn + pece * pece);

	float err_orbit = d - rho;
	integral->circleError += err_orbit * delT;

	float phi = atan2f(pece, pncn);
	while (phi - psi < -F_PI) {
		phi = phi + 2.0f * F_PI;
	}
	while (phi - psi > F_PI) {
		phi = phi - 2.0f * F_PI;
	}

	float psi_command = direction == true ?
	    phi + (F_PI / 2.0f + atanf(k_orbit * err_orbit) +
		   k_psi_int * integral->circleError) : phi - (F_PI / 2.0f +
							       atanf(k_orbit *
								     err_orbit)
							       +
							       k_psi_int *
							       integral->
							       circleError);

#ifdef SIM_OSX
	fprintf(stderr,
		"actual heading: %f, circle error: %f, circl integral: %f, heading command: %f",
		psi, err_orbit, integral->circleError, psi_command);
#endif
	return psi_command;
}

/**
 * Bound input value between limits
 */
static float bound(float val, float min, float max)
{
	if (val < min) {
		val = min;
	} else if (val > max) {
		val = max;
	}
	return val;
}

////Triggered by changes in FixedWingPathFollowerSettings and PathDesired
//static void FixedWingPathFollowerParamsUpdatedCb(UAVObjEvent * ev)
//{
//      FixedWingPathFollowerSettingsGet(&fixedwingpathfollowerSettings);
//      FlightStatusFlightModeGet(&flightMode);
//      PathDesiredGet(&pathDesired);
//      
//      float r[2] = {pathDesired.End[0]-pathDesired.Start[0], pathDesired.End[1]-pathDesired.Start[1]};
//      pathLength=sqrtf(r[0]*r[0]+r[1]*r[1]);
//      
//}
