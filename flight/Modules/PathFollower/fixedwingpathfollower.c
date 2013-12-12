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
#include "fixedwingpathfollower.h"
#include "fixedwingairspeeds.h"

#include "positionactual.h"
#include "velocityactual.h"
#include "flightstatus.h"
#include "airspeedactual.h"
#include "stabilizationdesired.h"
#include "pathdesired.h"
#include "systemsettings.h"

#include "coordinate_conversions.h"

// Private constants
#define MAX_QUEUE_SIZE 4
#define STACK_SIZE_BYTES 750
#define TASK_PRIORITY (tskIDLE_PRIORITY+2)
#define CRITICAL_ERROR_THRESHOLD_MS 5000	//Time in [ms] before an error becomes a critical error

// Private types
static struct Integral {
	float totalEnergyError;
	float airspeedError;

	float lineError;
	float circleError;
} *integral;

// Private variables
extern bool flightStatusUpdate;
static bool homeOrbit = true;

// Private functions
static uint8_t waypointFollowing(uint8_t flightMode, FixedWingPathFollowerSettingsCCData fixedwingpathfollowerSettings);
static float bound(float val, float min, float max);
static float followStraightLine(float r[3], float q[3], float p[3],
				float heading, float chi_inf, float k_path,
				float k_psi_int, float delT);
static float followOrbit(float c[3], float rho, bool direction, float p[3],
			 float phi, float k_orbit, float k_psi_int, float delT);

void initializeFixedWingPathFollower()
{
	integral = (struct Integral *)pvPortMalloc(sizeof(struct Integral));
	memset(integral, 0, sizeof(struct Integral));
}

uint8_t updateFixedWingDesiredStabilization(uint8_t flightMode, FixedWingPathFollowerSettingsCCData fixedwingpathfollowerSettings)
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
		integral->airspeedError = 0;
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
uint8_t waypointFollowing(uint8_t flightMode, FixedWingPathFollowerSettingsCCData fixedwingpathfollowerSettings)
{
	float dT = fixedwingpathfollowerSettings.UpdatePeriod / 1000.0f;	//Convert from [ms] to [s]

	VelocityActualData velocityActual;
	StabilizationDesiredData stabDesired;
	float trueAirspeed;

	float calibratedAirspeedActual;
	float airspeedDesired;
	float airspeedError;

	float pitchCommand;

	float powerCommand;
	float headingError_R;
	float rollCommand;

	//TODO: Move these out of the loop
	FixedWingAirspeedsData fixedwingAirspeeds;
	FixedWingAirspeedsGet(&fixedwingAirspeeds);

	VelocityActualGet(&velocityActual);
	StabilizationDesiredGet(&stabDesired);
	// TODO: Create UAVO that merges airspeed together
	AirspeedActualTrueAirspeedGet(&trueAirspeed);

	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	PathDesiredData pathDesired;
	PathDesiredGet(&pathDesired);

	if (flightStatusUpdate) {

		//Reset integrals
		integral->totalEnergyError = 0;
		integral->airspeedError = 0;
		integral->lineError = 0;
		integral->circleError = 0;

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
			    fixedwingAirspeeds.BestClimbRateSpeed;
			pathDesired.EndingVelocity =
			    fixedwingAirspeeds.BestClimbRateSpeed;
			pathDesired.Mode = PATHDESIRED_MODE_FLYVECTOR;

			homeOrbit = false;
		} else if (flightMode == FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD) {
			// Simple position hold: stay at present altitude and position

			//Offset by one so that the start and end points don't perfectly coincide
			pathDesired.Start[PATHDESIRED_START_NORTH] =
			    positionActual.North - 1;
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
			    fixedwingAirspeeds.BestClimbRateSpeed;
			pathDesired.EndingVelocity =
			    fixedwingAirspeeds.BestClimbRateSpeed;
			pathDesired.Mode = PATHDESIRED_MODE_FLYVECTOR;
		}
		PathDesiredSet(&pathDesired);

		flightStatusUpdate = false;
	}

	/**
	 * Compute speed error (required for throttle and pitch)
	 */

	// Current airspeed
	calibratedAirspeedActual = trueAirspeed;	//BOOOOOOOOOO!!! Where's the conversion from TAS to CAS?

	// Current heading
	float headingActual_R =
	    atan2f(velocityActual.East, velocityActual.North);

	// Desired airspeed
	airspeedDesired = pathDesired.EndingVelocity;

	// Airspeed error
	airspeedError = airspeedDesired - calibratedAirspeedActual;

	/**
	 * Compute desired throttle command
	 */

	//Proxy because instead of m*(1/2*v^2+g*h), it's v^2+2*gh. This saves processing power
	float totalEnergyProxySetpoint = powf(pathDesired.EndingVelocity,
					      2.0f) -
	    2.0f * GRAVITY * pathDesired.End[2];
	float totalEnergyProxyActual =
	    powf(trueAirspeed, 2.0f) - 2.0f * GRAVITY * positionActual.Down;
	float errorTotalEnergy =
	    totalEnergyProxySetpoint - totalEnergyProxyActual;

	float throttle_kp =
	    fixedwingpathfollowerSettings.
	    ThrottlePI[FIXEDWINGPATHFOLLOWERSETTINGSCC_THROTTLEPI_KP];
	float throttle_ki = fixedwingpathfollowerSettings.ThrottlePI[FIXEDWINGPATHFOLLOWERSETTINGSCC_THROTTLEPI_KI];
	float throttle_ilimit = fixedwingpathfollowerSettings.ThrottlePI[FIXEDWINGPATHFOLLOWERSETTINGSCC_THROTTLEPI_ILIMIT];

	//Integrate with bound. Make integral leaky for better performance. Approximately 30s time constant.
	if (throttle_ilimit > 0.0f) {
		integral->totalEnergyError =
		    bound(integral->totalEnergyError + errorTotalEnergy * dT,
			  -throttle_ilimit / throttle_ki,
			  throttle_ilimit / throttle_ki) * (1.0f -
							    1.0f / (1.0f +
								    30.0f /
								    dT));
	}

	powerCommand = errorTotalEnergy * throttle_kp
	    + integral->totalEnergyError * throttle_ki;

	float throttlelimit_neutral =
	    fixedwingpathfollowerSettings.
	    ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_THROTTLELIMIT_NEUTRAL];
	float throttlelimit_min =
	    fixedwingpathfollowerSettings.
	    ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_THROTTLELIMIT_MIN];
	float throttlelimit_max =
	    fixedwingpathfollowerSettings.
	    ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_THROTTLELIMIT_MAX];

	// set throttle
	stabDesired.Throttle = bound(powerCommand + throttlelimit_neutral,
				     throttlelimit_min, throttlelimit_max);
	/**
	 * Compute desired pitch command
	 */

	float airspeed_kp =
	    fixedwingpathfollowerSettings.
	    AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGSCC_AIRSPEEDPI_KP];
	float airspeed_ki =
	    fixedwingpathfollowerSettings.
	    AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGSCC_AIRSPEEDPI_KI];
	float airspeed_ilimit =
	    fixedwingpathfollowerSettings.
	    AirspeedPI[FIXEDWINGPATHFOLLOWERSETTINGSCC_AIRSPEEDPI_ILIMIT];

	if (airspeed_ki > 0.0f) {
		//Integrate with saturation
		integral->airspeedError =
		    bound(integral->airspeedError + airspeedError * dT,
			  -airspeed_ilimit / airspeed_ki,
			  airspeed_ilimit / airspeed_ki);
	}
	//Compute the cross feed from altitude to pitch, with saturation
	float pitchcrossfeed_kp =
	    fixedwingpathfollowerSettings.
	    VerticalToPitchCrossFeed
	    [FIXEDWINGPATHFOLLOWERSETTINGSCC_VERTICALTOPITCHCROSSFEED_KP];
	float pitchcrossfeed_min =
	    fixedwingpathfollowerSettings.
	    VerticalToPitchCrossFeed
	    [FIXEDWINGPATHFOLLOWERSETTINGSCC_VERTICALTOPITCHCROSSFEED_MAX];
	float pitchcrossfeed_max =
	    fixedwingpathfollowerSettings.
	    VerticalToPitchCrossFeed
	    [FIXEDWINGPATHFOLLOWERSETTINGSCC_VERTICALTOPITCHCROSSFEED_MAX];
	float alitudeError = -(pathDesired.End[2] - positionActual.Down);	//Negative to convert from Down to altitude
	float altitudeToPitchCommandComponent =
	    bound(alitudeError * pitchcrossfeed_kp,
		  -pitchcrossfeed_min,
		  pitchcrossfeed_max);

	//Compute the pitch command as err*Kp + errInt*Ki + X_feed.
	pitchCommand = -(airspeedError * airspeed_kp
			 + integral->airspeedError * airspeed_ki) +
	    altitudeToPitchCommandComponent;

	//Saturate pitch command
	float pitchlimit_neutral =
	    fixedwingpathfollowerSettings.
	    PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_PITCHLIMIT_NEUTRAL];
	float pitchlimit_min =
	    fixedwingpathfollowerSettings.
	    PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_PITCHLIMIT_MIN];
	float pitchlimit_max =
	    fixedwingpathfollowerSettings.
	    PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_PITCHLIMIT_MAX];

	stabDesired.Pitch = bound(pitchlimit_neutral +
				  pitchCommand, pitchlimit_min, pitchlimit_max);

	/**
	 * Compute desired roll command
	 */

	float p[3] =
	    { positionActual.North, positionActual.East, positionActual.Down };
	float *c = pathDesired.End;
	float *r = pathDesired.Start;
	float q[3] = { pathDesired.End[0] - pathDesired.Start[0],
		pathDesired.End[1] - pathDesired.Start[1],
		pathDesired.End[2] - pathDesired.Start[2]
	};

	float k_path = fixedwingpathfollowerSettings.VectorFollowingGain / pathDesired.EndingVelocity;	//Divide gain by airspeed so that the turn rate is independent of airspeed
	float k_orbit = fixedwingpathfollowerSettings.OrbitFollowingGain / pathDesired.EndingVelocity;	//Divide gain by airspeed so that the turn rate is independent of airspeed
	float k_psi_int = fixedwingpathfollowerSettings.FollowerIntegralGain;
//========================================
	//SHOULD NOT BE HARD CODED

	bool direction;

	float chi_inf = PI / 4.0f;	//THIS NEEDS TO BE A FUNCTION OF HOW LONG OUR PATH IS.

	//Saturate chi_inf. I.e., never approach the path at a steeper angle than 45 degrees
	chi_inf = chi_inf < PI / 4.0f ? PI / 4.0f : chi_inf;
//========================================      

	float rho;
	float headingCommand_R;

	float pncn = p[0] - c[0];
	float pece = p[1] - c[1];
	float d = sqrtf(pncn * pncn + pece * pece);

//Assume that we want a 15 degree bank angle. This should yield a nice, non-agressive turn
#define ROLL_FOR_HOLDING_CIRCLE 15.0f
	//Calculate radius, rho, using r*omega=v and omega = g/V_g * tan(phi)
	//THIS SHOULD ONLY BE CALCULATED ONCE, INSTEAD OF EVERY TIME
	rho = powf(pathDesired.EndingVelocity,
		 2) / (GRAVITY * tanf(fabs(ROLL_FOR_HOLDING_CIRCLE * DEG2RAD)));

	typedef enum {
		LINE,
		ORBIT
	} pathTypes_t;

	pathTypes_t pathType;

	//Determine if we should fly on a line or orbit path.
	switch (flightMode) {
	case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:
		if (d < rho + 5.0f * pathDesired.EndingVelocity || homeOrbit == true) {	//When approx five seconds from the circle, start integrating into it
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
	//It will simply keep following it to infinity if we don't stop it. So while
	//we don't know why the commutation to the next point failed, we don't know
	//we don't want the plane flying off.
	if (pathType == LINE) {

		//Compute the norm squared of the horizontal path length
		//IT WOULD BE NICE TO ONLY DO THIS ONCE PER WAYPOINT UPDATE, INSTEAD OF
		//EVERY LOOP
		float pathLength2 = q[0] * q[0] + q[1] * q[1];

		//Perform a quick vector math operation, |a| < a.b/|a| = |b|cos(theta),
		//to test if we've gone past the waypoint. Add in a distance equal to 5s
		//of flight time, for good measure to make sure we don't add jitter.
		if (((p[0] - r[0]) * q[0] + (p[1] - r[1]) * q[1]) >
		    pathLength2 + 5.0f * pathDesired.EndingVelocity) {
			//Whoops, we've really overflown our destination point, and haven't
			//received any instructions. Start circling
			//flightMode will reset the next time a waypoint changes, so there's
			//no danger of it getting stuck in orbit mode.
			flightMode = FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD;
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
	if (headingError_R < -PI)
		headingError_R += 2.0f * PI;
	if (headingError_R > PI)
		headingError_R -= 2.0f * PI;

	//GET RID OF THE RAD2DEG. IT CAN BE FACTORED INTO HeadingPI
	float rolllimit_neutral =
	    fixedwingpathfollowerSettings.
	    RollLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_ROLLLIMIT_NEUTRAL];
	float rolllimit_min =
	    fixedwingpathfollowerSettings.
	    RollLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_ROLLLIMIT_MIN];
	float rolllimit_max =
	    fixedwingpathfollowerSettings.
	    RollLimit[FIXEDWINGPATHFOLLOWERSETTINGSCC_ROLLLIMIT_MAX];
	float headingpi_kp =
	    fixedwingpathfollowerSettings.
	    HeadingPI[FIXEDWINGPATHFOLLOWERSETTINGSCC_HEADINGPI_KP];
	rollCommand = (headingError_R * headingpi_kp) * RAD2DEG;

	//Turn heading 

	stabDesired.Roll = bound(rolllimit_neutral +
				 rollCommand, rolllimit_min, rolllimit_max);

#ifdef SIM_OSX
	fprintf(stderr, " headingError_R: %f, rollCommand: %f\n",
		headingError_R, rollCommand);
#endif

	/**
	 * Compute desired yaw command
	 */
	// Coordinated flight, so we reset the desired yaw.
	stabDesired.Yaw = 0;

	stabDesired.
	    StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] =
	    STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	stabDesired.
	    StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] =
	    STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	stabDesired.
	    StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] =
	    STABILIZATIONDESIRED_STABILIZATIONMODE_NONE;

	StabilizationDesiredSet(&stabDesired);

	//Stuff some debug variables into PathDesired, because right now these
	//fields aren't being used.
	pathDesired.ModeParameters[0] = pitchCommand;
	pathDesired.ModeParameters[1] = airspeedError;
	pathDesired.ModeParameters[2] = integral->airspeedError;
	pathDesired.ModeParameters[3] = alitudeError;
	pathDesired.UID = errorTotalEnergy;

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
	while (chi_q - psi < -PI) {
		chi_q += 2.0f * PI;
	}
	while (chi_q - psi > PI) {
		chi_q -= 2.0f * PI;
	}

	float err_p = -sinf(chi_q) * (p[0] - r[0]) + cosf(chi_q) * (p[1] - r[1]);
	integral->lineError += delT * err_p;
	float psi_command = chi_q - chi_inf * 2.0f / PI * atanf(k_path * err_p) -
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
	while (phi - psi < -PI) {
		phi = phi + 2.0f * PI;
	}
	while (phi - psi > PI) {
		phi = phi - 2.0f * PI;
	}

	float psi_command = (direction == true) ?
	    phi + (PI / 2.0f + atanf(k_orbit * err_orbit) +
		   k_psi_int * integral->circleError) : phi - (PI / 2.0f +
							       atanf(k_orbit *
								     err_orbit) +
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

/**
 * @}
 * @}
 */
