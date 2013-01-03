/**
 ******************************************************************************
 *
 * @file       fixedwingpathfollower.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     PhoenixPilot, http://github.com/PhoenixPilot Copyright (C) 2012.
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

/**
 * Input object: @ref ActiveWaypoint
 * Input object: @ref PositionActual
 * Input object: @ref ManualControlCommand
 * Output object: @ref StabilizationDesired
 *
 * This module will periodically update the value of the @ref StabilizationDesired object.
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
#include "paths.h"

#include "accels.h"
#include "modulesettings.h"
#include "attitudeactual.h"
#include "pathdesired.h"	// object that will be updated by the module
#include "positionactual.h"
#include "manualcontrol.h"
#include "flightstatus.h"
#include "pathstatus.h"
#include "airspeedactual.h"
#include "gpsvelocity.h"
#include "gpsposition.h"
#include "fixedwingairspeeds.h"
#include "fixedwingpathfollowersettings.h"
#include "fixedwingpathfollowerstatus.h"
#include "homelocation.h"
#include "nedposition.h"
#include "stabilizationdesired.h"
#include "stabilizationsettings.h"
#include "systemsettings.h"
#include "velocitydesired.h"
#include "velocityactual.h"
#include "CoordinateConversions.h"

// Private constants
#define MAX_QUEUE_SIZE 4
#define STACK_SIZE_BYTES 1548
#define TASK_PRIORITY (tskIDLE_PRIORITY+2)
#define F_PI 3.14159265358979323846f
#define RAD2DEG (180.0f/F_PI)
#define DEG2RAD (F_PI/180.0f)
#define GEE 9.81f
// Private types

// Private variables
static bool module_enabled = false;
static xTaskHandle pathfollowerTaskHandle;
static PathDesiredData pathDesired;
static PathStatusData pathStatus;
static FixedWingPathFollowerSettingsData fixedwingpathfollowerSettings;
static FixedWingAirspeedsData fixedWingAirspeeds;

// Private functions
static void pathfollowerTask(void *parameters);
static void SettingsUpdatedCb(UAVObjEvent * ev);
static void updatePathVelocity();
static uint8_t updateFixedDesiredAttitude();
static void airspeedActualUpdatedCb(UAVObjEvent * ev);
static float bound(float val, float min, float max);

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
	ModuleSettingsInitialize();
	uint8_t module_state[MODULESETTINGS_STATE_NUMELEM];
	ModuleSettingsStateGet(module_state);
	if (module_state[MODULESETTINGS_STATE_FIXEDWINGPATHFOLLOWER] == MODULESETTINGS_STATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (module_enabled) {
		FixedWingPathFollowerSettingsInitialize();
		FixedWingAirspeedsInitialize();
		FixedWingPathFollowerStatusInitialize();
		PathDesiredInitialize();
		PathStatusInitialize();
		VelocityDesiredInitialize();
		AirspeedActualInitialize();
	}

	return 0;
}
MODULE_INITCALL(FixedWingPathFollowerInitialize, FixedWingPathFollowerStart)

static float northVelIntegral = 0;
static float eastVelIntegral = 0;
static float downVelIntegral = 0;

static float bearingIntegral = 0;
static float speedIntegral = 0;
static float accelIntegral = 0;
static float powerIntegral = 0;
static float airspeedErrorInt=0;

// correct speed by measured airspeed
static float indicatedAirspeedActualBias = 0;

/**
 * Module thread, should not return.
 */
static void pathfollowerTask(void *parameters)
{
	SystemSettingsData systemSettings;
	FlightStatusData flightStatus;
	
	portTickType lastUpdateTime;
	
	AirspeedActualConnectCallback(airspeedActualUpdatedCb);
	FixedWingPathFollowerSettingsConnectCallback(SettingsUpdatedCb);
	FixedWingAirspeedsConnectCallback(SettingsUpdatedCb);
	PathDesiredConnectCallback(SettingsUpdatedCb);
	
	FixedWingPathFollowerSettingsGet(&fixedwingpathfollowerSettings);
	PathDesiredGet(&pathDesired);
	
	// Main task loop
	lastUpdateTime = xTaskGetTickCount();
	while (1) {

		// Conditions when this runs:
		// 1. Must have FixedWing type airframe
		// 2. Flight mode is PositionHold and PathDesired.Mode is Endpoint  OR
		//    FlightMode is PathPlanner and PathDesired.Mode is Endpoint or Path

		SystemSettingsGet(&systemSettings);
		if ( (systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_FIXEDWING) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_FIXEDWINGELEVON) &&
			(systemSettings.AirframeType != SYSTEMSETTINGS_AIRFRAMETYPE_FIXEDWINGVTAIL) )
		{
			AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,SYSTEMALARMS_ALARM_WARNING);
			vTaskDelay(1000);
			continue;
		}

		// Continue collecting data if not enough time
		vTaskDelayUntil(&lastUpdateTime, fixedwingpathfollowerSettings.UpdatePeriod / portTICK_RATE_MS);

		
		FlightStatusGet(&flightStatus);
		PathStatusGet(&pathStatus);
	
		uint8_t result;
		// Check the combinations of flightmode and pathdesired mode
		switch(flightStatus.FlightMode) {
			case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
			case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:
				if (pathDesired.Mode == PATHDESIRED_MODE_FLYENDPOINT) {
					updatePathVelocity();
					result = updateFixedDesiredAttitude();
					if (result) {
						AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,SYSTEMALARMS_ALARM_OK);
					} else {
						AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,SYSTEMALARMS_ALARM_WARNING);
					}
				} else {
					AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,SYSTEMALARMS_ALARM_ERROR);
				}
				break;
			case FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER:
				pathStatus.Status = PATHSTATUS_STATUS_INPROGRESS;
				switch(pathDesired.Mode) {
					case PATHDESIRED_MODE_FLYENDPOINT:
					case PATHDESIRED_MODE_FLYVECTOR:
					case PATHDESIRED_MODE_FLYCIRCLERIGHT:
					case PATHDESIRED_MODE_FLYCIRCLELEFT:
						updatePathVelocity();
						result = updateFixedDesiredAttitude();
						if (result) {
							AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,SYSTEMALARMS_ALARM_OK);
						} else {
							AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,SYSTEMALARMS_ALARM_WARNING);
						}
						break;
					default:
						pathStatus.Status = PATHSTATUS_STATUS_CRITICAL;
						AlarmsSet(SYSTEMALARMS_ALARM_GUIDANCE,SYSTEMALARMS_ALARM_ERROR);
						break;
				}
				break;
			default:
				// Be cleaner and get rid of global variables
				northVelIntegral = 0;
				eastVelIntegral = 0;
				downVelIntegral = 0;
				bearingIntegral = 0;
				speedIntegral = 0;
				accelIntegral = 0;
				powerIntegral = 0;

				break;
		}
		PathStatusSet(&pathStatus);
	}
}

/**
 * Compute desired velocity from the current position and path
 *
 * Takes in @ref PositionActual and compares it to @ref PathDesired 
 * and computes @ref VelocityDesired
 */
static void updatePathVelocity()
{
	PositionActualData positionActual;
	PositionActualGet(&positionActual);
	VelocityActualData velocityActual;
	VelocityActualGet(&velocityActual);

	float cur[3] = {positionActual.North, positionActual.East, positionActual.Down};
	struct path_status progress;

	path_progress(&pathDesired, cur, &progress);
	
	float groundspeed = 0;
	float altitudeSetpoint = 0;
	switch (pathDesired.Mode) {
		case PATHDESIRED_MODE_FLYCIRCLERIGHT:
		case PATHDESIRED_MODE_DRIVECIRCLERIGHT:
		case PATHDESIRED_MODE_FLYCIRCLELEFT:
		case PATHDESIRED_MODE_DRIVECIRCLELEFT:
			groundspeed = pathDesired.EndingVelocity;
			altitudeSetpoint = pathDesired.End[2];
			break;
		case PATHDESIRED_MODE_FLYENDPOINT:
		case PATHDESIRED_MODE_DRIVEENDPOINT:
		case PATHDESIRED_MODE_FLYVECTOR:
		case PATHDESIRED_MODE_DRIVEVECTOR:
		default:
			groundspeed = pathDesired.StartingVelocity + (pathDesired.EndingVelocity - pathDesired.StartingVelocity) *
				bound(progress.fractional_progress,0,1);
			altitudeSetpoint = pathDesired.Start[2] + (pathDesired.End[2] - pathDesired.Start[2]) *
				bound(progress.fractional_progress,0,1);
			break;
	}
	// this ensures a significant forward component at least close to the real trajectory
	if (groundspeed<fixedWingAirspeeds.BestClimbRateSpeed/10.)
		groundspeed=fixedWingAirspeeds.BestClimbRateSpeed/10.;
	
	// calculate velocity - can be zero if waypoints are too close
	VelocityDesiredData velocityDesired;
	velocityDesired.North = progress.path_direction[0] * groundspeed;
	velocityDesired.East = progress.path_direction[1] * groundspeed;
	
	float error_speed = progress.error * fixedwingpathfollowerSettings.HorizontalPosP;

	// calculate correction - can also be zero if correction vector is 0 or no error present
	velocityDesired.North += progress.correction_direction[0] * error_speed;
	velocityDesired.East += progress.correction_direction[1] * error_speed;
	
	float downError = altitudeSetpoint - positionActual.Down;
	velocityDesired.Down = downError * fixedwingpathfollowerSettings.VerticalPosP;

	// update pathstatus
	pathStatus.error = progress.error;
	pathStatus.fractional_progress = progress.fractional_progress;

	pathStatus.fractional_progress = progress.fractional_progress;
	if (pathStatus.fractional_progress < 1)
		pathStatus.Status = PATHSTATUS_STATUS_INPROGRESS;
	else
		pathStatus.Status = PATHSTATUS_STATUS_COMPLETED;
	VelocityDesiredSet(&velocityDesired);
}

/**
 * Compute desired attitude from the desired velocity
 *
 * Takes in @ref NedActual which has the acceleration in the 
 * NED frame as the feedback term and then compares the 
 * @ref VelocityActual against the @ref VelocityDesired
 */
static uint8_t updateFixedDesiredAttitude()
{

	uint8_t result = 1;

	float dT = fixedwingpathfollowerSettings.UpdatePeriod / 1000.0f; //Convert from [ms] to [s]

	VelocityDesiredData velocityDesired;
	VelocityActualData velocityActual;
	StabilizationDesiredData stabDesired;
	AttitudeActualData attitudeActual;
	AccelsData accels;
	FixedWingPathFollowerSettingsData fixedwingpathfollowerSettings;
	StabilizationSettingsData stabSettings;
	FixedWingPathFollowerStatusData fixedwingpathfollowerStatus;
	AirspeedActualData airspeedActual;
	
	float groundspeedActual;
	float groundspeedDesired;
	float indicatedAirspeedActual;
	float indicatedAirspeedDesired;
	float airspeedError;
	
	float pitchCommand;

	float descentspeedDesired;
	float descentspeedError;
	float powerError;
	float powerCommand;

	float bearingError;
	float bearingCommand;

	FixedWingPathFollowerSettingsGet(&fixedwingpathfollowerSettings);

	FixedWingPathFollowerStatusGet(&fixedwingpathfollowerStatus);
	
	VelocityActualGet(&velocityActual);
//	VelocityDesiredGet(&velocityDesired);
	StabilizationDesiredGet(&stabDesired);
	VelocityDesiredGet(&velocityDesired);
	AttitudeActualGet(&attitudeActual);
	AccelsGet(&accels);
	StabilizationSettingsGet(&stabSettings);
	AirspeedActualGet(&airspeedActual);


	/**
	 * Compute speed error (required for throttle and pitch)
	 */

	// Current ground speed
	groundspeedActual  = sqrtf( velocityActual.East*velocityActual.East + velocityActual.North*velocityActual.North );
	// note that airspeedActualBias is ( calibratedAirspeed - groundSpeed ) at the time of measurement,
	// but thanks to accelerometers,  groundspeed reacts faster to changes in direction
	// than airspeed and gps sensors alone
	indicatedAirspeedActual     = groundspeedActual + indicatedAirspeedActualBias;

	// Desired ground speed
	groundspeedDesired = sqrtf(velocityDesired.North*velocityDesired.North + velocityDesired.East*velocityDesired.East);
	indicatedAirspeedDesired    = bound( groundspeedDesired + indicatedAirspeedActualBias,
							fixedWingAirspeeds.BestClimbRateSpeed,
							fixedWingAirspeeds.CruiseSpeed);

	// Airspeed error
	airspeedError = indicatedAirspeedDesired - indicatedAirspeedActual;

	// Vertical speed error
	descentspeedDesired = bound (
						velocityDesired.Down,
						-fixedWingAirspeeds.VerticalVelMax,
						fixedWingAirspeeds.VerticalVelMax);
	descentspeedError = descentspeedDesired - velocityActual.Down;

	// Error condition: wind speed is higher than maximum allowed speed. We are forced backwards!
	fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_WIND] = 0;
	if (groundspeedDesired - indicatedAirspeedActualBias <= 0 ) {
		fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_WIND] = 1;
		result = 0;
	}
	// Error condition: plane too slow or too fast
	fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_HIGHSPEED] = 0;
	fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_LOWSPEED] = 0;
	if ( indicatedAirspeedActual >  fixedWingAirspeeds.AirSpeedMax) {
		fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_OVERSPEED] = 1;
		result = 0;
	}
	if ( indicatedAirspeedActual >  fixedWingAirspeeds.CruiseSpeed * 1.2f) {
		fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_HIGHSPEED] = 1;
		result = 0;
	}
	if (indicatedAirspeedActual < fixedWingAirspeeds.BestClimbRateSpeed * 0.8f && 1) { //The next three && 1 are placeholders for UAVOs representing LANDING and TAKEOFF
		fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_LOWSPEED] = 1;
		result = 0;
	}
	if (indicatedAirspeedActual < fixedWingAirspeeds.StallSpeedClean && 1 && 1) { //Where the && 1 represents the UAVO that will control whether the airplane is prepped for landing or not
		fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_STALLSPEED] = 1;
		result = 0;
	}
	if (indicatedAirspeedActual < fixedWingAirspeeds.StallSpeedDirty && 1) {
		fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_STALLSPEED] = 1;
		result = 0;
	}
	
	if (indicatedAirspeedActual<1e-6) {
		// prevent division by zero, abort without controlling anything. This guidance mode is not suited for takeoff or touchdown, or handling stationary planes
		// also we cannot handle planes flying backwards, lets just wait until the nose drops
		fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_LOWSPEED] = 1;
		return 0;
	}

	/**
	 * Compute desired throttle command
	 */
	// compute proportional throttle response
	powerError = -descentspeedError +
		bound (
			 (airspeedError/fixedWingAirspeeds.BestClimbRateSpeed)* fixedwingpathfollowerSettings.AirspeedToVerticalCrossFeed[FIXEDWINGPATHFOLLOWERSETTINGS_AIRSPEEDTOVERTICALCROSSFEED_KP] ,
			 -fixedwingpathfollowerSettings.AirspeedToVerticalCrossFeed[FIXEDWINGPATHFOLLOWERSETTINGS_AIRSPEEDTOVERTICALCROSSFEED_MAX],
			 fixedwingpathfollowerSettings.AirspeedToVerticalCrossFeed[FIXEDWINGPATHFOLLOWERSETTINGS_AIRSPEEDTOVERTICALCROSSFEED_MAX]
			 );
	
	// compute saturated integral error throttle response. Make integral leaky for better performance. Approximately 30s time constant.
	if (fixedwingpathfollowerSettings.PowerPI[FIXEDWINGPATHFOLLOWERSETTINGS_POWERPI_KI] >0) {
		powerIntegral =	bound(powerIntegral + -descentspeedError * dT, 
										-fixedwingpathfollowerSettings.PowerPI[FIXEDWINGPATHFOLLOWERSETTINGS_POWERPI_ILIMIT]/fixedwingpathfollowerSettings.PowerPI[FIXEDWINGPATHFOLLOWERSETTINGS_POWERPI_KI],
										fixedwingpathfollowerSettings.PowerPI[FIXEDWINGPATHFOLLOWERSETTINGS_POWERPI_ILIMIT]/fixedwingpathfollowerSettings.PowerPI[FIXEDWINGPATHFOLLOWERSETTINGS_POWERPI_KI]
										)*(1.0f-1.0f/(1.0f+30.0f/dT));
	} else powerIntegral = 0;
	
	// Compute final throttle response
	powerCommand = bound(
			(powerError * fixedwingpathfollowerSettings.PowerPI[FIXEDWINGPATHFOLLOWERSETTINGS_POWERPI_KP] +
			powerIntegral*	fixedwingpathfollowerSettings.PowerPI[FIXEDWINGPATHFOLLOWERSETTINGS_POWERPI_KI]) + fixedwingpathfollowerSettings.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLELIMIT_NEUTRAL],
			fixedwingpathfollowerSettings.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLELIMIT_MIN],
			fixedwingpathfollowerSettings.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLELIMIT_MAX]);

	//Output internal state to telemetry
	fixedwingpathfollowerStatus.Error[FIXEDWINGPATHFOLLOWERSTATUS_ERROR_POWER] = powerError;
	fixedwingpathfollowerStatus.ErrorInt[FIXEDWINGPATHFOLLOWERSTATUS_ERRORINT_POWER] = powerIntegral;
	fixedwingpathfollowerStatus.Command[FIXEDWINGPATHFOLLOWERSTATUS_COMMAND_POWER] = powerCommand;

	// set throttle
	stabDesired.Throttle = powerCommand;

	// Error condition: plane cannot hold altitude at current speed.
	fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_LOWPOWER] = 0;
	if (
		powerCommand == fixedwingpathfollowerSettings.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLELIMIT_MAX] // throttle at maximum
		&& velocityActual.Down > 0 // we ARE going down
		&& descentspeedDesired < 0 // we WANT to go up
		&& airspeedError > 0 // we are too slow already
		) 
	{
		fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_LOWPOWER] = 1;
		result = 0;
	}
	// Error condition: plane keeps climbing despite minimum throttle (opposite of above)
	fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_HIGHPOWER] = 0;
	if (
		powerCommand == fixedwingpathfollowerSettings.ThrottleLimit[FIXEDWINGPATHFOLLOWERSETTINGS_THROTTLELIMIT_MIN] // throttle at minimum
		&& velocityActual.Down < 0 // we ARE going up
		&& descentspeedDesired > 0 // we WANT to go down
		&& airspeedError < 0 // we are too fast already
		) 
	{
		fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_HIGHPOWER] = 1;
		result = 0;
	}


	/**
	 * Compute desired pitch command
	 */
		
	if (fixedwingpathfollowerSettings.SpeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_SPEEDPI_KI] > 0){
		//Integrate with saturation
		airspeedErrorInt=bound(airspeedErrorInt + airspeedError * dT, 
				-fixedwingpathfollowerSettings.SpeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_SPEEDPI_ILIMIT]/fixedwingpathfollowerSettings.SpeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_SPEEDPI_KI],
				fixedwingpathfollowerSettings.SpeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_SPEEDPI_ILIMIT]/fixedwingpathfollowerSettings.SpeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_SPEEDPI_KI]);
	}				
	
	//Compute the cross feed from vertical speed to pitch, with saturation
	float verticalSpeedToPitchCommandComponent=bound (-descentspeedError * fixedwingpathfollowerSettings.VerticalToPitchCrossFeed[FIXEDWINGPATHFOLLOWERSETTINGS_VERTICALTOPITCHCROSSFEED_KP],
			 -fixedwingpathfollowerSettings.VerticalToPitchCrossFeed[FIXEDWINGPATHFOLLOWERSETTINGS_VERTICALTOPITCHCROSSFEED_MAX],
			 fixedwingpathfollowerSettings.VerticalToPitchCrossFeed[FIXEDWINGPATHFOLLOWERSETTINGS_VERTICALTOPITCHCROSSFEED_MAX]
			 );
	
	//Compute the pitch command as err*Kp + errInt*Ki + X_feed.
	pitchCommand= -(airspeedError*fixedwingpathfollowerSettings.SpeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_SPEEDPI_KP] 
						 + airspeedErrorInt*fixedwingpathfollowerSettings.SpeedPI[FIXEDWINGPATHFOLLOWERSETTINGS_SPEEDPI_KI]
						 )	+ verticalSpeedToPitchCommandComponent;
	
	fixedwingpathfollowerStatus.Error[FIXEDWINGPATHFOLLOWERSTATUS_ERROR_SPEED] = airspeedError;
	fixedwingpathfollowerStatus.ErrorInt[FIXEDWINGPATHFOLLOWERSTATUS_ERRORINT_SPEED] = airspeedErrorInt;
	fixedwingpathfollowerStatus.Command[FIXEDWINGPATHFOLLOWERSTATUS_COMMAND_SPEED] = pitchCommand;

	stabDesired.Pitch = bound(fixedwingpathfollowerSettings.PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGS_PITCHLIMIT_NEUTRAL] +
		pitchCommand,
		fixedwingpathfollowerSettings.PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGS_PITCHLIMIT_MIN],
		fixedwingpathfollowerSettings.PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGS_PITCHLIMIT_MAX]);

	// Error condition: high speed dive
	fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_PITCHCONTROL] = 0;
	if (
		pitchCommand == fixedwingpathfollowerSettings.PitchLimit[FIXEDWINGPATHFOLLOWERSETTINGS_PITCHLIMIT_MAX] // pitch demand is full up
		&& velocityActual.Down > 0 // we ARE going down
		&& descentspeedDesired < 0 // we WANT to go up
		&& airspeedError < 0 // we are too fast already
		) {
		fixedwingpathfollowerStatus.Errors[FIXEDWINGPATHFOLLOWERSTATUS_ERRORS_PITCHCONTROL] = 1;
		result = 0;
	}


	/**
	 * Compute desired roll command
	 */
	if (groundspeedDesired> 1e-6) {
		bearingError = RAD2DEG * (atan2f(velocityDesired.East,velocityDesired.North) - atan2f(velocityActual.East,velocityActual.North));
	} else {
		// if we are not supposed to move, keep going wherever we are now. Don't make things worse by changing direction.
		bearingError = 0;
	}
	
	if (bearingError<-180.0f) bearingError+=360.0f;
	if (bearingError>180.0f) bearingError-=360.0f;

	bearingIntegral = bound(bearingIntegral + bearingError * dT * fixedwingpathfollowerSettings.BearingPI[FIXEDWINGPATHFOLLOWERSETTINGS_BEARINGPI_KI], 
		-fixedwingpathfollowerSettings.BearingPI[FIXEDWINGPATHFOLLOWERSETTINGS_BEARINGPI_ILIMIT],
		fixedwingpathfollowerSettings.BearingPI[FIXEDWINGPATHFOLLOWERSETTINGS_BEARINGPI_ILIMIT]);
	bearingCommand = (bearingError * fixedwingpathfollowerSettings.BearingPI[FIXEDWINGPATHFOLLOWERSETTINGS_BEARINGPI_KP] +
		bearingIntegral);

	fixedwingpathfollowerStatus.Error[FIXEDWINGPATHFOLLOWERSTATUS_ERROR_BEARING] = bearingError;
	fixedwingpathfollowerStatus.ErrorInt[FIXEDWINGPATHFOLLOWERSTATUS_ERRORINT_BEARING] = bearingIntegral;
	fixedwingpathfollowerStatus.Command[FIXEDWINGPATHFOLLOWERSTATUS_COMMAND_BEARING] = bearingCommand;
	
	stabDesired.Roll = bound( fixedwingpathfollowerSettings.RollLimit[FIXEDWINGPATHFOLLOWERSETTINGS_ROLLLIMIT_NEUTRAL] +
		bearingCommand,
		fixedwingpathfollowerSettings.RollLimit[FIXEDWINGPATHFOLLOWERSETTINGS_ROLLLIMIT_MIN],
		fixedwingpathfollowerSettings.RollLimit[FIXEDWINGPATHFOLLOWERSETTINGS_ROLLLIMIT_MAX] );

	// TODO: find a check to determine loss of directional control. Likely needs some check of derivative


	/**
	 * Compute desired yaw command
	 */
	// TODO implement raw control mode for yaw and base on Accels.Y
	stabDesired.Yaw = 0;


	stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_NONE;
	
	StabilizationDesiredSet(&stabDesired);

	FixedWingPathFollowerStatusSet(&fixedwingpathfollowerStatus);

	return result;
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

static void SettingsUpdatedCb(UAVObjEvent * ev)
{
	FixedWingPathFollowerSettingsGet(&fixedwingpathfollowerSettings);
	FixedWingAirspeedsGet(&fixedWingAirspeeds);
	PathDesiredGet(&pathDesired);
}

static void airspeedActualUpdatedCb(UAVObjEvent * ev)
{

	AirspeedActualData airspeedActual;
	VelocityActualData velocityActual;

	AirspeedActualGet(&airspeedActual);
	VelocityActualGet(&velocityActual);
	float groundspeed = sqrtf(velocityActual.East*velocityActual.East + velocityActual.North*velocityActual.North );

	
	indicatedAirspeedActualBias = airspeedActual.CalibratedAirspeed - groundspeed;
	// note - we do fly by Indicated Airspeed (== calibrated airspeed)
	// however since airspeed is updated less often than groundspeed, we use sudden changes to groundspeed to offset the airspeed by the same measurement.

}
