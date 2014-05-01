/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup AltitudeHoldModule Altitude hold module
 * @{
 *
 * @file       altitudehold.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      This module runs an EKF to estimate altitude from just a barometric
 *             sensor and controls throttle to hold a fixed altitude
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
 * Input object: @ref PositionActual
 * Input object: @ref VelocityActual
 * Input object: @ref ManualControlCommand
 * Input object: @ref AltitudeHoldDesired
 * Output object: @ref StabilizationDesired
 *
 * Hold the VTOL aircraft at a fixed altitude by running nested
 * control loops to stay at the AltitudeHoldDesired height.
 */

#include "openpilot.h"
#include "physical_constants.h"
#include "misc_math.h"
#include "pid.h"

#include "attitudeactual.h"
#include "altitudeholdsettings.h"
#include "altitudeholddesired.h"
#include "flightstatus.h"
#include "stabilizationdesired.h"
#include "positionactual.h"
#include "velocityactual.h"
#include "modulesettings.h"

// Private constants
#define MAX_QUEUE_SIZE 4
#define STACK_SIZE_BYTES 540
#define TASK_PRIORITY (tskIDLE_PRIORITY+1)

// Private variables
static xTaskHandle altitudeHoldTaskHandle;
static xQueueHandle queue;
static bool module_enabled;

// Private functions
static void altitudeHoldTask(void *parameters);

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t AltitudeHoldStart()
{
	// Start main task if it is enabled
	if (module_enabled) {
		xTaskCreate(altitudeHoldTask, (signed char *)"AltitudeHold", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &altitudeHoldTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_ALTITUDEHOLD, altitudeHoldTaskHandle);
		return 0;
	}
	return -1;

}

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t AltitudeHoldInitialize()
{
#ifdef MODULE_AltitudeHold_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_ALTITUDEHOLD] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if(module_enabled) {
		AltitudeHoldSettingsInitialize();
		AltitudeHoldDesiredInitialize();

		// Create object queue
		queue = xQueueCreate(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));

		return 0;
	}

	return -1;
}
MODULE_INITCALL(AltitudeHoldInitialize, AltitudeHoldStart);

/**
 * Module thread, should not return.
 */
static void altitudeHoldTask(void *parameters)
{
	bool engaged = false;

	AltitudeHoldDesiredData altitudeHoldDesired;
	StabilizationDesiredData stabilizationDesired;
	AltitudeHoldSettingsData altitudeHoldSettings;

	UAVObjEvent ev;
	struct pid velocity_pid;

	// Listen for object updates.
	AltitudeHoldDesiredConnectQueue(queue);
	AltitudeHoldSettingsConnectQueue(queue);
	FlightStatusConnectQueue(queue);

	AltitudeHoldSettingsGet(&altitudeHoldSettings);
	pid_configure(&velocity_pid, altitudeHoldSettings.VelocityKp,
		          altitudeHoldSettings.VelocityKi, 0.0f, 1.0f);

	AlarmsSet(SYSTEMALARMS_ALARM_ALTITUDEHOLD, SYSTEMALARMS_ALARM_OK);

	// Main task loop
	const uint32_t dt_ms = 5;
	const float dt_s = dt_ms * 0.001f;
	uint32_t timeout = dt_ms;

	while (1) {
		if ( xQueueReceive(queue, &ev, MS2TICKS(timeout)) != pdTRUE ) {

		} else if (ev.obj == FlightStatusHandle()) {

			uint8_t flight_mode;
			FlightStatusFlightModeGet(&flight_mode);

			if (flight_mode == FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD && !engaged) {
				// Copy the current throttle as a starting point for integral
				StabilizationDesiredThrottleGet(&velocity_pid.iAccumulator);
				velocity_pid.iAccumulator *= 1000.0f; // pid library scales up accumulator by 1000
				engaged = true;
			} else if (flight_mode != FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD)
				engaged = false;

			// Run loop at 20 Hz when engaged otherwise just slowly wait for it to be engaged
			timeout = engaged ? dt_ms : 100;

		} else if (ev.obj == AltitudeHoldDesiredHandle()) {
			AltitudeHoldDesiredGet(&altitudeHoldDesired);
		} else if (ev.obj == AltitudeHoldSettingsHandle()) {
			AltitudeHoldSettingsGet(&altitudeHoldSettings);

			pid_configure(&velocity_pid, altitudeHoldSettings.VelocityKp,
				          altitudeHoldSettings.VelocityKi, 0.0f, 1.0f);
		}

		// When engaged compute altitude controller output
		if (engaged) {
			float position_z, velocity_z, altitude_error;

			PositionActualDownGet(&position_z);
			VelocityActualDownGet(&velocity_z);
			position_z = -position_z; // Use positive up convention
			velocity_z = -velocity_z; // Use positive up convention

			// Compute the altitude error
			altitude_error = altitudeHoldDesired.Altitude - position_z;

			// Velocity desired is from the outer controller plus the set point
			float velocity_desired = altitude_error * altitudeHoldSettings.PositionKp + altitudeHoldDesired.ClimbRate;
			float throttle_desired = pid_apply_antiwindup(&velocity_pid, 
			                    velocity_desired - velocity_z,
			                    0, 1.0f, // positive limits since this is throttle
			                    dt_s);

			if (altitudeHoldSettings.AttitudeComp == ALTITUDEHOLDSETTINGS_ATTITUDECOMP_TRUE) {
				// Throttle desired is at this point the mount desired in the up direction, we can
				// account for the attitude if desired
				AttitudeActualData attitudeActual;
				AttitudeActualGet(&attitudeActual);

				// Project a unit vector pointing up into the body frame and
				// get the z component
				float fraction = attitudeActual.q1 * attitudeActual.q1 -
				                 attitudeActual.q2 * attitudeActual.q2 -
				                 attitudeActual.q3 * attitudeActual.q3 +
				                 attitudeActual.q4 * attitudeActual.q4;

				// Dividing by the fraction remaining in the vertical projection will
				// attempt to compensate for tilt. This acts like the thrust is linear
				// with the output which isn't really true. If the fraction is starting
				// to go negative we are inverted and should shut off throttle
				throttle_desired = (fraction > 0.1f) ? (throttle_desired / fraction) : 0.0f;
			}

			StabilizationDesiredGet(&stabilizationDesired);
			stabilizationDesired.Throttle = bound_min_max(throttle_desired, 0.0f, 1.0f);
			stabilizationDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
			stabilizationDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
			stabilizationDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK;
			stabilizationDesired.Roll = altitudeHoldDesired.Roll;
			stabilizationDesired.Pitch = altitudeHoldDesired.Pitch;
			stabilizationDesired.Yaw = altitudeHoldDesired.Yaw;
			StabilizationDesiredSet(&stabilizationDesired);
		}

	}
}

/**
 * @}
 * @}
 */

