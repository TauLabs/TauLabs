/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup AltitudeHoldModule Altitude hold module
 * @{
 *
 * @file       altitudehold.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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
 * Input object: @ref AltitudeHoldDesired
 * Input object: @ref BaroAltitude
 * Input object: @ref Accels
 * Output object: @ref StabilizationDesired
 * Output object: @ref AltHoldSmoothed
 *
 * Runs an EKF on the @ref accels and @ref BaroAltitude to estimate altitude, velocity
 * and acceleration which is output in @ref AltHoldSmoothed.  Then a control value is
 * computed for @StabilizationDesired throttle.  Roll and pitch are set to Attitude
 * mode and use the values from @AltHoldDesired.	
 *
 * The module executes in its own thread in this example.
 */

#include "openpilot.h"
#include "physical_constants.h"
#include <math.h>
#include "coordinate_conversions.h"
#include "altholdsmoothed.h"
#include "attitudeactual.h"
#include "altitudeholdsettings.h"
#include "altitudeholddesired.h"	// object that will be updated by the module
#include "baroaltitude.h"
#include "positionactual.h"
#include "flightstatus.h"
#include "stabilizationdesired.h"
#include "accels.h"
#include "modulesettings.h"

// Private constants
#define MAX_QUEUE_SIZE 4
#define STACK_SIZE_BYTES 1200
#define TASK_PRIORITY (tskIDLE_PRIORITY+1)
#define ACCEL_DOWNSAMPLE 4

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
		AltHoldSmoothedInitialize();

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
	/**
	 * AH_RESET - reinitialize filter
	 * AH_WAITING_INIT - waiting for initialization of filter
	 * AH_WAITING_BARO - awaiting an update from the baro
	 * AH_RUNNING - ready to perform update of EKF
	 */
	enum altitudehold_state {AH_RESET, AH_WAITIING_INIT, AH_WAITING_BARO, AH_RUNNING} state = AH_RESET;
	bool engaged = false;
	float starting_altitude;
	float throttleIntegral;
	float error;

	/* EKF variables */
	float z[4];
	float V[4][4];
	uint32_t accel_downsample_count = 0;
	float accels_accum[3] = {0,0,0};

	AltitudeHoldDesiredData altitudeHoldDesired;
	StabilizationDesiredData stabilizationDesired;
	AltitudeHoldSettingsData altitudeHoldSettings;

	portTickType this_time_ms;
	portTickType last_update_time_ms = TICKS2MS(xTaskGetTickCount());
	UAVObjEvent ev;

	// Listen for object updates.
	AltitudeHoldDesiredConnectQueue(queue);
	BaroAltitudeConnectQueue(queue);
	FlightStatusConnectQueue(queue);
	AccelsConnectQueue(queue);
	AltitudeHoldSettingsConnectQueue(queue);

	AltitudeHoldSettingsGet(&altitudeHoldSettings);
	BaroAltitudeAltitudeGet(&starting_altitude);

	AlarmsSet(SYSTEMALARMS_ALARM_ALTITUDEHOLD, SYSTEMALARMS_ALARM_ERROR);

	// Main task loop
	while (1) {

		if (state == AH_RESET) {
			/* Reset all the EKF variables */
			memset(z, 0, sizeof(z));
			memset(V, 0, sizeof(V));
			memset(accels_accum, 0, sizeof(accels_accum));
			accel_downsample_count = 0;

			/* Set initial variances */
			V[0][0] = 10.0f;
			V[1][1] = 100.0f;
			V[2][2] = 100.0f;
			V[3][3] = 1000.0f;

			state = AH_WAITING_BARO;
		}

		// Wait until the sensors are updated, if a timeout then go to failsafe
		if ( xQueueReceive(queue, &ev, MS2TICKS(100)) != pdTRUE )
		{
			if(!engaged)
				throttleIntegral = 0;

			// Todo: Add alarm if it should be running
			continue;
		} else if (ev.obj == BaroAltitudeHandle()) {

			if (state == AH_WAITIING_INIT) {
				AccelsData accels;
				AccelsGet(&accels);
				BaroAltitudeData baro;
				BaroAltitudeGet(&baro);

				z[0] = baro.Altitude;
				z[1] = 0;
				z[2] = accels.z;
				z[3] = 0;
				state = AH_WAITING_BARO;
			} else
				state = AH_RUNNING;

		} else if (ev.obj == FlightStatusHandle()) {
			FlightStatusData flightStatus;
			FlightStatusGet(&flightStatus);

			if(flightStatus.FlightMode == FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD && !engaged) {
				// Copy the current throttle as a starting point for integral
				StabilizationDesiredThrottleGet(&throttleIntegral);
				error = 0;
				engaged = true;

				AltHoldSmoothedAltitudeGet(&starting_altitude);
			} else if (flightStatus.FlightMode != FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD)
				engaged = false;
		} else if (ev.obj == AccelsHandle()) {
			AccelsData accels;
			AccelsGet(&accels);

			accels_accum[0] += accels.x;
			accels_accum[1] += accels.y;
			accels_accum[2] += accels.z;
			accel_downsample_count++;

			this_time_ms = TICKS2MS(xTaskGetTickCount());

			AttitudeActualData attitudeActual;
			AttitudeActualGet(&attitudeActual);
			BaroAltitudeData baro;
			BaroAltitudeGet(&baro);

			if (state != AH_RUNNING)
				continue;
			state = AH_WAITING_BARO;

			/* Downsample accels to stop this calculation consuming too much CPU */
			accels.x = accels_accum[0] / accel_downsample_count;
			accels.y = accels_accum[1] / accel_downsample_count;
			accels.z = accels_accum[2] / accel_downsample_count;
			accels_accum[0] = accels_accum[1] = accels_accum[2] = 0;
			accel_downsample_count = 0;

			static uint32_t timeval;
			float dT = PIOS_DELAY_DiffuS(timeval) / 1.0e6f;
			timeval = PIOS_DELAY_GetRaw();

			/* Local working variables */
			float z_new[4];
			float P[4][4], K[4][2], x[2];
			float G[4] = {1.0e-15f, 1.0e-15f, altitudeHoldSettings.AccelDrift, 1.0e-7};
			float S[2] = {altitudeHoldSettings.PressureNoise, altitudeHoldSettings.AccelNoise};

			if (S[0] == 0 || S[1] == 0 || G[2] == 0) {
				continue;
			}

			float Rbe[3][3];
			Quaternion2R(&attitudeActual.q1, Rbe);
			x[0] = baro.Altitude;
			x[1] = -(Rbe[0][2]*accels.x+ Rbe[1][2]*accels.y + Rbe[2][2]*accels.z + GRAVITY);

			P[0][0] = dT*(V[0][1]+dT*V[1][1])+V[0][0]+G[0]+dT*V[1][0];
			P[0][1] = dT*(V[0][2]+dT*V[1][2])+V[0][1]+dT*V[1][1];
			P[0][2] = V[0][2]+dT*V[1][2];
			P[0][3] = V[0][3]+dT*V[1][3];
			P[1][0] = dT*(V[1][1]+dT*V[2][1])+V[1][0]+dT*V[2][0];
			P[1][1] = dT*(V[1][2]+dT*V[2][2])+V[1][1]+G[1]+dT*V[2][1];
			P[1][2] = V[1][2]+dT*V[2][2];
			P[1][3] = V[1][3]+dT*V[2][3];
			P[2][0] = V[2][0]+dT*V[2][1];
			P[2][1] = V[2][1]+dT*V[2][2];
			P[2][2] = V[2][2]+G[2];
			P[2][3] = V[2][3];
			P[3][0] = V[3][0]+dT*V[3][1];
			P[3][1] = V[3][1]+dT*V[3][2];
			P[3][2] = V[3][2];
			P[3][3] = V[3][3]+G[3];

			K[0][0] = -(V[2][2]*S[0]+V[2][3]*S[0]+V[3][2]*S[0]+V[3][3]*S[0]+G[2]*S[0]+G[3]*S[0]+S[0]*S[1])/(V[0][0]*G[2]+V[0][0]*G[3]+V[2][2]*G[0]+V[2][3]*G[0]+V[3][2]*G[0]+V[3][3]*G[0]+V[0][0]*S[1]+V[2][2]*S[0]+V[2][3]*S[0]+V[3][2]*S[0]+V[3][3]*S[0]+V[0][0]*V[2][2]-V[0][2]*V[2][0]+V[0][0]*V[2][3]+V[0][0]*V[3][2]-V[0][2]*V[3][0]-V[2][0]*V[0][3]+V[0][0]*V[3][3]-V[0][3]*V[3][0]+G[0]*G[2]+G[0]*G[3]+G[0]*S[1]+G[2]*S[0]+G[3]*S[0]+S[0]*S[1]+(dT*dT)*V[1][1]*V[2][2]-(dT*dT)*V[1][2]*V[2][1]+(dT*dT)*V[1][1]*V[2][3]+(dT*dT)*V[1][1]*V[3][2]-(dT*dT)*V[1][2]*V[3][1]-(dT*dT)*V[2][1]*V[1][3]+(dT*dT)*V[1][1]*V[3][3]-(dT*dT)*V[1][3]*V[3][1]+dT*V[0][1]*G[2]+dT*V[1][0]*G[2]+dT*V[0][1]*G[3]+dT*V[1][0]*G[3]+dT*V[0][1]*S[1]+dT*V[1][0]*S[1]+(dT*dT)*V[1][1]*G[2]+(dT*dT)*V[1][1]*G[3]+(dT*dT)*V[1][1]*S[1]+dT*V[0][1]*V[2][2]+dT*V[1][0]*V[2][2]-dT*V[0][2]*V[2][1]-dT*V[2][0]*V[1][2]+dT*V[0][1]*V[2][3]+dT*V[0][1]*V[3][2]+dT*V[1][0]*V[2][3]+dT*V[1][0]*V[3][2]-dT*V[0][2]*V[3][1]-dT*V[2][0]*V[1][3]-dT*V[0][3]*V[2][1]-dT*V[1][2]*V[3][0]+dT*V[0][1]*V[3][3]+dT*V[1][0]*V[3][3]-dT*V[0][3]*V[3][1]-dT*V[3][0]*V[1][3])+1.0f;
			K[0][1] = ((V[0][2]+V[0][3])*S[0]+dT*(V[1][2]+V[1][3])*S[0])/(V[0][0]*G[2]+V[0][0]*G[3]+V[2][2]*G[0]+V[2][3]*G[0]+V[3][2]*G[0]+V[3][3]*G[0]+V[0][0]*S[1]+V[2][2]*S[0]+V[2][3]*S[0]+V[3][2]*S[0]+V[3][3]*S[0]+V[0][0]*V[2][2]-V[0][2]*V[2][0]+V[0][0]*V[2][3]+V[0][0]*V[3][2]-V[0][2]*V[3][0]-V[2][0]*V[0][3]+V[0][0]*V[3][3]-V[0][3]*V[3][0]+G[0]*G[2]+G[0]*G[3]+G[0]*S[1]+G[2]*S[0]+G[3]*S[0]+S[0]*S[1]+(dT*dT)*V[1][1]*V[2][2]-(dT*dT)*V[1][2]*V[2][1]+(dT*dT)*V[1][1]*V[2][3]+(dT*dT)*V[1][1]*V[3][2]-(dT*dT)*V[1][2]*V[3][1]-(dT*dT)*V[2][1]*V[1][3]+(dT*dT)*V[1][1]*V[3][3]-(dT*dT)*V[1][3]*V[3][1]+dT*V[0][1]*G[2]+dT*V[1][0]*G[2]+dT*V[0][1]*G[3]+dT*V[1][0]*G[3]+dT*V[0][1]*S[1]+dT*V[1][0]*S[1]+(dT*dT)*V[1][1]*G[2]+(dT*dT)*V[1][1]*G[3]+(dT*dT)*V[1][1]*S[1]+dT*V[0][1]*V[2][2]+dT*V[1][0]*V[2][2]-dT*V[0][2]*V[2][1]-dT*V[2][0]*V[1][2]+dT*V[0][1]*V[2][3]+dT*V[0][1]*V[3][2]+dT*V[1][0]*V[2][3]+dT*V[1][0]*V[3][2]-dT*V[0][2]*V[3][1]-dT*V[2][0]*V[1][3]-dT*V[0][3]*V[2][1]-dT*V[1][2]*V[3][0]+dT*V[0][1]*V[3][3]+dT*V[1][0]*V[3][3]-dT*V[0][3]*V[3][1]-dT*V[3][0]*V[1][3]);
			K[1][0] = (V[1][0]*G[2]+V[1][0]*G[3]+V[1][0]*S[1]+V[1][0]*V[2][2]-V[2][0]*V[1][2]+V[1][0]*V[2][3]+V[1][0]*V[3][2]-V[2][0]*V[1][3]-V[1][2]*V[3][0]+V[1][0]*V[3][3]-V[3][0]*V[1][3]+(dT*dT)*V[2][1]*V[3][2]-(dT*dT)*V[2][2]*V[3][1]+(dT*dT)*V[2][1]*V[3][3]-(dT*dT)*V[3][1]*V[2][3]+dT*V[1][1]*G[2]+dT*V[2][0]*G[2]+dT*V[1][1]*G[3]+dT*V[2][0]*G[3]+dT*V[1][1]*S[1]+dT*V[2][0]*S[1]+(dT*dT)*V[2][1]*G[2]+(dT*dT)*V[2][1]*G[3]+(dT*dT)*V[2][1]*S[1]+dT*V[1][1]*V[2][2]-dT*V[1][2]*V[2][1]+dT*V[1][1]*V[2][3]+dT*V[1][1]*V[3][2]+dT*V[2][0]*V[3][2]-dT*V[1][2]*V[3][1]-dT*V[2][1]*V[1][3]-dT*V[3][0]*V[2][2]+dT*V[1][1]*V[3][3]+dT*V[2][0]*V[3][3]-dT*V[3][0]*V[2][3]-dT*V[1][3]*V[3][1])/(V[0][0]*G[2]+V[0][0]*G[3]+V[2][2]*G[0]+V[2][3]*G[0]+V[3][2]*G[0]+V[3][3]*G[0]+V[0][0]*S[1]+V[2][2]*S[0]+V[2][3]*S[0]+V[3][2]*S[0]+V[3][3]*S[0]+V[0][0]*V[2][2]-V[0][2]*V[2][0]+V[0][0]*V[2][3]+V[0][0]*V[3][2]-V[0][2]*V[3][0]-V[2][0]*V[0][3]+V[0][0]*V[3][3]-V[0][3]*V[3][0]+G[0]*G[2]+G[0]*G[3]+G[0]*S[1]+G[2]*S[0]+G[3]*S[0]+S[0]*S[1]+(dT*dT)*V[1][1]*V[2][2]-(dT*dT)*V[1][2]*V[2][1]+(dT*dT)*V[1][1]*V[2][3]+(dT*dT)*V[1][1]*V[3][2]-(dT*dT)*V[1][2]*V[3][1]-(dT*dT)*V[2][1]*V[1][3]+(dT*dT)*V[1][1]*V[3][3]-(dT*dT)*V[1][3]*V[3][1]+dT*V[0][1]*G[2]+dT*V[1][0]*G[2]+dT*V[0][1]*G[3]+dT*V[1][0]*G[3]+dT*V[0][1]*S[1]+dT*V[1][0]*S[1]+(dT*dT)*V[1][1]*G[2]+(dT*dT)*V[1][1]*G[3]+(dT*dT)*V[1][1]*S[1]+dT*V[0][1]*V[2][2]+dT*V[1][0]*V[2][2]-dT*V[0][2]*V[2][1]-dT*V[2][0]*V[1][2]+dT*V[0][1]*V[2][3]+dT*V[0][1]*V[3][2]+dT*V[1][0]*V[2][3]+dT*V[1][0]*V[3][2]-dT*V[0][2]*V[3][1]-dT*V[2][0]*V[1][3]-dT*V[0][3]*V[2][1]-dT*V[1][2]*V[3][0]+dT*V[0][1]*V[3][3]+dT*V[1][0]*V[3][3]-dT*V[0][3]*V[3][1]-dT*V[3][0]*V[1][3]);
			K[1][1] = (V[1][2]*G[0]+V[1][3]*G[0]+V[1][2]*S[0]+V[1][3]*S[0]+V[0][0]*V[1][2]-V[1][0]*V[0][2]+V[0][0]*V[1][3]-V[1][0]*V[0][3]+(dT*dT)*V[0][1]*V[2][2]+(dT*dT)*V[1][0]*V[2][2]-(dT*dT)*V[0][2]*V[2][1]-(dT*dT)*V[2][0]*V[1][2]+(dT*dT)*V[0][1]*V[2][3]+(dT*dT)*V[1][0]*V[2][3]-(dT*dT)*V[2][0]*V[1][3]-(dT*dT)*V[0][3]*V[2][1]+(dT*dT*dT)*V[1][1]*V[2][2]-(dT*dT*dT)*V[1][2]*V[2][1]+(dT*dT*dT)*V[1][1]*V[2][3]-(dT*dT*dT)*V[2][1]*V[1][3]+dT*V[2][2]*G[0]+dT*V[2][3]*G[0]+dT*V[2][2]*S[0]+dT*V[2][3]*S[0]+dT*V[0][0]*V[2][2]+dT*V[0][1]*V[1][2]-dT*V[0][2]*V[1][1]-dT*V[0][2]*V[2][0]+dT*V[0][0]*V[2][3]+dT*V[0][1]*V[1][3]-dT*V[1][1]*V[0][3]-dT*V[2][0]*V[0][3])/(V[0][0]*G[2]+V[0][0]*G[3]+V[2][2]*G[0]+V[2][3]*G[0]+V[3][2]*G[0]+V[3][3]*G[0]+V[0][0]*S[1]+V[2][2]*S[0]+V[2][3]*S[0]+V[3][2]*S[0]+V[3][3]*S[0]+V[0][0]*V[2][2]-V[0][2]*V[2][0]+V[0][0]*V[2][3]+V[0][0]*V[3][2]-V[0][2]*V[3][0]-V[2][0]*V[0][3]+V[0][0]*V[3][3]-V[0][3]*V[3][0]+G[0]*G[2]+G[0]*G[3]+G[0]*S[1]+G[2]*S[0]+G[3]*S[0]+S[0]*S[1]+(dT*dT)*V[1][1]*V[2][2]-(dT*dT)*V[1][2]*V[2][1]+(dT*dT)*V[1][1]*V[2][3]+(dT*dT)*V[1][1]*V[3][2]-(dT*dT)*V[1][2]*V[3][1]-(dT*dT)*V[2][1]*V[1][3]+(dT*dT)*V[1][1]*V[3][3]-(dT*dT)*V[1][3]*V[3][1]+dT*V[0][1]*G[2]+dT*V[1][0]*G[2]+dT*V[0][1]*G[3]+dT*V[1][0]*G[3]+dT*V[0][1]*S[1]+dT*V[1][0]*S[1]+(dT*dT)*V[1][1]*G[2]+(dT*dT)*V[1][1]*G[3]+(dT*dT)*V[1][1]*S[1]+dT*V[0][1]*V[2][2]+dT*V[1][0]*V[2][2]-dT*V[0][2]*V[2][1]-dT*V[2][0]*V[1][2]+dT*V[0][1]*V[2][3]+dT*V[0][1]*V[3][2]+dT*V[1][0]*V[2][3]+dT*V[1][0]*V[3][2]-dT*V[0][2]*V[3][1]-dT*V[2][0]*V[1][3]-dT*V[0][3]*V[2][1]-dT*V[1][2]*V[3][0]+dT*V[0][1]*V[3][3]+dT*V[1][0]*V[3][3]-dT*V[0][3]*V[3][1]-dT*V[3][0]*V[1][3]);
			K[2][0] = (V[2][0]*G[3]-V[3][0]*G[2]+V[2][0]*S[1]+V[2][0]*V[3][2]-V[3][0]*V[2][2]+V[2][0]*V[3][3]-V[3][0]*V[2][3]+dT*V[2][1]*G[3]-dT*V[3][1]*G[2]+dT*V[2][1]*S[1]+dT*V[2][1]*V[3][2]-dT*V[2][2]*V[3][1]+dT*V[2][1]*V[3][3]-dT*V[3][1]*V[2][3])/(V[0][0]*G[2]+V[0][0]*G[3]+V[2][2]*G[0]+V[2][3]*G[0]+V[3][2]*G[0]+V[3][3]*G[0]+V[0][0]*S[1]+V[2][2]*S[0]+V[2][3]*S[0]+V[3][2]*S[0]+V[3][3]*S[0]+V[0][0]*V[2][2]-V[0][2]*V[2][0]+V[0][0]*V[2][3]+V[0][0]*V[3][2]-V[0][2]*V[3][0]-V[2][0]*V[0][3]+V[0][0]*V[3][3]-V[0][3]*V[3][0]+G[0]*G[2]+G[0]*G[3]+G[0]*S[1]+G[2]*S[0]+G[3]*S[0]+S[0]*S[1]+(dT*dT)*V[1][1]*V[2][2]-(dT*dT)*V[1][2]*V[2][1]+(dT*dT)*V[1][1]*V[2][3]+(dT*dT)*V[1][1]*V[3][2]-(dT*dT)*V[1][2]*V[3][1]-(dT*dT)*V[2][1]*V[1][3]+(dT*dT)*V[1][1]*V[3][3]-(dT*dT)*V[1][3]*V[3][1]+dT*V[0][1]*G[2]+dT*V[1][0]*G[2]+dT*V[0][1]*G[3]+dT*V[1][0]*G[3]+dT*V[0][1]*S[1]+dT*V[1][0]*S[1]+(dT*dT)*V[1][1]*G[2]+(dT*dT)*V[1][1]*G[3]+(dT*dT)*V[1][1]*S[1]+dT*V[0][1]*V[2][2]+dT*V[1][0]*V[2][2]-dT*V[0][2]*V[2][1]-dT*V[2][0]*V[1][2]+dT*V[0][1]*V[2][3]+dT*V[0][1]*V[3][2]+dT*V[1][0]*V[2][3]+dT*V[1][0]*V[3][2]-dT*V[0][2]*V[3][1]-dT*V[2][0]*V[1][3]-dT*V[0][3]*V[2][1]-dT*V[1][2]*V[3][0]+dT*V[0][1]*V[3][3]+dT*V[1][0]*V[3][3]-dT*V[0][3]*V[3][1]-dT*V[3][0]*V[1][3]);
			K[2][1] = (V[0][0]*G[2]+V[2][2]*G[0]+V[2][3]*G[0]+V[2][2]*S[0]+V[2][3]*S[0]+V[0][0]*V[2][2]-V[0][2]*V[2][0]+V[0][0]*V[2][3]-V[2][0]*V[0][3]+G[0]*G[2]+G[2]*S[0]+(dT*dT)*V[1][1]*V[2][2]-(dT*dT)*V[1][2]*V[2][1]+(dT*dT)*V[1][1]*V[2][3]-(dT*dT)*V[2][1]*V[1][3]+dT*V[0][1]*G[2]+dT*V[1][0]*G[2]+(dT*dT)*V[1][1]*G[2]+dT*V[0][1]*V[2][2]+dT*V[1][0]*V[2][2]-dT*V[0][2]*V[2][1]-dT*V[2][0]*V[1][2]+dT*V[0][1]*V[2][3]+dT*V[1][0]*V[2][3]-dT*V[2][0]*V[1][3]-dT*V[0][3]*V[2][1])/(V[0][0]*G[2]+V[0][0]*G[3]+V[2][2]*G[0]+V[2][3]*G[0]+V[3][2]*G[0]+V[3][3]*G[0]+V[0][0]*S[1]+V[2][2]*S[0]+V[2][3]*S[0]+V[3][2]*S[0]+V[3][3]*S[0]+V[0][0]*V[2][2]-V[0][2]*V[2][0]+V[0][0]*V[2][3]+V[0][0]*V[3][2]-V[0][2]*V[3][0]-V[2][0]*V[0][3]+V[0][0]*V[3][3]-V[0][3]*V[3][0]+G[0]*G[2]+G[0]*G[3]+G[0]*S[1]+G[2]*S[0]+G[3]*S[0]+S[0]*S[1]+(dT*dT)*V[1][1]*V[2][2]-(dT*dT)*V[1][2]*V[2][1]+(dT*dT)*V[1][1]*V[2][3]+(dT*dT)*V[1][1]*V[3][2]-(dT*dT)*V[1][2]*V[3][1]-(dT*dT)*V[2][1]*V[1][3]+(dT*dT)*V[1][1]*V[3][3]-(dT*dT)*V[1][3]*V[3][1]+dT*V[0][1]*G[2]+dT*V[1][0]*G[2]+dT*V[0][1]*G[3]+dT*V[1][0]*G[3]+dT*V[0][1]*S[1]+dT*V[1][0]*S[1]+(dT*dT)*V[1][1]*G[2]+(dT*dT)*V[1][1]*G[3]+(dT*dT)*V[1][1]*S[1]+dT*V[0][1]*V[2][2]+dT*V[1][0]*V[2][2]-dT*V[0][2]*V[2][1]-dT*V[2][0]*V[1][2]+dT*V[0][1]*V[2][3]+dT*V[0][1]*V[3][2]+dT*V[1][0]*V[2][3]+dT*V[1][0]*V[3][2]-dT*V[0][2]*V[3][1]-dT*V[2][0]*V[1][3]-dT*V[0][3]*V[2][1]-dT*V[1][2]*V[3][0]+dT*V[0][1]*V[3][3]+dT*V[1][0]*V[3][3]-dT*V[0][3]*V[3][1]-dT*V[3][0]*V[1][3]);
			K[3][0] = (-V[2][0]*G[3]+V[3][0]*G[2]+V[3][0]*S[1]-V[2][0]*V[3][2]+V[3][0]*V[2][2]-V[2][0]*V[3][3]+V[3][0]*V[2][3]-dT*V[2][1]*G[3]+dT*V[3][1]*G[2]+dT*V[3][1]*S[1]-dT*V[2][1]*V[3][2]+dT*V[2][2]*V[3][1]-dT*V[2][1]*V[3][3]+dT*V[3][1]*V[2][3])/(V[0][0]*G[2]+V[0][0]*G[3]+V[2][2]*G[0]+V[2][3]*G[0]+V[3][2]*G[0]+V[3][3]*G[0]+V[0][0]*S[1]+V[2][2]*S[0]+V[2][3]*S[0]+V[3][2]*S[0]+V[3][3]*S[0]+V[0][0]*V[2][2]-V[0][2]*V[2][0]+V[0][0]*V[2][3]+V[0][0]*V[3][2]-V[0][2]*V[3][0]-V[2][0]*V[0][3]+V[0][0]*V[3][3]-V[0][3]*V[3][0]+G[0]*G[2]+G[0]*G[3]+G[0]*S[1]+G[2]*S[0]+G[3]*S[0]+S[0]*S[1]+(dT*dT)*V[1][1]*V[2][2]-(dT*dT)*V[1][2]*V[2][1]+(dT*dT)*V[1][1]*V[2][3]+(dT*dT)*V[1][1]*V[3][2]-(dT*dT)*V[1][2]*V[3][1]-(dT*dT)*V[2][1]*V[1][3]+(dT*dT)*V[1][1]*V[3][3]-(dT*dT)*V[1][3]*V[3][1]+dT*V[0][1]*G[2]+dT*V[1][0]*G[2]+dT*V[0][1]*G[3]+dT*V[1][0]*G[3]+dT*V[0][1]*S[1]+dT*V[1][0]*S[1]+(dT*dT)*V[1][1]*G[2]+(dT*dT)*V[1][1]*G[3]+(dT*dT)*V[1][1]*S[1]+dT*V[0][1]*V[2][2]+dT*V[1][0]*V[2][2]-dT*V[0][2]*V[2][1]-dT*V[2][0]*V[1][2]+dT*V[0][1]*V[2][3]+dT*V[0][1]*V[3][2]+dT*V[1][0]*V[2][3]+dT*V[1][0]*V[3][2]-dT*V[0][2]*V[3][1]-dT*V[2][0]*V[1][3]-dT*V[0][3]*V[2][1]-dT*V[1][2]*V[3][0]+dT*V[0][1]*V[3][3]+dT*V[1][0]*V[3][3]-dT*V[0][3]*V[3][1]-dT*V[3][0]*V[1][3]);
			K[3][1] = (V[0][0]*G[3]+V[3][2]*G[0]+V[3][3]*G[0]+V[3][2]*S[0]+V[3][3]*S[0]+V[0][0]*V[3][2]-V[0][2]*V[3][0]+V[0][0]*V[3][3]-V[0][3]*V[3][0]+G[0]*G[3]+G[3]*S[0]+(dT*dT)*V[1][1]*V[3][2]-(dT*dT)*V[1][2]*V[3][1]+(dT*dT)*V[1][1]*V[3][3]-(dT*dT)*V[1][3]*V[3][1]+dT*V[0][1]*G[3]+dT*V[1][0]*G[3]+(dT*dT)*V[1][1]*G[3]+dT*V[0][1]*V[3][2]+dT*V[1][0]*V[3][2]-dT*V[0][2]*V[3][1]-dT*V[1][2]*V[3][0]+dT*V[0][1]*V[3][3]+dT*V[1][0]*V[3][3]-dT*V[0][3]*V[3][1]-dT*V[3][0]*V[1][3])/(V[0][0]*G[2]+V[0][0]*G[3]+V[2][2]*G[0]+V[2][3]*G[0]+V[3][2]*G[0]+V[3][3]*G[0]+V[0][0]*S[1]+V[2][2]*S[0]+V[2][3]*S[0]+V[3][2]*S[0]+V[3][3]*S[0]+V[0][0]*V[2][2]-V[0][2]*V[2][0]+V[0][0]*V[2][3]+V[0][0]*V[3][2]-V[0][2]*V[3][0]-V[2][0]*V[0][3]+V[0][0]*V[3][3]-V[0][3]*V[3][0]+G[0]*G[2]+G[0]*G[3]+G[0]*S[1]+G[2]*S[0]+G[3]*S[0]+S[0]*S[1]+(dT*dT)*V[1][1]*V[2][2]-(dT*dT)*V[1][2]*V[2][1]+(dT*dT)*V[1][1]*V[2][3]+(dT*dT)*V[1][1]*V[3][2]-(dT*dT)*V[1][2]*V[3][1]-(dT*dT)*V[2][1]*V[1][3]+(dT*dT)*V[1][1]*V[3][3]-(dT*dT)*V[1][3]*V[3][1]+dT*V[0][1]*G[2]+dT*V[1][0]*G[2]+dT*V[0][1]*G[3]+dT*V[1][0]*G[3]+dT*V[0][1]*S[1]+dT*V[1][0]*S[1]+(dT*dT)*V[1][1]*G[2]+(dT*dT)*V[1][1]*G[3]+(dT*dT)*V[1][1]*S[1]+dT*V[0][1]*V[2][2]+dT*V[1][0]*V[2][2]-dT*V[0][2]*V[2][1]-dT*V[2][0]*V[1][2]+dT*V[0][1]*V[2][3]+dT*V[0][1]*V[3][2]+dT*V[1][0]*V[2][3]+dT*V[1][0]*V[3][2]-dT*V[0][2]*V[3][1]-dT*V[2][0]*V[1][3]-dT*V[0][3]*V[2][1]-dT*V[1][2]*V[3][0]+dT*V[0][1]*V[3][3]+dT*V[1][0]*V[3][3]-dT*V[0][3]*V[3][1]-dT*V[3][0]*V[1][3]);

			z_new[0] = -K[0][0]*(dT*z[1]-x[0]+z[0])+dT*z[1]-K[0][1]*(-x[1]+z[2]+z[3])+z[0];
			z_new[1] = -K[1][0]*(dT*z[1]-x[0]+z[0])+dT*z[2]-K[1][1]*(-x[1]+z[2]+z[3])+z[1];
			z_new[2] = -K[2][0]*(dT*z[1]-x[0]+z[0])-K[2][1]*(-x[1]+z[2]+z[3])+z[2];
			z_new[3] = -K[3][0]*(dT*z[1]-x[0]+z[0])-K[3][1]*(-x[1]+z[2]+z[3])+z[3];

			memcpy(z, z_new, sizeof(z_new));

			V[0][0] = -K[0][1]*P[2][0]-K[0][1]*P[3][0]-P[0][0]*(K[0][0]-1.0f);
			V[0][1] = -K[0][1]*P[2][1]-K[0][1]*P[3][2]-P[0][1]*(K[0][0]-1.0f);
			V[0][2] = -K[0][1]*P[2][2]-K[0][1]*P[3][2]-P[0][2]*(K[0][0]-1.0f);
			V[0][3] = -K[0][1]*P[2][3]-K[0][1]*P[3][3]-P[0][3]*(K[0][0]-1.0f);
			V[1][0] = P[1][0]-K[1][0]*P[0][0]-K[1][1]*P[2][0]-K[1][1]*P[3][0];
			V[1][1] = P[1][1]-K[1][0]*P[0][1]-K[1][1]*P[2][1]-K[1][1]*P[3][2];
			V[1][2] = P[1][2]-K[1][0]*P[0][2]-K[1][1]*P[2][2]-K[1][1]*P[3][2];
			V[1][3] = P[1][3]-K[1][0]*P[0][3]-K[1][1]*P[2][3]-K[1][1]*P[3][3];
			V[2][0] = -K[2][0]*P[0][0]-K[2][1]*P[3][0]-P[2][0]*(K[2][1]-1.0f);
			V[2][1] = -K[2][0]*P[0][1]-K[2][1]*P[3][2]-P[2][1]*(K[2][1]-1.0f);
			V[2][2] = -K[2][0]*P[0][2]-K[2][1]*P[3][2]-P[2][2]*(K[2][1]-1.0f);
			V[2][3] = -K[2][0]*P[0][3]-K[2][1]*P[3][3]-P[2][3]*(K[2][1]-1.0f);
			V[3][0] = -K[3][0]*P[0][0]-K[3][1]*P[2][0]-P[3][0]*(K[3][1]-1.0f);
			V[3][1] = -K[3][0]*P[0][1]-K[3][1]*P[2][1]-P[3][2]*(K[3][1]-1.0f);
			V[3][2] = -K[3][0]*P[0][2]-K[3][1]*P[2][2]-P[3][2]*(K[3][1]-1.0f);
			V[3][3] = -K[3][0]*P[0][3]-K[3][1]*P[2][3]-P[3][3]*(K[3][1]-1.0f);

			AltHoldSmoothedData altHold;
			AltHoldSmoothedGet(&altHold);
			altHold.Altitude = z[0];
			altHold.Velocity = z[1];
			altHold.Accel = z[2];
			AltHoldSmoothedSet(&altHold);

			if (isnan(altHold.Altitude) || isnan(altHold.Velocity) || isnan(altHold.Accel)) {
				state = AH_RESET;
				AlarmsSet(SYSTEMALARMS_ALARM_ALTITUDEHOLD, SYSTEMALARMS_ALARM_CRITICAL);
			} else
				AlarmsClear(SYSTEMALARMS_ALARM_ALTITUDEHOLD);


			// Verify that we are  in altitude hold mode
			FlightStatusData flightStatus;
			FlightStatusGet(&flightStatus);
			if(flightStatus.FlightMode != FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD) {
				engaged = false;
			}

			if (!engaged)
				continue;

			// Compute the altitude error
			error = (starting_altitude + altitudeHoldDesired.Altitude) - altHold.Altitude;

			// Compute integral off altitude error
			throttleIntegral += error * altitudeHoldSettings.Ki * dT;

			// Only update stabilizationDesired less frequently
			if((this_time_ms - last_update_time_ms) < 20)
				continue;

			last_update_time_ms = this_time_ms;

			// Instead of explicit limit on integral you output limit feedback
			StabilizationDesiredGet(&stabilizationDesired);
			stabilizationDesired.Throttle = error * altitudeHoldSettings.Kp + throttleIntegral -
			altHold.Velocity * altitudeHoldSettings.Kd - altHold.Accel * altitudeHoldSettings.Ka;
			if(stabilizationDesired.Throttle > 1) {
				throttleIntegral -= (stabilizationDesired.Throttle - 1);
				stabilizationDesired.Throttle = 1;
			}
			else if (stabilizationDesired.Throttle < 0) {
				throttleIntegral -= stabilizationDesired.Throttle;
				stabilizationDesired.Throttle = 0;
			}

			stabilizationDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDEPLUS;
			stabilizationDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDEPLUS;
			stabilizationDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK;
			stabilizationDesired.Roll = altitudeHoldDesired.Roll;
			stabilizationDesired.Pitch = altitudeHoldDesired.Pitch;
			stabilizationDesired.Yaw = altitudeHoldDesired.Yaw;
			StabilizationDesiredSet(&stabilizationDesired);

		} else if (ev.obj == AltitudeHoldDesiredHandle()) {
			AltitudeHoldDesiredGet(&altitudeHoldDesired);
		} else if (ev.obj == AltitudeHoldSettingsHandle()) {
			AltitudeHoldSettingsGet(&altitudeHoldSettings);
		}

	}
}

/**
 * @}
 * @}
 */

