/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup Flight Stats Module
 * @{ 
 *
 * @file       flightstatsmodule.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 * @brief      Collects statistcs during the flight
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
#include "modulesettings.h"
#include "pios_thread.h"

#include "misc_math.h"

#include "airspeedactual.h"
#include "flightbatterystate.h"
#include "flightstatus.h"
#include "flightstats.h"
#include "flightstatssettings.h"
#include "gyros.h"
#include "positionactual.h"
#include "velocityactual.h"

// Private constants
#define STACK_SIZE_BYTES 600
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW

// Private types

// Private variables
static bool module_enabled;
static struct pios_thread *flightStatsTaskHandle;
static volatile FlightStatsSettingsData settings;
static PositionActualData lastPositionActual;
static float initial_consumed_energy;
static float previous_consumed_energy;

// Private functions
static void flightStatsTask(void *parameters);
static bool isArmed();
static void resetStats(FlightStatsData *stats);
static void collectStats(FlightStatsData *stats);

// Local variables

// External variables

/**
 * Initialize the FlightStats module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t FlightStatsModuleInitialize(void)
{
#ifdef MODULE_FLIGHTSTATS_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_FLIGHTSTATS] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (!module_enabled)
		return -1;

	FlightStatsInitialize();
	FlightStatsSettingsInitialize();

	// Get settings and connect
	FlightStatsSettingsConnectCopy(&settings);

	return 0;
}

/**
 * Start the FlightStats module
 * \return -1 if initialization failed
 * \return 0 on success
 */
int32_t FlightStatModuleStart(void)
{
	//Check if module is enabled or not
	if (module_enabled == false) {
		return -1;
	}

	// Start flight stats task
	flightStatsTaskHandle = PIOS_Thread_Create(flightStatsTask, "FlightStats", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);

	TaskMonitorAdd(TASKINFO_RUNNING_FLIGHTSTATS, flightStatsTaskHandle);
	
	return 0;
}

MODULE_INITCALL(FlightStatsModuleInitialize, FlightStatModuleStart);

static void flightStatsTask(void *parameters)
{
	FlightStatsData flightStatsData;
	bool first_run = true;

	resetStats(&flightStatsData);
	flightStatsData.State = FLIGHTSTATS_STATE_IDLE;

	// Loop forever
	while (1) {
		// Update stats at about 10Hz
		PIOS_Thread_Sleep(100);
		switch (flightStatsData.State) {
			case FLIGHTSTATS_STATE_IDLE:
				if (isArmed()) {
					switch (settings.StatsBehavior) {
					case FLIGHTSTATSSETTINGS_STATSBEHAVIOR_RESETONBOOT:
						flightStatsData.State = FLIGHTSTATS_STATE_COLLECTING;
						break;
					case FLIGHTSTATSSETTINGS_STATSBEHAVIOR_RESETONARM:
						flightStatsData.State = FLIGHTSTATS_STATE_RESET;
						break;
					}
					first_run = true;
				}
				break;
			case FLIGHTSTATS_STATE_RESET:
				resetStats(&flightStatsData);
				flightStatsData.State = FLIGHTSTATS_STATE_COLLECTING;
				break;
			case FLIGHTSTATS_STATE_COLLECTING:
				if (first_run) { // get some initial values
					// initial position
					PositionActualGet(&lastPositionActual);

					// get the initial battery voltage and consumed energy
					if (FlightBatteryStateHandle()) {
						FlightBatteryStateConsumedEnergyGet(&initial_consumed_energy);

						// either start a new calculation of consumed energy, or combine with data
						// from previous flight
						if (settings.StatsBehavior == FLIGHTSTATSSETTINGS_STATSBEHAVIOR_RESETONARM) {
							previous_consumed_energy = 0.f;
						}
						else {
							previous_consumed_energy = flightStatsData.ConsumedEnergy;
						}

						// only get the initial voltage if we reset on arm or if it is uninitialized
						if ((settings.StatsBehavior == FLIGHTSTATSSETTINGS_STATSBEHAVIOR_RESETONARM)\
							|| (flightStatsData.InitialBatteryVoltage == 0)){
							float voltage;
							FlightBatteryStateVoltageGet(&voltage);
							flightStatsData.InitialBatteryVoltage = roundf(1000.f * voltage);
						}
					}
					first_run = false;
				}
				collectStats(&flightStatsData);
				if (!isArmed()) {
					flightStatsData.State = FLIGHTSTATS_STATE_IDLE;
				}
				FlightStatsSet(&flightStatsData);
				break;
		}
	}
}

/**
 * Check whether FC is armed
 * \return true if armed
 */
static bool isArmed()
{
	uint8_t arm_status;
	FlightStatusArmedGet(&arm_status);
	return (arm_status == FLIGHTSTATUS_ARMED_ARMED);
}

/**
 * Reset the statistics
 */
static void resetStats(FlightStatsData *stats)
{
	// reset everything
	memset((void*)stats, 0, sizeof(FlightStatsData));
}

/**
 * Collect the statistics
 */
static void collectStats(FlightStatsData *stats)
{
	float tmp;
	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	// Total (horizontal) distance
	stats->DistanceTravelled  += sqrtf(powf(positionActual.North - lastPositionActual.North, 2.f) \
									   + powf(positionActual.East - lastPositionActual.East, 2.f));

	// Max distance to home
	tmp =  sqrtf(powf(positionActual.North, 2.f) + powf(positionActual.East, 2.f));
	stats->MaxDistanceToHome = MAX(stats->MaxDistanceToHome, tmp);

	// Max altitude
	stats->MaxAltitude = MAX(stats->MaxAltitude, -1.f * positionActual.Down);

	VelocityActualData velocityActual;
	VelocityActualGet(&velocityActual);

	// Max groundspeed
	tmp =  sqrtf(powf(velocityActual.North, 2.f) + powf(velocityActual.East, 2.f));
	stats->MaxGroundSpeed = MAX(stats->MaxGroundSpeed, tmp);

	// Max climb rate
	stats->MaxClimbRate = MAX(stats->MaxClimbRate, -1.f * velocityActual.Down);

	// Max descent rate
	stats->MaxDescentRate = MAX(stats->MaxDescentRate, velocityActual.Down);

	// Max airspeed
	if (AirspeedActualHandle()) {
		AirspeedActualTrueAirspeedGet(&tmp);
		stats->MaxAirSpeed = MAX(stats->MaxAirSpeed, tmp);
	}

	// Max roll/pitch/yaw rates
	GyrosData gyros;
	GyrosGet(&gyros);
	stats->MaxPitchRate = MAX(stats->MaxPitchRate, fabsf(gyros.y));
	stats->MaxRollRate = MAX(stats->MaxRollRate, fabsf(gyros.x));
	stats->MaxYawRate = MAX(stats->MaxYawRate, fabsf(gyros.z));

	// Consumed energy
	if (FlightBatteryStateHandle()) {
		FlightBatteryStateConsumedEnergyGet(&tmp);
		stats->ConsumedEnergy = previous_consumed_energy + tmp - initial_consumed_energy;
	}

	// update things for next call
	lastPositionActual = positionActual;
}
