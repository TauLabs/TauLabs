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
static FlightStatsSettingsData settings;
static PositionActualData lastPositionActual;
static float initial_consumed_energy;

// Private functions
static void flightStatsTask(void *parameters);
static void settingsUpdatedCb(UAVObjEvent * ev);
static bool isArmed();
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

	// Get settings and connect callback
	FlightStatsSettingsGet(&settings);
	FlightStatsSettingsConnectCallback(settingsUpdatedCb);

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
	bool first_init = true;
	FlightStatsData flightStatsData;

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
							if (first_init) {
								flightStatsData.State = FLIGHTSTATS_STATE_RESET;
							}
							else {
								flightStatsData.State = FLIGHTSTATS_STATE_INITIALIZING;
							}
							break;
						case FLIGHTSTATSSETTINGS_STATSBEHAVIOR_RESETONARM:
							flightStatsData.State = FLIGHTSTATS_STATE_RESET;
							break;
					}
				}
				break;
			case FLIGHTSTATS_STATE_RESET:
				memset((void*)&flightStatsData, 0, sizeof(flightStatsData));
				flightStatsData.State = FLIGHTSTATS_STATE_INITIALIZING;
				break;
			case FLIGHTSTATS_STATE_INITIALIZING:
				PositionActualGet(&lastPositionActual);
				if (first_init || settings.StatsBehavior == FLIGHTSTATSSETTINGS_STATSBEHAVIOR_RESETONARM) {
					if (FlightBatteryStateHandle()) {
						float voltage;
						FlightBatteryStateVoltageGet(&voltage);
						flightStatsData.InitialBatteryVoltage = roundf(1000.f * voltage);
						FlightBatteryStateConsumedEnergyGet(&initial_consumed_energy);
					}
					first_init = false;
				}
				flightStatsData.State = FLIGHTSTATS_STATE_COLLECTING;
				break;
			case FLIGHTSTATS_STATE_COLLECTING:
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
 * Update the settings
 */
static void settingsUpdatedCb(UAVObjEvent * ev)
{
	FlightStatsSettingsGet(&settings);
}

/**
 * Check whether FC is armed
 * \return true if armed
 */
static bool isArmed()
{
	uint8_t arm_status;
	FlightStatusArmedGet(&arm_status);
	if (arm_status == FLIGHTSTATUS_ARMED_ARMED) {
		return true;
	}
	else {
		return false;
	}
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

	// Max descend rate
	stats->MaxDescendRate = MAX(stats->MaxDescendRate, velocityActual.Down);

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
		stats->ConsumedEnergy = tmp - initial_consumed_energy;
	}

	// update things for next call
	lastPositionActual = positionActual;
}
