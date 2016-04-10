/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup AutotuningModule Autotuning Module
 * @{
 *
 * @file       autotune.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2016
 * @brief      State machine to run autotuning. Low level work done by @ref
 *             StabilizationModule 
 *
 * @see        The GNU Public License (GPL) Version 3
 *
 ******************************************************************************/
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
#include "pios.h"
#include "physical_constants.h"
#include "flightstatus.h"
#include "modulesettings.h"
#include "manualcontrolcommand.h"
#include "manualcontrolsettings.h"
#include "gyros.h"
#include "actuatordesired.h"
#include "stabilizationdesired.h"
#include "stabilizationsettings.h"
#include "systemident.h"

// The actually system ident code
#include "rate_torque_si.h"

#include <pios_board_info.h>
#include "pios_thread.h"

// Private constants
#define STACK_SIZE_BYTES 1504
#define TASK_PRIORITY PIOS_THREAD_PRIO_NORMAL

// Private types
enum AUTOTUNE_STATE {AT_INIT, AT_START, AT_RUN, AT_FINISHED, AT_SET};

// Private variables
static struct pios_thread *taskHandle;
static bool module_enabled;

// Private functions
static void AutotuneTask(void *parameters);

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t AutotuneInitialize(void)
{
	// Create a queue, connect to manual control command and flightstatus
#ifdef MODULE_Autotune_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_AUTOTUNE] == MODULESETTINGS_ADMINSTATE_ENABLED)
		module_enabled = true;
	else
		module_enabled = false;
#endif

	if (module_enabled) {
		SystemIdentInitialize();
	}

	return 0;
}

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t AutotuneStart(void)
{
	// Start main task if it is enabled
	if(module_enabled) {
		// Watchdog must be registered before starting task
		PIOS_WDG_RegisterFlag(PIOS_WDG_AUTOTUNE);

		// Start main task
		taskHandle = PIOS_Thread_Create(AutotuneTask, "Autotune", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);

		TaskMonitorAdd(TASKINFO_RUNNING_AUTOTUNE, taskHandle);
	}
	return 0;
}

MODULE_INITCALL(AutotuneInitialize, AutotuneStart)

static void UpdateSystemIdent(uintptr_t rtsi_handle, const float *noise,
		float dT_s, uint32_t predicts) {
	SystemIdentData relay;
	rtsi_get_gains(rtsi_handle, relay.Beta);
	rtsi_get_tau(rtsi_handle, &relay.Tau);
	rtsi_get_bias(rtsi_handle, relay.Bias);
	if (noise) {
		relay.Noise[SYSTEMIDENT_NOISE_ROLL]  = noise[0];
		relay.Noise[SYSTEMIDENT_NOISE_PITCH] = noise[1];
		relay.Noise[SYSTEMIDENT_NOISE_YAW]   = noise[2];
	}
	relay.Period = dT_s * 1000.0f;

	relay.NumAfPredicts = predicts;
	SystemIdentSet(&relay);
}

static void UpdateStabilizationDesired(bool doingIdent) {
	StabilizationDesiredData stabDesired;
	StabilizationDesiredGet(&stabDesired);

	uint8_t rollMax, pitchMax;

	float manualRate[STABILIZATIONSETTINGS_MANUALRATE_NUMELEM];

	StabilizationSettingsRollMaxGet(&rollMax);
	StabilizationSettingsPitchMaxGet(&pitchMax);
	StabilizationSettingsManualRateGet(manualRate);

	ManualControlCommandRollGet(&stabDesired.Roll);
	stabDesired.Roll *= rollMax;
	ManualControlCommandPitchGet(&stabDesired.Pitch);
	stabDesired.Pitch *= pitchMax;

	ManualControlCommandYawGet(&stabDesired.Yaw);
	stabDesired.Yaw *= manualRate[STABILIZATIONSETTINGS_MANUALRATE_YAW];

	if (doingIdent) {
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL]  = STABILIZATIONDESIRED_STABILIZATIONMODE_SYSTEMIDENT;
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_SYSTEMIDENT;
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_SYSTEMIDENT;
	} else {
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL]  = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_RATE;
	}

	ManualControlCommandThrottleGet(&stabDesired.Throttle);

	StabilizationDesiredSet(&stabDesired);
}

/**
 * Module thread, should not return.
 */
static void AutotuneTask(void *parameters)
{
	enum AUTOTUNE_STATE state = AT_INIT;

	uint32_t lastUpdateTime = PIOS_Thread_Systime();

	float noise[3] = {0};

	uintptr_t rtsi_handle;
	rtsi_alloc(&rtsi_handle);

	uint32_t last_time = 0.0f;
	const uint32_t DT_MS = 3;

	while(1) {

		PIOS_WDG_UpdateFlag(PIOS_WDG_AUTOTUNE);
		// TODO:
		// 1. get from queue
		// 2. based on whether it is flightstatus or manualcontrol

		uint32_t diffTime;

		const uint32_t PREPARE_TIME = 2000;
		const uint32_t MEASURE_TIME = 60000;

		static uint32_t updateCounter = 0;

		bool doingIdent = false;

		FlightStatusData flightStatus;
		FlightStatusGet(&flightStatus);

		// Only allow this module to run when autotuning
		if (flightStatus.FlightMode != FLIGHTSTATUS_FLIGHTMODE_AUTOTUNE) {
			state = AT_INIT;
			PIOS_Thread_Sleep(50);
			continue;
		}

		float throttle;

		ManualControlCommandThrottleGet(&throttle);
				
		switch(state) {
			case AT_INIT:

				lastUpdateTime = PIOS_Thread_Systime();

				// Only start when armed and flying
				if (flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMED && throttle > 0) {

					rtsi_init(rtsi_handle);
					UpdateSystemIdent(rtsi_handle, NULL, 0.0f, 0);

					state = AT_START;

				}
				break;

			case AT_START:

				diffTime = PIOS_Thread_Systime() - lastUpdateTime;

				// Spend the first block of time in normal rate mode to get airborne
				if (diffTime > PREPARE_TIME) {
					state = AT_RUN;
					lastUpdateTime = PIOS_Thread_Systime();
				}


				last_time = PIOS_DELAY_GetRaw();

				updateCounter = 0;

				break;

			case AT_RUN:

				diffTime = PIOS_Thread_Systime() - lastUpdateTime;

				doingIdent = true;

				// Update the system identification, but only when throttle is applied
				// so bad values don't result when landing
				if (throttle > 0) {
					float y[3];
					GyrosxGet(y+0);
					GyrosyGet(y+1);
					GyroszGet(y+2);

					float u[3];
					ActuatorDesiredRollGet(u+0);
					ActuatorDesiredPitchGet(u+1);
					ActuatorDesiredYawGet(u+2);

					float dT_s = PIOS_DELAY_DiffuS(last_time) * 1.0e-6f;

					rtsi_predict(rtsi_handle, u, y, DT_MS * 0.001f);

					// Get current rate estimates to track noise around that
					float X[3];
					rtsi_get_rates(rtsi_handle, X);

					for (uint32_t i = 0; i < 3; i++) {
						const float NOISE_ALPHA = 0.9997f;  // 10 second time constant at 300 Hz
						noise[i] = NOISE_ALPHA * noise[i] + (1-NOISE_ALPHA) * (y[i] - X[i]) * (y[i] - X[i]);
					}

					// Update uavo every 256 cycles to avoid
					// telemetry spam
					if (!((updateCounter++) & 0xff)) {
						UpdateSystemIdent(rtsi_handle, noise, dT_s, updateCounter);
					}
				}

				if (diffTime > MEASURE_TIME) { // Move on to next state
					state = AT_FINISHED;
					lastUpdateTime = PIOS_Thread_Systime();
				}

				last_time = PIOS_DELAY_GetRaw();

				break;

			case AT_FINISHED:

				// Wait until disarmed and landed before saving the settings

				UpdateSystemIdent(rtsi_handle, noise, 0, updateCounter);
				if (flightStatus.Armed == FLIGHTSTATUS_ARMED_DISARMED && throttle <= 0)
					state = AT_SET;

				break;

			case AT_SET:
				// If at some point we want to store the settings at the end of
				// autotune, that can be done here. However, that will await further
				// testing.

				// Save the settings locally. Note this is done after disarming.
				UAVObjSave(SystemIdentHandle(), 0);
				state = AT_INIT;
				break;

			default:
				// Set an alarm or some shit like that
				break;
		}

		// Update based on manual controls
		UpdateStabilizationDesired(doingIdent);

		PIOS_Thread_Sleep(DT_MS);
	}
}

/**
 * @}
 * @}
 */
