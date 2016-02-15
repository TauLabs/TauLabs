/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup DacBeepCodes Dac Beep Codes Module
 * @{
 *
 * @file       dacbeepcodes.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015-2016
 * @brief      Based on flight status (e.g. RSSI/Battery) module beeps
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
#include "pios_thread.h"
#include "pios_dacbeep_priv.h"

#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "manualcontrolcommand.h"

#define STACK_SIZE_BYTES 648
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW

// Private variables
static struct pios_thread *dacBeepCodeTaskHandle;
static bool module_enabled;

// Private functions
static void dacBeepCodeTask(void *parameters);

// External variables
extern uintptr_t dacbeep_handle;

 /**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t DacBeepCodesStart()
{
	// Start main task if it is enabled
	if (module_enabled) {
		dacBeepCodeTaskHandle = PIOS_Thread_Create(dacBeepCodeTask, "DacBeepCodes", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
		TaskMonitorAdd(TASKINFO_RUNNING_ALTITUDEHOLD, dacBeepCodeTaskHandle);
		return 0;
	}
	return -1;

}

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t DacBeepCodesInitialize()
{
	if (dacbeep_handle)
		module_enabled = true;

	if(module_enabled) {

		return 0;
	}

	return -1;
}

MODULE_INITCALL(DacBeepCodesInitialize, DacBeepCodesStart);

/**
 * Module thread, should not return.
 */
static void dacBeepCodeTask(void *parameters)
{
	while(1) {

		int16_t rssi = 0;
		ManualControlCommandRssiGet(&rssi);
		if (rssi < 0) rssi = 0;
		if (rssi > 100) rssi = 100;

		// Determine frequency from battery voltage
		float volt = 0.0f;
		if (FlightBatteryStateHandle())
			FlightBatteryStateVoltageGet(&volt);

		// We want to map RSSI inversely into pulse duration. At 100% shoot for
		// a period of 800 ms. At 50% get to 150ms. Sharp inflection at 50%.
		// Stops at 100 ms to set max rep rate at 10 Hz
		// Using piecewise linear to avoid exponential calculation
		int32_t period_ms = 100 + ((rssi < 50) ? rssi : (13 * (rssi - 50) + 50));

		// Scale audio pitches based on voltage thresholds in ALARM and WARNING
		// this makes the audio start to change once you hit this range and have
		// a fair bit of dynamic range
		float volt_threshold[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_NUMELEM];
		FlightBatterySettingsVoltageThresholdsGet(volt_threshold);
		const float MIN_VOLT = volt_threshold[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_ALARM];
		const float MAX_VOLT = volt_threshold[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_WARNING];
		const uint16_t MIN_FREQ_HZ = 1500;
		const uint16_t RANGE_FREQ_HZ = 4000;
		if (volt < MIN_VOLT) volt = MIN_VOLT;
		if (volt > MAX_VOLT) volt = MAX_VOLT;
		uint16_t freq = MIN_FREQ_HZ;
		if (MAX_VOLT > MIN_VOLT)
			freq = MIN_FREQ_HZ + (volt - MIN_VOLT) / (MAX_VOLT - MIN_VOLT) * RANGE_FREQ_HZ;

		PIOS_DACBEEP_Beep(dacbeep_handle, freq, 80);
		PIOS_Thread_Sleep(period_ms);
	}
}

