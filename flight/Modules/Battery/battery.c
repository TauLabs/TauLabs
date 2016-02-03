/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup BatteryModule Battery Module
 * @{
 *
 * @file       battery.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      Module to read the battery Voltage and Current periodically and set alarms appropriately.
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

#include "flightbatterystate.h"
#include "flightbatterysettings.h"
#include "modulesettings.h"
#include "pios_thread.h"

// ****************
// Private constants
#define STACK_SIZE_BYTES            576
#define TASK_PRIORITY               PIOS_THREAD_PRIO_LOW
#define SAMPLE_PERIOD_MS            500
// Private types

// Private variables
static bool module_enabled = false;
static struct pios_thread *batteryTaskHandle;
static int8_t voltageADCPin = -1; //ADC pin for voltage
static int8_t currentADCPin = -1; //ADC pin for current

// ****************
// Private functions
static void batteryTask(void * parameters);
static void settingsUpdatedCb(UAVObjEvent * objEv);;

static int32_t BatteryStart(void)
{
	if (module_enabled) {

		FlightBatterySettingsConnectCallback(settingsUpdatedCb);

		// Start tasks
		batteryTaskHandle = PIOS_Thread_Create(batteryTask, "batteryBridge", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
		TaskMonitorAdd(TASKINFO_RUNNING_BATTERY, batteryTaskHandle);
		return 0;
	}
	return -1;
}
/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t BatteryInitialize(void)
{
#ifdef MODULE_Battery_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_BATTERY] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
		return 0;
	}
#endif
	FlightBatterySettingsInitialize();
	FlightBatteryStateInitialize();

	return 0;
}
MODULE_INITCALL(BatteryInitialize, BatteryStart)

static bool battery_settings_updated;

/**
 * Main task. It does not return.
 */
static void batteryTask(void * parameters)
{
	const float dT = SAMPLE_PERIOD_MS / 1000.0f;

	settingsUpdatedCb(NULL);

	// Main task loop
	uint32_t lastSysTime;
	lastSysTime = PIOS_Thread_Systime();
	while (true) {
		PIOS_Thread_Sleep_Until(&lastSysTime, SAMPLE_PERIOD_MS);

		FlightBatteryStateData flightBatteryData;
		FlightBatterySettingsData batterySettings;
		float energyRemaining;

		FlightBatteryStateGet(&flightBatteryData);

		if (battery_settings_updated) {
			battery_settings_updated = false;
			FlightBatterySettingsGet(&batterySettings);

			voltageADCPin = batterySettings.VoltagePin;
			if (voltageADCPin == FLIGHTBATTERYSETTINGS_VOLTAGEPIN_NONE)
				voltageADCPin = -1;

			currentADCPin = batterySettings.CurrentPin;
			if (currentADCPin == FLIGHTBATTERYSETTINGS_CURRENTPIN_NONE)
				currentADCPin = -1;
		}

		bool adc_pin_invalid = false;
		bool adc_offset_invalid = false;

		// handle voltage
		if (voltageADCPin >= 0) {
			float adc_voltage = (float)PIOS_ADC_GetChannelVolt(voltageADCPin);
			float scaled_voltage = 0.0f;

			// A negative result indicates an error (PIOS_ADC_GetChannelVolt returns negative on error)
			if(adc_voltage < 0.0f)
				adc_pin_invalid = true;
			else {
				// scale to actual voltage
				scaled_voltage = (adc_voltage * 1000.0f
						/ batterySettings.SensorCalibrationFactor[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONFACTOR_VOLTAGE])
						+ batterySettings.SensorCalibrationOffset[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONOFFSET_VOLTAGE]; //in Volts

				// disallow negative values as these are cast to unsigned integral types
				// in some telemetry layers
				if(scaled_voltage < 0.0f) {
					scaled_voltage = 0.0f;
					adc_offset_invalid = true;
				}
			}

			flightBatteryData.Voltage = scaled_voltage;

			// Set alarm if insufficient voltage, but only if the threshold is positive
			if ((batterySettings.VoltageThresholds[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_ALARM]) > 0
			        && (flightBatteryData.Voltage < batterySettings.VoltageThresholds[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_ALARM])) {
				AlarmsSet(SYSTEMALARMS_ALARM_BATTERY, SYSTEMALARMS_ALARM_CRITICAL);
			} else if ((batterySettings.VoltageThresholds[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_WARNING] > 0)
			           && (flightBatteryData.Voltage < batterySettings.VoltageThresholds[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_WARNING])) {
				AlarmsSet(SYSTEMALARMS_ALARM_BATTERY, SYSTEMALARMS_ALARM_WARNING);
			} else {
				AlarmsClear(SYSTEMALARMS_ALARM_BATTERY);
			}
		} else {
			flightBatteryData.Voltage = 0;
		}

		// handle current
		if (currentADCPin >= 0) {
			float adc_voltage = (float)PIOS_ADC_GetChannelVolt(currentADCPin);
			float scaled_current = 0.0f;

			// A negative result indicates an error (PIOS_ADC_GetChannelVolt returns -1 on error)
			if(adc_voltage < 0.0f)
				adc_pin_invalid = true;
			else {
				// scale to actual current
				scaled_current = (adc_voltage * 1000.0f
						/ batterySettings.SensorCalibrationFactor[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONFACTOR_CURRENT])
						+ batterySettings.SensorCalibrationOffset[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONOFFSET_CURRENT]; //in Amps

				// disallow negative values as these are cast to unsigned integral types
				// in some telemetry layers
				if(scaled_current < 0.0f) {
					scaled_current = 0.0f;
					adc_offset_invalid = true;
				}
			}

			flightBatteryData.Current = scaled_current;

			if (flightBatteryData.Current > flightBatteryData.PeakCurrent)
				flightBatteryData.PeakCurrent = flightBatteryData.Current; //in Amps

			flightBatteryData.ConsumedEnergy += (flightBatteryData.Current * dT * 1000.0f / 3600.0f); //in mAh

			//Apply a 2 second rise time low-pass filter to average the current
			float alpha = 1.0f - dT / (dT + 2.0f);
			flightBatteryData.AvgCurrent = alpha * flightBatteryData.AvgCurrent + (1 - alpha) * flightBatteryData.Current; //in Amps

			energyRemaining = batterySettings.Capacity - flightBatteryData.ConsumedEnergy; // in mAh
			if (flightBatteryData.AvgCurrent > 0)
				flightBatteryData.EstimatedFlightTime = (energyRemaining / (flightBatteryData.AvgCurrent * 1000.0f)) * 3600.0f; //in Sec
			else
				flightBatteryData.EstimatedFlightTime = 9999;

			// Set alarm if insufficient time remaining, but only if the threshold is positive
			if ((batterySettings.FlightTimeThresholds[FLIGHTBATTERYSETTINGS_FLIGHTTIMETHRESHOLDS_ALARM] > 0)
			        && (flightBatteryData.EstimatedFlightTime < batterySettings.FlightTimeThresholds[FLIGHTBATTERYSETTINGS_FLIGHTTIMETHRESHOLDS_ALARM])) {
				AlarmsSet(SYSTEMALARMS_ALARM_FLIGHTTIME, SYSTEMALARMS_ALARM_CRITICAL);
			} else if ((batterySettings.FlightTimeThresholds[FLIGHTBATTERYSETTINGS_FLIGHTTIMETHRESHOLDS_WARNING] > 0)
			           && (flightBatteryData.EstimatedFlightTime < batterySettings.FlightTimeThresholds[FLIGHTBATTERYSETTINGS_FLIGHTTIMETHRESHOLDS_WARNING])) {
				AlarmsSet(SYSTEMALARMS_ALARM_FLIGHTTIME, SYSTEMALARMS_ALARM_WARNING);
			} else {
				AlarmsClear(SYSTEMALARMS_ALARM_FLIGHTTIME);
			}
		} else {
			flightBatteryData.Current = 0;
		}

		if(adc_pin_invalid)
			AlarmsSet(SYSTEMALARMS_ALARM_ADC, SYSTEMALARMS_ALARM_CRITICAL);
		else if(adc_offset_invalid)
			AlarmsSet(SYSTEMALARMS_ALARM_ADC, SYSTEMALARMS_ALARM_WARNING);
		else if(voltageADCPin >= 0 || currentADCPin >= 0)
			AlarmsSet(SYSTEMALARMS_ALARM_ADC, SYSTEMALARMS_ALARM_OK);
		else
			AlarmsSet(SYSTEMALARMS_ALARM_ADC, SYSTEMALARMS_ALARM_UNINITIALISED);

		FlightBatteryStateSet(&flightBatteryData);
	}
}

//! Indicates the battery settings have been updated
static void settingsUpdatedCb(UAVObjEvent * objEv)
{
	battery_settings_updated = true;
}

/**
  * @}
  * @}
  */
