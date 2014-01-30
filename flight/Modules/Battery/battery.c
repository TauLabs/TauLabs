/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup BatteryModule Battery Module
 * @{
 *
 * @file       battery.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

// ****************
// Private constants
#define STACK_SIZE_BYTES            468
#define TASK_PRIORITY               (tskIDLE_PRIORITY + 1)
#define SAMPLE_PERIOD_MS            500
// Private types

// Private variables
static bool module_enabled = false;
static xTaskHandle batteryTaskHandle;
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
		xTaskCreate(batteryTask, (signed char *) "batteryBridge", STACK_SIZE_BYTES / 4, NULL, TASK_PRIORITY, &batteryTaskHandle);
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
#define HAS_SENSOR(x) batterySettings.SensorType[x]==FLIGHTBATTERYSETTINGS_SENSORTYPE_ENABLED

static bool battery_settings_updated;

/**
 * Main task. It does not return.
 */
static void batteryTask(void * parameters)
{
	const float dT = SAMPLE_PERIOD_MS / 1000.0f;

	settingsUpdatedCb(NULL);

	// Main task loop
	portTickType lastSysTime;
	lastSysTime = xTaskGetTickCount();
	while (true) {
		vTaskDelayUntil(&lastSysTime, MS2TICKS(SAMPLE_PERIOD_MS));

		FlightBatteryStateData flightBatteryData;
		FlightBatterySettingsData batterySettings;
		float energyRemaining;

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

		//calculate the battery parameters
		if (voltageADCPin >= 0) {
			flightBatteryData.Voltage = ((float) PIOS_ADC_GetChannelVolt(voltageADCPin)) / batterySettings.SensorCalibrationFactor[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONFACTOR_VOLTAGE] * 1000.0f +
							batterySettings.SensorCalibrationOffset[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONOFFSET_VOLTAGE]; //in Volts
		} else {
			flightBatteryData.Voltage = 0; //Dummy placeholder value. This is in case we get another source of battery current which is not from the ADC
		}

		if (currentADCPin >= 0) {
			flightBatteryData.Current = ((float) PIOS_ADC_GetChannelVolt(currentADCPin)) / batterySettings.SensorCalibrationFactor[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONFACTOR_CURRENT] * 1000.0f +
							batterySettings.SensorCalibrationOffset[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONOFFSET_CURRENT]; //in Amps
			if (flightBatteryData.Current > flightBatteryData.PeakCurrent)
				flightBatteryData.PeakCurrent = flightBatteryData.Current; //in Amps
		} else { //If there's no current measurement, we still need to assign one. Make it negative, so it can never trigger an alarm
			flightBatteryData.Current = -1; //Dummy placeholder value. This is in case we get another source of battery current which is not from the ADC
		}

		flightBatteryData.ConsumedEnergy += (flightBatteryData.Current * dT * 1000.0f / 3600.0f); //in mAh

		//Apply a 2 second rise time low-pass filter to average the current
		float alpha = 1.0f - dT / (dT + 2.0f);
		flightBatteryData.AvgCurrent = alpha * flightBatteryData.AvgCurrent + (1 - alpha) * flightBatteryData.Current; //in Amps

		energyRemaining = batterySettings.Capacity - flightBatteryData.ConsumedEnergy; // in mAh
		if (flightBatteryData.AvgCurrent > 0)
			flightBatteryData.EstimatedFlightTime = (energyRemaining / (flightBatteryData.AvgCurrent * 1000.0f)) * 3600.0f; //in Sec
		else
			flightBatteryData.EstimatedFlightTime = 9999;

		//generate alarms where needed...
		if ((flightBatteryData.Voltage <= 0) && (flightBatteryData.Current <= 0)) {
			//FIXME: There's no guarantee that a floating ADC will give 0. So this
			// check might fail, even when there's nothing attached.
			AlarmsSet(SYSTEMALARMS_ALARM_BATTERY, SYSTEMALARMS_ALARM_ERROR);
			AlarmsSet(SYSTEMALARMS_ALARM_FLIGHTTIME, SYSTEMALARMS_ALARM_ERROR);
		} else {
			// FIXME: should make the timer alarms user configurable
			if (flightBatteryData.EstimatedFlightTime < 30)
				AlarmsSet(SYSTEMALARMS_ALARM_FLIGHTTIME, SYSTEMALARMS_ALARM_CRITICAL);
			else if (flightBatteryData.EstimatedFlightTime < 120)
				AlarmsSet(SYSTEMALARMS_ALARM_FLIGHTTIME, SYSTEMALARMS_ALARM_WARNING);
			else
				AlarmsClear(SYSTEMALARMS_ALARM_FLIGHTTIME);

			// FIXME: should make the battery voltage detection dependent on battery type.
			/*Not so sure. Some users will want to run their batteries harder than others, so it should be the user's choice. [KDS]*/
			if (flightBatteryData.Voltage < batterySettings.VoltageThresholds[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_ALARM])
				AlarmsSet(SYSTEMALARMS_ALARM_BATTERY, SYSTEMALARMS_ALARM_CRITICAL);
			else if (flightBatteryData.Voltage < batterySettings.VoltageThresholds[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_WARNING])
				AlarmsSet(SYSTEMALARMS_ALARM_BATTERY, SYSTEMALARMS_ALARM_WARNING);
			else
				AlarmsClear(SYSTEMALARMS_ALARM_BATTERY);
		}

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
