/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 * @file       sanitycheck.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Utilities to validate a flight configuration
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
#include "taskmonitor.h"
#include <pios_board_info.h>
#include "flightstatus.h"
#include "sanitycheck.h"
#include "manualcontrolsettings.h"
#include "stabilizationsettings.h"
#include "stateestimation.h"
#include "systemalarms.h"
#include "systemsettings.h"

#include "pios_sensors.h"

/****************************
 * Current checks:
 * 1. If a flight mode switch allows autotune and autotune module not running
 * 2. If airframe is a multirotor and either manual is available or a stabilization mode uses "none"
 ****************************/

//! Check it is safe to arm in this position
static int32_t check_safe_to_arm();

//! Check a stabilization mode switch position for safety
static int32_t check_stabilization_settings(int index, bool multirotor);

//! Check a stabilization mode switch position for safety
static int32_t check_stabilization_rates();

//! Check the system is safe for autonomous flight
static int32_t check_safe_autonomous();

//!  Set the error code and alarm state
static void set_config_error(SystemAlarmsConfigErrorOptions error_code);

/**
 * Run a preflight check over the hardware configuration
 * and currently active modules
 */
int32_t configuration_check()
{
	SystemAlarmsConfigErrorOptions error_code = SYSTEMALARMS_CONFIGERROR_NONE;
	
	// Get board type
	const struct pios_board_info * bdinfo = &pios_board_info_blob;	
	bool coptercontrol = bdinfo->board_type == 0x04;

	// For when modules are not running we should explicitly check the objects are
	// valid
	if (ManualControlSettingsHandle() == NULL ||
		SystemSettingsHandle() == NULL) {
		AlarmsSet(SYSTEMALARMS_ALARM_SYSTEMCONFIGURATION, SYSTEMALARMS_ALARM_CRITICAL);
		return 0;
	}

	// Classify airframe type
	bool multirotor = true;
	uint8_t airframe_type;
	SystemSettingsAirframeTypeGet(&airframe_type);
	switch(airframe_type) {
		case SYSTEMSETTINGS_AIRFRAMETYPE_QUADX:
		case SYSTEMSETTINGS_AIRFRAMETYPE_QUADP:
		case SYSTEMSETTINGS_AIRFRAMETYPE_HEXA:
		case SYSTEMSETTINGS_AIRFRAMETYPE_OCTO:
		case SYSTEMSETTINGS_AIRFRAMETYPE_HEXAX:
		case SYSTEMSETTINGS_AIRFRAMETYPE_OCTOV:
		case SYSTEMSETTINGS_AIRFRAMETYPE_OCTOCOAXP:
		case SYSTEMSETTINGS_AIRFRAMETYPE_HEXACOAX:
		case SYSTEMSETTINGS_AIRFRAMETYPE_TRI:
			multirotor = true;
			break;
		default:
			multirotor = false;
	}

	// For each available flight mode position sanity check the available
	// modes
	uint8_t num_modes;
	uint8_t modes[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_NUMELEM];
	ManualControlSettingsFlightModeNumberGet(&num_modes);
	ManualControlSettingsFlightModePositionGet(modes);
	

	for(uint32_t i = 0; i < num_modes; i++) {
		switch(modes[i]) {
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_MANUAL:
				if (multirotor) {
					error_code = SYSTEMALARMS_CONFIGERROR_STABILIZATION;
				}
				break;
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_ACRO:
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_LEVELING:
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_MWRATE:
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_HORIZON:
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_AXISLOCK:
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_VIRTUALBAR:
				// always ok
				break;
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_STABILIZED1:
				error_code = (error_code == SYSTEMALARMS_CONFIGERROR_NONE) ? check_stabilization_settings(1, multirotor) : error_code;
				break;
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_STABILIZED2:
				error_code = (error_code == SYSTEMALARMS_CONFIGERROR_NONE) ? check_stabilization_settings(2, multirotor) : error_code;
				break;
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_STABILIZED3:
				error_code = (error_code == SYSTEMALARMS_CONFIGERROR_NONE) ? check_stabilization_settings(3, multirotor) : error_code;
				break;
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_AUTOTUNE:
				if (!TaskMonitorQueryRunning(TASKINFO_RUNNING_AUTOTUNE))
					error_code = SYSTEMALARMS_CONFIGERROR_AUTOTUNE;
				break;
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_ALTITUDEHOLD:
				if (coptercontrol)
					error_code = SYSTEMALARMS_CONFIGERROR_ALTITUDEHOLD;
				else {
					if ( !TaskMonitorQueryRunning(TASKINFO_RUNNING_ALTITUDEHOLD) )
						error_code = SYSTEMALARMS_CONFIGERROR_ALTITUDEHOLD;
				}
				break;
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_POSITIONHOLD:
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_RETURNTOHOME:
				if (coptercontrol) {
					error_code = SYSTEMALARMS_CONFIGERROR_PATHPLANNER;
				}
				else {
					if (!TaskMonitorQueryRunning(TASKINFO_RUNNING_PATHFOLLOWER)) {
						error_code = SYSTEMALARMS_CONFIGERROR_PATHPLANNER;
					} else {
						error_code = check_safe_autonomous();
					}
				}
				break;
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_PATHPLANNER:
			case MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_TABLETCONTROL:
				if (coptercontrol) {
					error_code = SYSTEMALARMS_CONFIGERROR_PATHPLANNER;
				}
				else {
					if (!TaskMonitorQueryRunning(TASKINFO_RUNNING_PATHFOLLOWER) ||
						!TaskMonitorQueryRunning(TASKINFO_RUNNING_PATHPLANNER)) {
						error_code = SYSTEMALARMS_CONFIGERROR_PATHPLANNER;
					} else {
						error_code = check_safe_autonomous();
					}
				}
				break;
			default:
				// Uncovered modes are automatically an error
				error_code = SYSTEMALARMS_CONFIGERROR_UNDEFINED;
		}
	}

	// Check the stabilization rates are within what the sensors can track
	error_code = (error_code == SYSTEMALARMS_CONFIGERROR_NONE) ? check_stabilization_rates() : error_code;

	// Only check safe to arm if no other errors exist
	error_code = (error_code == SYSTEMALARMS_CONFIGERROR_NONE) ? check_safe_to_arm() : error_code;

	set_config_error(error_code);

	return 0;
}


/**
 * Checks the stabiliation settings for a paritcular mode and makes
 * sure it is appropriate for the airframe
 * @param[in] index Which stabilization mode to check
 * @returns SYSTEMALARMS_CONFIGERROR_NONE or SYSTEMALARMS_CONFIGERROR_MULTIROTOR
 */
static int32_t check_stabilization_settings(int index, bool multirotor)
{
	// Make sure the modes have identical sizes
	if (MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_NUMELEM != MANUALCONTROLSETTINGS_STABILIZATION2SETTINGS_NUMELEM ||
		MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_NUMELEM != MANUALCONTROLSETTINGS_STABILIZATION3SETTINGS_NUMELEM)
		return SYSTEMALARMS_CONFIGERROR_MULTIROTOR;

	uint8_t modes[MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_NUMELEM];

	// Get the different axis modes for this switch position
	switch(index) {
		case 1:
			ManualControlSettingsStabilization1SettingsGet(modes);
			break;
		case 2:
			ManualControlSettingsStabilization2SettingsGet(modes);
			break;
		case 3:
			ManualControlSettingsStabilization3SettingsGet(modes);
			break;
		default:
			return SYSTEMALARMS_CONFIGERROR_NONE;
	}

	// For multirotors verify that nothing is set to "none"
	if (multirotor) {
		for(uint32_t i = 0; i < NELEMENTS(modes); i++) {
			if (modes[i] == MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_NONE)
				return SYSTEMALARMS_CONFIGERROR_MULTIROTOR;

			// If this axis allows enabling an autotune behavior without the module
			// running then set an alarm now that aututune module initializes the
			// appropriate objects
			if ((modes[i] == MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_SYSTEMIDENT) &&
				(!TaskMonitorQueryRunning(TASKINFO_RUNNING_AUTOTUNE)))
				return SYSTEMALARMS_CONFIGERROR_AUTOTUNE;
		}
	}

	// POI mode is only valid for YAW in the case it is enabled and camera stab is running
	if (modes[MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_ROLL] == MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_POI ||
		modes[MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_PITCH] == MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_POI)
		return SYSTEMALARMS_CONFIGERROR_STABILIZATION;
	if (modes[MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_YAW] == MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_POI) {
#if !defined(CAMERASTAB_POI_MODE)
		return SYSTEMALARMS_CONFIGERROR_STABILIZATION;
#endif
		// TODO: Need to check camera stab is actually running
	}


	// Warning: This assumes that certain conditions in the XML file are met.  That 
	// MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_NONE has the same numeric value for each channel
	// and is the same for STABILIZATIONDESIRED_STABILIZATIONMODE_NONE

	return SYSTEMALARMS_CONFIGERROR_NONE;
}

/**
 * If the system is disarmed, look for a variety of conditions that
 * make it unsafe to arm (that might not be dangerous to engage once
 * flying).
 */
static int32_t check_safe_to_arm()
{
	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);

	// Only arm in traditional modes where pilot has control
	if (flightStatus.Armed != FLIGHTSTATUS_ARMED_ARMED) {
		switch (flightStatus.FlightMode) {
			case FLIGHTSTATUS_FLIGHTMODE_MANUAL:
			case FLIGHTSTATUS_FLIGHTMODE_ACRO:
			case FLIGHTSTATUS_FLIGHTMODE_LEVELING:
			case FLIGHTSTATUS_FLIGHTMODE_MWRATE:
			case FLIGHTSTATUS_FLIGHTMODE_HORIZON:
			case FLIGHTSTATUS_FLIGHTMODE_AXISLOCK:
			case FLIGHTSTATUS_FLIGHTMODE_VIRTUALBAR:
			case FLIGHTSTATUS_FLIGHTMODE_STABILIZED1:
			case FLIGHTSTATUS_FLIGHTMODE_STABILIZED2:
			case FLIGHTSTATUS_FLIGHTMODE_STABILIZED3:
			case FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD:
				break;
			default:
				// Any mode not specifically allowed prevents arming
				return SYSTEMALARMS_CONFIGERROR_UNSAFETOARM;
		}
	}

	return SYSTEMALARMS_CONFIGERROR_NONE;
}

/**
 * If an autonomous mode is available, make sure all the configurations are
 * valid for it
 */
static int32_t check_safe_autonomous()
{
	// The current filter combinations are safe for navigation
	//   Attitude   |  Navigation
	//     Comp     |     Raw          (not recommended)
	//     Comp     |     INS          (recommmended)
	//    Anything  |     None         (unsafe)
	//   INSOutdoor |     INS
	//   INSIndoor  |     INS          (unsafe)


#if !defined(COPTERCONTROL)
	StateEstimationData stateEstimation;
	StateEstimationGet(&stateEstimation);

	if (stateEstimation.AttitudeFilter == STATEESTIMATION_ATTITUDEFILTER_INSINDOOR)
		return SYSTEMALARMS_CONFIGERROR_NAVFILTER;

	// Anything not allowed is invalid, safe default
	if (stateEstimation.NavigationFilter != STATEESTIMATION_NAVIGATIONFILTER_INS &&
		stateEstimation.NavigationFilter != STATEESTIMATION_NAVIGATIONFILTER_RAW)
		return SYSTEMALARMS_CONFIGERROR_NAVFILTER;
#endif

	return SYSTEMALARMS_CONFIGERROR_NONE;
}

/**
 * Check the rates achieved via stabilization are ones that can be tracked
 * by the gyros as configured.
 * @return error code if not
 */
static int32_t check_stabilization_rates()
{
	const float MAXIMUM_SAFE_FRACTIONAL_RATE = 0.85;
	int32_t max_safe_rate = PIOS_SENSORS_GetMaxGyro() * MAXIMUM_SAFE_FRACTIONAL_RATE;
	float rates[3];

	StabilizationSettingsManualRateGet(rates);
	if (rates[0] > max_safe_rate || rates[1] > max_safe_rate || rates[2] > max_safe_rate)
		return SYSTEMALARMS_CONFIGERROR_STABILIZATION;

	StabilizationSettingsMaximumRateGet(rates);
	if (rates[0] > max_safe_rate || rates[1] > max_safe_rate || rates[2] > max_safe_rate)
		return SYSTEMALARMS_CONFIGERROR_STABILIZATION;

	return SYSTEMALARMS_CONFIGERROR_NONE;
}

/**
 * Set the error code and alarm state
 * @param[in] error code
 */
static void set_config_error(SystemAlarmsConfigErrorOptions error_code)
{
	// Get the severity of the alarm given the error code
	SystemAlarmsAlarmOptions severity;
	switch (error_code) {
	case SYSTEMALARMS_CONFIGERROR_NONE:
		severity = SYSTEMALARMS_ALARM_OK;
		break;
	case SYSTEMALARMS_CONFIGERROR_STABILIZATION:
	case SYSTEMALARMS_CONFIGERROR_MULTIROTOR:
	case SYSTEMALARMS_CONFIGERROR_AUTOTUNE:
	case SYSTEMALARMS_CONFIGERROR_ALTITUDEHOLD:
	case SYSTEMALARMS_CONFIGERROR_POSITIONHOLD:
	case SYSTEMALARMS_CONFIGERROR_PATHPLANNER:
	case SYSTEMALARMS_CONFIGERROR_NAVFILTER:
	case SYSTEMALARMS_CONFIGERROR_UNSAFETOARM:
		severity = SYSTEMALARMS_ALARM_ERROR;
		break;
	default:
		severity = SYSTEMALARMS_ALARM_ERROR;
		error_code = SYSTEMALARMS_CONFIGERROR_UNDEFINED;
		break;
	}

	// Make sure not to set the error code if it didn't change
	SystemAlarmsConfigErrorOptions current_error_code;
	SystemAlarmsConfigErrorGet((uint8_t *) &current_error_code);
	if (current_error_code != error_code) {
		SystemAlarmsConfigErrorSet((uint8_t *) &error_code);
	}

	// AlarmSet checks only updates on toggle
	AlarmsSet(SYSTEMALARMS_ALARM_SYSTEMCONFIGURATION, (uint8_t) severity);
}

/**
 * @}
 */
