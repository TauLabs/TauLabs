/**
 ******************************************************************************
 * @addtogroup Modules Tau Labs Modules
 * @{
 * @addtogroup Control Control Module
 * @{
 *
 * @file       geofence_control.c
 * @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
 * @brief      Geofence controller when vehicle leaves geofence
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
#include "control.h"
#include "geofence_control.h"
#include "transmitter_control.h"
#include "crash_commands.h"

#include "flightstatus.h"
#include "stabilizationdesired.h"
#include "systemalarms.h"

//! Initialize the geofence controller
int32_t geofence_control_initialize()
{
	return 0;
}

//! Perform any updates to the geofence controller
int32_t geofence_control_update()
{
	// This is where we would hold the vehicle in geofence mode until it's safely back inside the boundary.
	return 0;
}

/**
 * Select and use geofence control
 * @param [in] reset_controller True if previously another controller was used
 */
int32_t geofence_control_select(bool reset_controller)
{
	SystemAlarmsGeoFenceOptions geofence_alarm_code;
	SystemAlarmsGeoFenceGet(&geofence_alarm_code);

	FlightStatusFlightModeOptions old_flight_status;
	FlightStatusFlightModeOptions new_flight_status;

	switch (geofence_alarm_code){
	case SYSTEMALARMS_GEOFENCE_LEAVINGBOUNDARY:
		new_flight_status = FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME;
		break;
	case SYSTEMALARMS_GEOFENCE_LEFTBOUNDARY:
	case SYSTEMALARMS_GEOFENCE_INSUFFICIENTVERTICES: // Insufficient vertices and faces shouldn't cause a crash, but might want to cause a sanity check error. The problem is that if geofencing is turned on, and there's no computer nearby, this can cripple flight at non-standard fields.
	case SYSTEMALARMS_GEOFENCE_INSUFFICIENTFACES:    // NOTE: This currently causes a crash in case of a mid-air reboot
		new_flight_status = FLIGHTSTATUS_FLIGHTMODE_STABILIZED1;
		nice_crash();
		break;
	case SYSTEMALARMS_GEOFENCE_UNDEFINED:
	default:
		new_flight_status = FLIGHTSTATUS_FLIGHTMODE_STABILIZED1;
		nasty_crash();
		break;
	}

	FlightStatusFlightModeGet(&old_flight_status);
	if (old_flight_status != new_flight_status || reset_controller) {
		FlightStatusFlightModeSet(&new_flight_status);
	}

	return 0;
}

//! Get any control events
enum control_events geofence_control_get_events()
{
	// For now ARM / DISARM events still come from the transmitter.  This
	// means the normal disarm timeout still applies.  To be replaced later
	// by a full state machine determining how geofence should handle
	// arming/disarming
	return transmitter_control_get_events();
}
