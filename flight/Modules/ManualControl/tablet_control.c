/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Control Control Module
 * @{
 *
 * @file       tablet_control.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Use tablet for control source
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
#include "tablet_control.h"
#include "transmitter_control.h"
#include "physical_constants.h"

#include "flightstatus.h"
#include "gpsposition.h"
#include "homelocation.h"
#include "pathdesired.h"
#include "positionactual.h"
#include "tabletinfo.h"
#include "systemsettings.h"

#if !defined(COPTERCONTROL)

//! Private methods

//! Private constants
#define HOME_ALTITUDE_OFFSET 5
#define FOLLOWME_RADIUS      20

//! Initialize the tablet controller
int32_t tablet_control_initialize()
{
	TabletInfoInitialize();
	return 0;
}

//! Process updates for the tablet controller
int32_t tablet_control_update()
{
	// TODO: Determine what to do when there are insufficient updates
	// from the tablet.  For now the transmitter is the authority so
	// that is what determines whether to fall into failsafe.
	return 0;
}

/**
 * Select and use tablet control
 * @param [in] reset_controller True if previously another controller was used
 */
int32_t tablet_control_select(bool reset_controller)
{
	TabletInfoData tabletInfo;
	TabletInfoGet(&tabletInfo);

	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);

	if (PathDesiredHandle() == NULL)
		return -1;

	PathDesiredData pathDesired;
	PathDesiredGet(&pathDesired);

	uint8_t mode = flightStatus.FlightMode;
	static TabletInfoTabletModeDesiredOptions last_tablet_mode;

	switch(tabletInfo.TabletModeDesired) {
		case TABLETINFO_TABLETMODEDESIRED_POSITIONHOLD:
			mode = FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD;
			break;
		case TABLETINFO_TABLETMODEDESIRED_RETURNTOHOME:
			mode = FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME;
			break;
		case TABLETINFO_TABLETMODEDESIRED_RETURNTOTABLET:
			mode = FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER;
			break;
		case TABLETINFO_TABLETMODEDESIRED_PATHPLANNER:
			mode = FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER;
			break;
		case TABLETINFO_TABLETMODEDESIRED_FOLLOWME:
			mode = FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER;
			break;
		case TABLETINFO_TABLETMODEDESIRED_LAND:
		default:
			AlarmsSet(SYSTEMALARMS_ALARM_MANUALCONTROL, SYSTEMALARMS_ALARM_ERROR);

			// Fail out.  This will trigger failsafe mode.
			return -1;
	}

	// Cache the last tablet mode
	last_tablet_mode = tabletInfo.TabletModeDesired;

	// Update mode if changed
	if (mode != flightStatus.FlightMode) {
		flightStatus.FlightMode = mode;
		FlightStatusSet(&flightStatus);
	}		

	return 0;
}

//! Get any control events
enum control_events tablet_control_get_events()
{
	// For now ARM / DISARM events still come from the transmitter
	return transmitter_control_get_events();
}

#else

int32_t tablet_control_initialize()
{
	return 0;
}

//! Process updates for the tablet controller
int32_t tablet_control_update()
{
	return 0;
}

int32_t tablet_control_select(bool reset_controller)
{
	return 0;
}

//! When not supported force disarming
enum control_events tablet_control_get_events()
{
	return CONTROL_EVENTS_DISARM;
}

#endif

/**
 * @}
 * @}
 */
