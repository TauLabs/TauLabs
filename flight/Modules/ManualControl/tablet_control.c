/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Control Control Module
 * @{
 *
 * @file       tablet_control.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
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
static int32_t tabletInfo_to_ned(TabletInfoData *tabletInfo, float *NED);

//! Private constants
#define HOME_ALTITUDE_OFFSET 15
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
			if (mode != FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD || 
			    last_tablet_mode != tabletInfo.TabletModeDesired) {
				mode = FLIGHTSTATUS_FLIGHTMODE_TABLETCONTROL;

				PositionActualData positionActual;
				PositionActualGet(&positionActual);

				pathDesired.End[0] = positionActual.North;
				pathDesired.End[1] = positionActual.East;
				pathDesired.End[2] = positionActual.Down;
				pathDesired.Mode = PATHDESIRED_MODE_HOLDPOSITION;
				pathDesired.StartingVelocity = 5;
				pathDesired.EndingVelocity = 5;

				PathDesiredSet(&pathDesired);				
			}
			break;
		case TABLETINFO_TABLETMODEDESIRED_RETURNTOHOME:
			mode = FLIGHTSTATUS_FLIGHTMODE_TABLETCONTROL;

			pathDesired.End[0] = 0;
			pathDesired.End[1] = 0;
			pathDesired.End[2] = -HOME_ALTITUDE_OFFSET;
			pathDesired.Mode = PATHDESIRED_MODE_HOLDPOSITION;
			pathDesired.StartingVelocity = 5;
			pathDesired.EndingVelocity = 5;

			PathDesiredSet(&pathDesired);

			break;
		case TABLETINFO_TABLETMODEDESIRED_RETURNTOTABLET:
		{
			float NED[3];

			mode = FLIGHTSTATUS_FLIGHTMODE_TABLETCONTROL;
			tabletInfo_to_ned(&tabletInfo, NED);

			pathDesired.End[0] = NED[0];
			pathDesired.End[1] = NED[1];
			pathDesired.End[2] = -HOME_ALTITUDE_OFFSET;
			pathDesired.Mode = PATHDESIRED_MODE_HOLDPOSITION;
			pathDesired.StartingVelocity = 5;
			pathDesired.EndingVelocity = 5;

			PathDesiredSet(&pathDesired);
		}

			break;
		case TABLETINFO_TABLETMODEDESIRED_PATHPLANNER:
			mode = FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER;
			break;
		case TABLETINFO_TABLETMODEDESIRED_FOLLOWME:
		{
			mode = FLIGHTSTATUS_FLIGHTMODE_TABLETCONTROL;
			
			// Follow the tablet location at a fixed height, but always following by
			// a set radius. This mode is updated every cycle, unlike the others.
			float NED[3];
			tabletInfo_to_ned(&tabletInfo, NED);

			PositionActualData positionActual;
			PositionActualGet(&positionActual);

			float DeltaN = NED[0] - positionActual.North;
			float DeltaE = NED[1] - positionActual.East;
			float dist = sqrt(DeltaN * DeltaN + DeltaE * DeltaE);

			// If outside the follow radius code to the nearest point on the border
			// otherwise stay in the same location
			if (dist > FOLLOWME_RADIUS) {
				float frac = (dist - FOLLOWME_RADIUS) / dist;
				pathDesired.End[0] = positionActual.North + frac * DeltaN;
				pathDesired.End[1] = positionActual.East + frac * DeltaE;
				pathDesired.End[2] = -HOME_ALTITUDE_OFFSET;
			} else {
				pathDesired.End[0] = positionActual.North;
				pathDesired.End[1] = positionActual.East;
				pathDesired.End[2] = -HOME_ALTITUDE_OFFSET;
			}
			pathDesired.Mode = FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER;
			pathDesired.StartingVelocity = 5;
			pathDesired.EndingVelocity = 5;

			PathDesiredSet(&pathDesired);
		}
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


/**
 * Convert the tablet information (in LLA * 10e6) to NED reference frame
 * @param[in] tabletInfo The information from the tablet
 * @param[out] NED Computed NED values
 */
static int32_t tabletInfo_to_ned(TabletInfoData *tabletInfo, float *NED)
{
	// TODO: Abstract out this code and also precompute the part based
	// on home location.

	HomeLocationData homeLocation;
	HomeLocationGet(&homeLocation);

	GPSPositionData gpsPosition;
	GPSPositionGet(&gpsPosition);

	float lat = homeLocation.Latitude / 10.0e6f * DEG2RAD;
	float alt = homeLocation.Altitude;

	float T[3];
	T[0] = alt+6.378137E6f;
	T[1] = cosf(lat)*(alt+6.378137E6f);
	T[2] = -1.0f;

	// Tablet altitude is in WSG84 but we use height above the geoid elsewhere so use the
	// GPS GeoidSeparation as a proxy
	// [WARNING] Android altitude can be either referenced to WGS84 ellipsoid or EGM1996 geoid. See:
	// http://stackoverflow.com/questions/11168306/is-androids-gps-altitude-incorrect-due-to-not-including-geoid-height
	// and https://code.google.com/p/android/issues/detail?id=53471
	// This means that "(tabletInfo->Altitude + gpsPosition.GeoidSeparation - homeLocation.Altitude)"
	// will be correct or incorrect depending on the device.
	float dL[3] = {(tabletInfo->Latitude - homeLocation.Latitude) / 10.0e6f * DEG2RAD,
		(tabletInfo->Longitude - homeLocation.Longitude) / 10.0e6f * DEG2RAD,
		(tabletInfo->Altitude + gpsPosition.GeoidSeparation - homeLocation.Altitude)};

	NED[0] = T[0] * dL[0];
	NED[1] = T[1] * dL[1];
	NED[2] = T[2] * dL[2];

	return 0;
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
