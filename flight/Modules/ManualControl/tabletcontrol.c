/**
 ******************************************************************************
 * @addtogroup OpenPilotModules OpenPilot Modules
 * @{
 * @addtogroup ManualControlModule Manual Control Module
 * @brief Takes input from the tablet to control the UAV
 * @{
 *
 * When ManualControlCommand.FlightMode is Tablet then control is passed to
 * this code which uses information from @ref TabletInfo to determine what
 * flight mode and location to use.
 *
 * @file       tabletcontrol.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @brief      Part of the ManualControl module which deals with tablet links
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
#include "flightstatus.h"
#include "gpsposition.h"
#include "homelocation.h"
#include "pathdesired.h"
#include "positionactual.h"
#include "tabletinfo.h"

#define HOME_ALTITUDE_OFFSET 5
#define FOLLOWME_RADIUS      20

//! Private methods
int32_t tabletInfoToNED(TabletInfoData *tabletInfo, float *NED);

/**
 * Process the information from the tablet and control UAV
 * accordingly
 */
int32_t processTabletInfo()
{
	TabletInfoData tabletInfo;
	TabletInfoGet(&tabletInfo);

	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);

	PathDesiredData pathDesired;
	PathDesiredGet(&pathDesired);

	uint8_t mode = flightStatus.FlightMode;

	switch(tabletInfo.TabletModeDesired) {
		case TABLETINFO_TABLETMODEDESIRED_POSITIONHOLD:
			if (mode != FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD) {
				mode = FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD;

				PositionActualData positionActual;
				PositionActualGet(&positionActual);

				pathDesired.End[0] = positionActual.North;
				pathDesired.End[1] = positionActual.East;
				pathDesired.End[2] = -HOME_ALTITUDE_OFFSET;
				pathDesired.Mode = PATHDESIRED_MODE_FLYENDPOINT;
				pathDesired.StartingVelocity = 5;
				pathDesired.EndingVelocity = 5;

				PathDesiredSet(&pathDesired);				
			}
			break;
		case TABLETINFO_TABLETMODEDESIRED_RETURNTOHOME:
			mode = FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD;

			pathDesired.End[0] = 0;
			pathDesired.End[1] = 0;
			pathDesired.End[2] = -HOME_ALTITUDE_OFFSET;
			pathDesired.Mode = PATHDESIRED_MODE_FLYENDPOINT;
			pathDesired.StartingVelocity = 5;
			pathDesired.EndingVelocity = 5;

			PathDesiredSet(&pathDesired);

			break;
		case TABLETINFO_TABLETMODEDESIRED_RETURNTOTABLET:
		{
			float NED[3];

			mode = FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD;
			tabletInfoToNED(&tabletInfo, NED);

			pathDesired.End[0] = NED[0];
			pathDesired.End[1] = NED[1];
			pathDesired.End[2] = -HOME_ALTITUDE_OFFSET;
			pathDesired.Mode = PATHDESIRED_MODE_FLYENDPOINT;
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
			// Follow the tablet location at a fixed height, but always following by
			// a set radius
			float NED[3];
			tabletInfoToNED(&tabletInfo, NED);

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
				pathDesired.Mode = PATHDESIRED_MODE_FLYENDPOINT;
			} else {
				pathDesired.End[0] = positionActual.North;
				pathDesired.End[1] = positionActual.East;
				pathDesired.End[2] = -HOME_ALTITUDE_OFFSET;
				pathDesired.Mode = PATHDESIRED_MODE_FLYENDPOINT;
			}
			pathDesired.StartingVelocity = 5;
			pathDesired.EndingVelocity = 5;

			PathDesiredSet(&pathDesired);
		}
			break;
		case TABLETINFO_TABLETMODEDESIRED_LAND:
		default:
			AlarmsSet(SYSTEMALARMS_ALARM_MANUALCONTROL, SYSTEMALARMS_ALARM_ERROR);
			break;
	}

	// Update mode if changed
	if (mode != flightStatus.FlightMode) {
		flightStatus.FlightMode = mode;
		FlightStatusSet(&flightStatus);
	}		

	return 0;
}

/**
 * Convert the tablet information (in LLA * 10e6) to NED reference frame
 * @param[in] tabletInfo The information from the tablet
 * @param[out] NED Computed NED values
 */
int32_t tabletInfoToNED(TabletInfoData *tabletInfo, float *NED)
{
	// TODO: Abstract out this code and also precompute the part based
	// on home location.

	HomeLocationData homeLocation;
	HomeLocationGet(&homeLocation);

	GPSPositionData gpsPosition;
	GPSPositionGet(&gpsPosition);

	const float DEG2RAD = 3.141592653589793f / 180.0f;
	float lat = homeLocation.Latitude / 10.0e6f * DEG2RAD;
	float alt = homeLocation.Altitude;

	float T[3];
	T[0] = alt+6.378137E6f;
	T[1] = cosf(lat)*(alt+6.378137E6f);
	T[2] = -1.0f;

	// Tablet altitude is in WSG84 but we use height above the geoid elsewhere so use the
	// GPS GeoidSeparatino as a proxy
	float dL[3] = {(tabletInfo->Latitude - homeLocation.Latitude) / 10.0e6f * DEG2RAD,
		(tabletInfo->Longitude - homeLocation.Longitude) / 10.0e6f * DEG2RAD,
		(tabletInfo->Altitude + gpsPosition.GeoidSeparation - homeLocation.Altitude)};

	NED[0] = T[0] * dL[0];
	NED[1] = T[1] * dL[1];
	NED[2] = T[2] * dL[2];

	return 0;
}