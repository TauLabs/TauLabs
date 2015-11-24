/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup GeoFence GeoFence Module
 * @{
 *
 * @file       geofence.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      Check the UAV is within the geofence boundaries
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
#include "misc_math.h"
#include "physical_constants.h"

#include "geofencesettings.h"
#include "positionactual.h"
#include "modulesettings.h"


//
// Configuration
//
#define SAMPLE_PERIOD_MS     250

// Private types

// Private variables

// Private functions
static void settingsUpdated(UAVObjEvent* ev, void *ctx, void *obj, int len);
static void checkPosition(UAVObjEvent* ev, void *ctx, void *obj, int len);

// Private variables
static bool module_enabled;
static GeoFenceSettingsData *geofenceSettings;

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t GeofenceInitialize(void)
{
	module_enabled = false;

#ifdef MODULE_Geofence_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_GEOFENCE] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (module_enabled) {

		GeoFenceSettingsInitialize();

		// allocate and initialize the static data storage only if module is enabled
		geofenceSettings = (GeoFenceSettingsData *) PIOS_malloc(sizeof(GeoFenceSettingsData));
		if (geofenceSettings == NULL) {
			module_enabled = false;
			return -1;
		}

		GeoFenceSettingsConnectCallback(settingsUpdated);
		settingsUpdated(NULL, NULL, NULL, 0);

		// Schedule periodic task to check position
		UAVObjEvent ev = {
			.obj = PositionActualHandle(),
			.instId = 0,
			.event = 0,
		};
		EventPeriodicCallbackCreate(&ev, checkPosition, SAMPLE_PERIOD_MS);

		return 0;
	}

	return -1;
}

/* stub: module has no module thread */
int32_t GeofenceStart(void)
{
	return 0;
}

MODULE_INITCALL(GeofenceInitialize, GeofenceStart);

/**
 * Periodic callback that processes changes in position and
 * sets the alarm.
 */
static void checkPosition(UAVObjEvent* ev, void *ctx, void *obj, int len)
{
	(void) ev; (void) ctx; (void) obj; (void) len;
	if (PositionActualHandle()) {
		PositionActualData positionActual;
		PositionActualGet(&positionActual);

		const float distance2 = powf(positionActual.North, 2) + powf(positionActual.East, 2);

		// ErrorRadius is squared when it is fetched, so this is correct
		if (distance2 > geofenceSettings->ErrorRadius) {
			AlarmsSet(SYSTEMALARMS_ALARM_GEOFENCE, SYSTEMALARMS_ALARM_ERROR);
		} else if (distance2 > geofenceSettings->WarningRadius) {
			AlarmsSet(SYSTEMALARMS_ALARM_GEOFENCE, SYSTEMALARMS_ALARM_WARNING);
		} else {
			AlarmsClear(SYSTEMALARMS_ALARM_GEOFENCE);
		}
	}
}

/**
 * Update the settings
 */
static void settingsUpdated(UAVObjEvent* ev, void *ctx, void *obj, int len)
{
	(void) ev; (void) ctx; (void) obj; (void) len;
	GeoFenceSettingsGet(geofenceSettings);

	// Cache squared distances to save computations
	geofenceSettings->WarningRadius = powf(geofenceSettings->WarningRadius, 2);
	geofenceSettings->ErrorRadius = powf(geofenceSettings->ErrorRadius, 2);
}

/**
 * @}
 * @}
 */
