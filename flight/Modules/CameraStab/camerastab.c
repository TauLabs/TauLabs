/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup CameraStab Camera Stabilization Module
 * @{
 *
 * @file       camerastab.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      Stabilize camera against the roll pitch and yaw of aircraft
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

/**
 * Output object: @ref CameraDesired
 *
 * This module will periodically calculate the output values for stabilizing the camera.
 * It supports controlling the camera an angle specified by the sticks (provided by
 * @ref AccessoryDesired) or at a rate specified by the sticks.  Alternatively it can
 * just try and hold the camera at a fixed angle.  This module is designed to be used
 * when the flight control is on the airframe of the aircraft.  If the controller is
 * placed on the gimbal using the standard stabilization code will work.
 *
 */

#include "openpilot.h"
#include "misc_math.h"
#include "physical_constants.h"
#include "pios_thread.h"
#include "pios_can.h"

#include "accessorydesired.h"
#include "attitudeactual.h"
#include "camerastabsettings.h"
#include "cameradesired.h"
#include "homelocation.h"
#include "modulesettings.h"
#include "misc_math.h"
#include "poilocation.h"
#include "positionactual.h"
#include "tabletinfo.h"

//
// Configuration
//
#define SAMPLE_PERIOD_MS     10
#define LOAD_DELAY           7000

// Private types
enum {ROLL,PITCH,YAW,MAX_AXES};

// Private variables
static struct CameraStab_data {
	uint32_t lastSysTime;
	uint8_t AttitudeFilter;
	float attitude_filtered[MAX_AXES];
	float inputs[CAMERASTABSETTINGS_INPUT_NUMELEM];
	float FFlastAttitude[MAX_AXES];
	float FFlastFilteredAttitude[MAX_AXES];
	float FFfilterAccumulator[MAX_AXES];
	CameraStabSettingsData settings;
} *csd;

// Private functions
static void attitudeUpdated(UAVObjEvent* ev);
static void settings_updated_cb(UAVObjEvent * ev);
static void applyFF(uint8_t index, float dT_ms, float *attitude, CameraStabSettingsData* cameraStab);
static void gimbal_can_message();

#if defined(CAMERASTAB_POI_MODE)
static void tablet_info_flag_update(UAVObjEvent * ev);
static void tablet_info_process();
static bool tablet_info_updated = false;
#endif /* CAMERASTAB_POI_MODE */

// Private variables
static bool module_enabled;
/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t CameraStabInitialize(void)
{
	module_enabled = false;

#ifdef MODULE_CameraStab_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_CAMERASTAB] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (module_enabled) {

		// allocate and initialize the static data storage only if module is enabled
		csd = (struct CameraStab_data *) PIOS_malloc(sizeof(struct CameraStab_data));
		if (csd == NULL) {
			module_enabled = false;
			return -1;
		}

		// make sure that all inputs[] are zeroed
		memset(csd, 0, sizeof(struct CameraStab_data));
		csd->lastSysTime = PIOS_Thread_Systime() - SAMPLE_PERIOD_MS;

		AttitudeActualInitialize();
		CameraStabSettingsInitialize();
		CameraDesiredInitialize();

		CameraStabSettingsConnectCallback(settings_updated_cb);
		settings_updated_cb(NULL);
#if defined(CAMERASTAB_POI_MODE)
		PoiLocationInitialize();
		TabletInfoInitialize();
		TabletInfoConnectCallback(tablet_info_flag_update);
#endif /* CAMERASTAB_POI_MODE */

		UAVObjEvent ev = {
			.obj = AttitudeActualHandle(),
			.instId = 0,
			.event = 0,
		};
		EventPeriodicCallbackCreate(&ev, attitudeUpdated, SAMPLE_PERIOD_MS);

		return 0;
	}

	return -1;
}

/* stub: module has no module thread */
int32_t CameraStabStart(void)
{
	return 0;
}

MODULE_INITCALL(CameraStabInitialize, CameraStabStart)

/**
 * Periodic callback that processes changes in the attitude
 * and recalculates the desied gimbal angle.
 */
static void attitudeUpdated(UAVObjEvent* ev)
{	
	if (ev->obj != AttitudeActualHandle())
		return;

	AccessoryDesiredData accessory;

	CameraStabSettingsData *settings = &csd->settings;

	// Check how long since last update, time delta between calls in ms
	uint32_t thisSysTime = PIOS_Thread_Systime();
	float dT_ms = thisSysTime - csd->lastSysTime;
	csd->lastSysTime = thisSysTime;

	if (dT_ms <= 0)
		return;

	float attitude;
	float output;

	for (uint8_t i = 0; i < MAX_AXES; i++) {

		// Get the current attitude from the selected axis
		switch (i) {
		case ROLL:
			AttitudeActualRollGet(&attitude);
			break;
		case PITCH:
			AttitudeActualPitchGet(&attitude);
			break;
		case YAW:
			AttitudeActualYawGet(&attitude);
			break;
		}
		float rt_ms = (float)settings->AttitudeFilter;
		csd->attitude_filtered[i] = (rt_ms / (rt_ms + dT_ms)) * csd->attitude_filtered[i] + (dT_ms / (rt_ms + dT_ms)) * attitude;
		attitude = csd->attitude_filtered[i];

		// Compute new smoothed setpoint
		if (settings->Input[i] != CAMERASTABSETTINGS_INPUT_NONE && settings->Input[i] != CAMERASTABSETTINGS_INPUT_POI) {
			if (AccessoryDesiredInstGet(settings->Input[i] - CAMERASTABSETTINGS_INPUT_ACCESSORY0, &accessory) == 0) {
				float input;
				float input_rate;
				rt_ms = (float) settings->InputFilter;
				switch (settings->StabilizationMode[i]) {
				case CAMERASTABSETTINGS_STABILIZATIONMODE_ATTITUDE:
					input = accessory.AccessoryVal * settings->InputRange[i];
					csd->inputs[i] = (rt_ms / (rt_ms + dT_ms)) * csd->inputs[i] + (dT_ms / (rt_ms + dT_ms)) * input;
					break;
				case CAMERASTABSETTINGS_STABILIZATIONMODE_AXISLOCK:
					input_rate = accessory.AccessoryVal * settings->InputRate[i];
					if (fabsf(input_rate) > settings->MaxAxisLockRate)
						csd->inputs[i] = bound_sym(csd->inputs[i] + input_rate * dT_ms / 1000.0f, settings->InputRange[i]);
					break;
				default:
					input = 0;
				}
			}
			switch(i) {
			case PITCH:
				CameraDesiredDeclinationSet(&csd->inputs[i]);
				break;
			case YAW:
				CameraDesiredBearingSet(&csd->inputs[i]);
				break;
			default:
				break;
			}
		}
#if defined(CAMERASTAB_POI_MODE)		
		else if (settings->Input[i] == CAMERASTABSETTINGS_INPUT_POI) {
			// Process any updates of the tablet location if it wants to
			// be the POI
			tablet_info_process();

			PositionActualData positionActual;
			PositionActualGet(&positionActual);
			PoiLocationData poi;
			PoiLocationGet(&poi);

			float dLoc[3];

			dLoc[0] = poi.North - positionActual.North;
			dLoc[1] = poi.East - positionActual.East;
			dLoc[2] = poi.Down - positionActual.Down;

			// Compute the pitch and yaw to the POI location, assuming UAVO is level facing north
			float distance = sqrtf(powf(dLoc[0], 2) + powf(dLoc[1], 2));
			float pitch = atan2f(-dLoc[2], distance) * RAD2DEG;
			float yaw = atan2f(dLoc[1], dLoc[0]) * RAD2DEG;
			if (yaw < 0.0f)
				yaw += 360.0f;

			// Only try and track objects more than 2 m away
			if (distance > 2) {
				switch (i) {
				case CAMERASTABSETTINGS_INPUT_ROLL:
					// Does not make sense to use position to control yaw
					break;
				case CAMERASTABSETTINGS_INPUT_PITCH:
					// Store the absolute declination relative to UAV
					CameraDesiredDeclinationSet(&pitch);
					csd->inputs[CAMERASTABSETTINGS_INPUT_PITCH] = -pitch;
					break;
				case CAMERASTABSETTINGS_INPUT_YAW:
					CameraDesiredBearingSet(&yaw);
					csd->inputs[CAMERASTABSETTINGS_INPUT_YAW] = yaw;
					break;
				}
			}
		}
#endif /* CAMERASTAB_POI_MODE */		

		// Add Servo FeedForward
		applyFF(i, dT_ms, &attitude, settings);

		// Set output channels
		output = bound_sym((attitude + csd->inputs[i]) / settings->OutputRange[i], 1.0f);
		if (thisSysTime > LOAD_DELAY) {
			switch (i) {
			case ROLL:
				CameraDesiredRollSet(&output);
				break;
			case PITCH:
				CameraDesiredPitchSet(&output);
				break;
			case YAW:
				CameraDesiredYawSet(&output);
				break;
			}
		}
	}

	// Send a message over CAN, if include
	gimbal_can_message();
}

/**
 * Apply feedforward compensation to the outputs to get a faster response from
 * the servos.  This code is separate from the code in the mixer to allow different
 * time constants to be used, although for cameras a time constant of 0 is common.
 * The code tracks the difference in the attitude between calls and adds a scaling
 * of the difference to the accumulated "boost".
 */
static void applyFF(uint8_t index, float dT_ms, float *attitude, CameraStabSettingsData* cameraStab)
{
	float accumulator = csd->FFfilterAccumulator[index];

	accumulator += (*attitude - csd->FFlastAttitude[index]) * cameraStab->FeedForward[index];
	csd->FFlastAttitude[index] = *attitude;
	*attitude += accumulator;

	float filter = (float) cameraStab->FeedForwardTime / dT_ms;
	if (filter > 1)
		accumulator -= accumulator / filter;
	else
		accumulator = 0;

	// For non-zero filter times store accumulation
	csd->FFfilterAccumulator[index] = accumulator;
	*attitude += accumulator;

	//acceleration and deceleration limit
	float delta = *attitude - csd->FFlastFilteredAttitude[index];
	float maxDelta = cameraStab->MaxAccel * dT_ms / 1000.0f;
	if(fabsf(delta) > maxDelta) //we are accelerating too hard
	{
		*attitude = csd->FFlastFilteredAttitude[index] + (delta > 0 ? maxDelta : - maxDelta);
	}
	csd->FFlastFilteredAttitude[index] = *attitude;
}

/**
 * Called when the settings are updated to store a local copy
 * @param[in] ev The update event
 */
static void settings_updated_cb(UAVObjEvent * ev)
{
	CameraStabSettingsGet(&csd->settings);
}

#if defined(CAMERASTAB_POI_MODE)
/**
 * When the tablet info changes update the POI location to match
 * the current tablet location
 */
static void tablet_info_flag_update(UAVObjEvent * ev)
{
	if (ev->obj == NULL || ev->obj != TabletInfoHandle())
		return;

	tablet_info_updated = true;
}

/**
 * @brief Convert tablet location (LLA) to local coordinates (NED)
*/
static void tablet_info_process() {

	// Only process new information
	if (tablet_info_updated == false)
		return;
	tablet_info_updated = false;

	TabletInfoData tablet;
	TabletInfoGet(&tablet);
	if (tablet.Connected == TABLETINFO_CONNECTED_TRUE && tablet.POI == TABLETINFO_POI_TRUE)
	{
		HomeLocationData homeLocation;
		HomeLocationGet(&homeLocation);

		PoiLocationData poi;

		float lat, alt;
		lat = homeLocation.Latitude / 10.0e6f * DEG2RAD;
		alt = homeLocation.Altitude;

		float T[3];
		T[0] = alt+6.378137E6f;
		T[1] = cosf(lat)*(alt+6.378137E6f);
		T[2] = -1.0f;

		float dL[3] = {(tablet.Latitude - homeLocation.Latitude) / 10.0e6f * DEG2RAD,
			(tablet.Longitude - homeLocation.Longitude) / 10.0e6f * DEG2RAD,
			(tablet.Altitude)};

		poi.North = T[0] * dL[0];
		poi.East = T[1] * dL[1];
		poi.Down = T[2] * dL[2];
		PoiLocationSet(&poi);
	}
}

#endif /* CAMERASTAB_POI_MODE */


#if defined(PIOS_INCLUDE_CAN)
extern uintptr_t pios_can_id;
#endif /* PIOS_INCLUDE_CAN */

/**
 * Relay control messages to an external gimbal
 */
static void gimbal_can_message()
{
#if defined(PIOS_INCLUDE_CAN)
	AttitudeActualData attitude;
	AttitudeActualGet(&attitude);

	CameraDesiredData cameraDesired;
	CameraDesiredGet(&cameraDesired);

	struct pios_can_gimbal_message bgc_message = {
		.fc_roll = attitude.Roll,
		.fc_pitch = attitude.Pitch,
		.fc_yaw = attitude.Yaw,
		.setpoint_roll = 0,
		.setpoint_pitch = cameraDesired.Declination,
		.setpoint_yaw = cameraDesired.Bearing
	};

	PIOS_CAN_TxData(pios_can_id, PIOS_CAN_GIMBAL, (uint8_t *) &bgc_message);
#endif /* PIOS_INCLUDE_CAN */
}


/**
 * @}
 * @}
 */
