/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup OsdCan OSD CAN bus interface
 * @{
 *
 * @file       osdcan.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
 * @brief      Relay messages between OSD and FC
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
#include "pios_can.h"

#include "attitudeactual.h"

//
// Configuration
//
#define SAMPLE_PERIOD_MS     10
#define LOAD_DELAY           7000

// Private functions
static void attitudeUpdated(UAVObjEvent* ev);

// Private variables
static bool module_enabled;

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t OsdCanInitialize(void)
{
	module_enabled = true;

	// TODO: setting to enable or disable

	if (module_enabled) {

		AttitudeActualInitialize();

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
int32_t OsdCanStart(void)
{
	return 0;
}

MODULE_INITCALL(OsdCanInitialize, OsdCanStart)

#if defined(PIOS_INCLUDE_CAN)
extern uintptr_t pios_can_id;
#endif /* PIOS_INCLUDE_CAN */

/**
 * Periodic callback that processes changes in the attitude
 * and recalculates the desied gimbal angle.
 */
static void attitudeUpdated(UAVObjEvent* ev)
{	
	if (ev->obj != AttitudeActualHandle())
		return;

#if defined(PIOS_INCLUDE_CAN)

	PIOS_LED_Toggle(PIOS_LED_LINK);
	
	AttitudeActualData attitude;
	AttitudeActualGet(&attitude);

	struct pios_can_roll_pitch_message pios_can_roll_pitch_message = {
		.fc_roll = attitude.Roll,
		.fc_pitch = attitude.Pitch
	};

	PIOS_CAN_TxData(pios_can_id, PIOS_CAN_ATTITUDE_ROLL_PITCH, (uint8_t *) &pios_can_roll_pitch_message);

#endif /* PIOS_INCLUDE_CAN */

}


/**
 * @}
 * @}
 */
