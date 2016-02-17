/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{
 * @addtogroup UAVOMSPBridge UAVO to MSP Bridge Module
 * @{
 *
 * @file       mspuavobridge.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015-2016
 * @brief      Queries a MWOSD stream and populates appropriate UAOVs
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
#include "physical_constants.h"
#include "modulesettings.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "manualcontrolcommand.h"
#include "accessorydesired.h"
#include "attitudeactual.h"
#include "airspeedactual.h"
#include "actuatorsettings.h"
#include "actuatordesired.h"
#include "flightstatus.h"
#include "systemstats.h"
#include "systemalarms.h"
#include "homelocation.h"
#include "baroaltitude.h"
#include "pios_thread.h"
#include "pios_sensors.h"

#include "baroaltitude.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "modulesettings.h"

#include "msplib.h"

#if defined(PIOS_INCLUDE_MSP_BRIDGE)

#define STACK_SIZE_BYTES 700
#define TASK_PRIORITY               PIOS_THREAD_PRIO_LOW

static bool module_enabled;
extern uintptr_t pios_com_msp_id;
static struct msp_bridge *msp;
static int32_t MSPuavoBridgeInitialize(void);
static void MSPuavoBridgeTask(void *parameters);
static void setMSPSpeed(struct msp_bridge *m);


static void unpack_attitude(const struct msp_packet_attitude *attitude)
{
	AttitudeActualData attActual;
	AttitudeActualGet(&attActual);
	attActual.Roll = attitude->x * 0.1f;
	attActual.Pitch = attitude->y * -0.1f;
	attActual.Yaw = attitude->h;
	AttitudeActualSet(&attActual);
}

static bool msp_response_cb(uint8_t cmd, const uint8_t *data, size_t len)
{
	union msp_data msp_data;
	memcpy(msp_data.data, data, len);

	switch(cmd) {
	case MSP_ATTITUDE:
		unpack_attitude(&msp_data.attitude);
		break;
	}

	return true;
}

/**
 * Module start routine automatically called after initialization routine
 * @return 0 when was successful
 */
static int32_t MSPuavoBridgeStart(void)
{
	if (!module_enabled) {
		return -1;
	}

	struct pios_thread *task = PIOS_Thread_Create(MSPuavoBridgeTask, "MSPuavoBridge",
	                                              STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
	TaskMonitorAdd(TASKINFO_RUNNING_UAVOMSPBRIDGE, task);

	return 0;
}

/**
 * Module initialization routine
 * @return 0 when initialization was successful
 */
static int32_t MSPuavoBridgeInitialize(void)
{
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);

#ifdef MODULE_MSPUAVOBridge_BUILTIN
	module_enabled = true;
#else
	module_enabled = module_state[MODULESETTINGS_ADMINSTATE_UAVOMSPBRIDGE] == MODULESETTINGS_ADMINSTATE_ENABLED;
#endif

	module_enabled &= (pios_com_msp_id != 0);
	if (module_enabled) {

		msp = msp_init(pios_com_msp_id);
		if (msp != NULL) {
			setMSPSpeed(msp);
			msp_set_response_cb(msp, msp_response_cb);

			return 0;
		}
	}

	return -1;
}
MODULE_INITCALL(MSPuavoBridgeInitialize, MSPuavoBridgeStart)

/**
 * Main task routine
 * @param[in] parameters parameter given by PIOS_Thread_Create()
 */
static void MSPuavoBridgeTask(void *parameters)
{
	uint32_t i = 0;

	while(1) {
		uint8_t b = 0;
		uint16_t count = PIOS_COM_ReceiveBuffer(msp->com, &b, 1, 1);
		if (count) {
			msp_receive_byte(msp, b);
		}

		if (msp->state == MSP_IDLE) {
			if ((i++ % 20) == 0) {
				msp_send_request(msp, MSP_ATTITUDE);
			}
		}
	}
}

static void setMSPSpeed(struct msp_bridge *m)
{
	if (m->com) {
		uint8_t speed;
		ModuleSettingsMSPSpeedGet(&speed);

		switch (speed) {
		case MODULESETTINGS_MSPSPEED_2400:
			PIOS_COM_ChangeBaud(m->com, 2400);
			break;
		case MODULESETTINGS_MSPSPEED_4800:
			PIOS_COM_ChangeBaud(m->com, 4800);
			break;
		case MODULESETTINGS_MSPSPEED_9600:
			PIOS_COM_ChangeBaud(m->com, 9600);
			break;
		case MODULESETTINGS_MSPSPEED_19200:
			PIOS_COM_ChangeBaud(m->com, 19200);
			break;
		case MODULESETTINGS_MSPSPEED_38400:
			PIOS_COM_ChangeBaud(m->com, 38400);
			break;
		case MODULESETTINGS_MSPSPEED_57600:
			PIOS_COM_ChangeBaud(m->com, 57600);
			break;
		case MODULESETTINGS_MSPSPEED_115200:
			PIOS_COM_ChangeBaud(m->com, 115200);
			break;
		}
	}
}

#endif //PIOS_INCLUDE_MSP_BRIDGE
/**
 * @}
 * @}
 */
