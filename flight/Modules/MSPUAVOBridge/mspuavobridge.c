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

static void unpack_status(const struct msp_packet_status *status)
{
	FlightStatusArmedOptions armed = (status->flags & 0x01) ? FLIGHTSTATUS_ARMED_ARMED : FLIGHTSTATUS_ARMED_DISARMED;
	FlightStatusArmedSet(&armed);

	FlightStatusFlightModeOptions mode =  FLIGHTSTATUS_FLIGHTMODE_MANUAL;
	for (uint32_t i = 1; msp_boxes[i].mode != MSP_BOX_LAST && i < NELEMENTS(msp_boxes); i++) {
		if (status->flags & (1 << i)) {
			mode = msp_boxes[i].tlmode;
			FlightStatusFlightModeSet(&mode);
			break;
		}
	}
}

static void unpack_attitude(const struct msp_packet_attitude *attitude)
{
	AttitudeActualData attActual;
	AttitudeActualGet(&attActual);
	attActual.Roll = attitude->x * 0.1f;
	attActual.Pitch = attitude->y * -0.1f;
	attActual.Yaw = attitude->h;
	AttitudeActualSet(&attActual);
}

static void unpack_analog(const struct msp_packet_analog *analog)
{
	// Packet contains RSSI as 0 to 1023
	int16_t rssi = analog->rssi / 10;
	if (rssi > 100) rssi = 100;
	ManualControlCommandRssiSet(&rssi);


	if (FlightBatteryStateHandle() != NULL) {


		const float voltage = analog->vbat * 0.1f;
		const float current = analog->current * 0.01f;
		const float consumed = analog->powerMeterSum;

		FlightBatteryStateData flight_battery;
		FlightBatteryStateGet(&flight_battery);
		// If settings exist the module itself is running and we are measuring
		// this and should not overwrite the voltage
		if (FlightBatterySettingsHandle() == NULL)
			flight_battery.Voltage = voltage;
		flight_battery.Current = current;
		flight_battery.ConsumedEnergy = consumed;
		FlightBatteryStateSet(&flight_battery);
	}
}

static void unpack_altitude(const struct msp_packet_altitude *altitude)
{
	float alt = altitude->alt * 0.01f;
	if (BaroAltitudeHandle())
		BaroAltitudeAltitudeSet(&alt);
}

/**
 * Callback method when a response packet is received and has correct checksum
 * unpacks the various data types into UAVOs so they can be visualized by OSD
 * @param[in] cmd the packet type
 * @param[in] data the packet data
 * @param[in] len the payload length
 @ return true if packet type known, false otherwise
 */
static bool msp_response_cb(uint8_t cmd, const uint8_t *data, size_t len)
{
	union msp_data msp_data;
	memcpy(msp_data.data, data, len);

	switch(cmd) {
	case MSP_STATUS:
		unpack_status(&msp_data.status);
		return true;
	case MSP_ATTITUDE:
		unpack_attitude(&msp_data.attitude);
		return true;
	case MSP_ANALOG:
		unpack_analog(&msp_data.analog);
		return true;
	case MSP_ALTITUDE:
		unpack_altitude(&msp_data.altitude);
		return true;
	}

	return false;
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

struct msp_message_scheduler {
	uint32_t last_message_time;
	uint32_t message_period;
	uint8_t message_id;
};

//! Array of messages and their desired periods
struct msp_message_scheduler msp_messages[] = {
	{
		.last_message_time = 0,
		.message_period = 20,
		.message_id = MSP_ATTITUDE,
	},
	{
		.last_message_time = 0,
		.message_period = 20,
		.message_id = MSP_ALTITUDE,
	},
	{
		.last_message_time = 0,
		.message_period = 50,
		.message_id = MSP_ANALOG,
	},
	{
		.last_message_time = 0,
		.message_period = 50,
		.message_id = MSP_STATUS,
	},
};

//! Return true if period has expired
static bool time_expired(uint32_t last_time, uint32_t period)
{
	uint32_t time_since_idle_ms = PIOS_Thread_Systime() - last_time;
	return time_since_idle_ms > period;
}

/**
 * Main task routine
 * @param[in] parameters parameter given by PIOS_Thread_Create()
 */
static void MSPuavoBridgeTask(void *parameters)
{
	const uint32_t MSP_IDLE_TIMEOUT = 100;

	uint32_t last_idle_ms = PIOS_Thread_Systime();

	while(1) {
		uint8_t b = 0;
		uint16_t count = PIOS_COM_ReceiveBuffer(msp->com, &b, 1, 1);
		if (count) {
			msp_receive_byte(msp, b);
		} else if (msp->state == MSP_IDLE) {
			// track last time we are in idle
			last_idle_ms = PIOS_Thread_Systime();
		} else if (time_expired(last_idle_ms, MSP_IDLE_TIMEOUT)) { 
			// check if we seem to be stalled and reset MSP parser
			msp->state = MSP_IDLE;
		}

		if (msp->state == MSP_IDLE) {

			for (int32_t i = 0; i < NELEMENTS(msp_messages); i++) {
				// Find the first message that is scheduled to be requested and do so
				if (time_expired(msp_messages[i].last_message_time, msp_messages[i].message_period)) {
					msp_send_request(msp, msp_messages[i].message_id);
					msp_messages[i].last_message_time = PIOS_Thread_Systime();
					break;
				}
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
