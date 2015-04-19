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
#include "baroaltitude.h"
#include "flightbatterystate.h"
#include "flightstatus.h"
#include "rfm22bstatus.h"


//
// Configuration
//
#define STACK_SIZE_BYTES 1312
#define TASK_PRIORITY PIOS_THREAD_PRIO_NORMAL
#define SAMPLE_PERIOD_MS     10

// Private functions
static void osdCanTask(void* parameters);

// Private variables
static bool module_enabled;
static struct pios_queue *queue;
static struct pios_thread *taskHandle;

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t OsdCanInitialize(void)
{
	module_enabled = true;

	// TODO: setting to enable or disable
	queue = PIOS_Queue_Create(3, sizeof(UAVObjEvent));

	if (queue == NULL)
		module_enabled = false;

	return -1;
}

/* stub: module has no module thread */
int32_t OsdCanStart(void)
{
	if (module_enabled) {

		UAVObjEvent ev = {
			.obj = AttitudeActualHandle(),
			.instId = 0,
			.event = 0,
		};
		EventPeriodicQueueCreate(&ev, queue, SAMPLE_PERIOD_MS);

		FlightStatusConnectQueue(queue);
		BaroAltitudeConnectQueue(queue);
		if (FlightBatteryStateHandle())
			FlightBatteryStateConnectQueue(queue);
		if (RFM22BStatusHandle())
			RFM22BStatusConnectQueue(queue);

		taskHandle = PIOS_Thread_Create(osdCanTask, "OsdCan", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
	}

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
static void osdCanTask(void* parameters)
{	
	UAVObjEvent ev;

	while(1) {

		if (PIOS_Queue_Receive(queue, &ev, SAMPLE_PERIOD_MS) == false)
			continue;

#if defined(PIOS_INCLUDE_CAN)
		if (ev.obj == AttitudeActualHandle()) {

			
			AttitudeActualData attitude;
			AttitudeActualGet(&attitude);

			struct pios_can_roll_pitch_message pios_can_roll_pitch_message = {
				.fc_roll = attitude.Roll,
				.fc_pitch = attitude.Pitch
			};

			PIOS_CAN_TxData(pios_can_id, PIOS_CAN_ATTITUDE_ROLL_PITCH, (uint8_t *) &pios_can_roll_pitch_message);

			struct pios_can_yaw_message pios_can_yaw_message = {
				.fc_yaw = attitude.Yaw,
			};

			PIOS_CAN_TxData(pios_can_id, PIOS_CAN_ATTITUDE_YAW, (uint8_t *) &pios_can_yaw_message);

		} else if (ev.obj == FlightStatusHandle()) {

			FlightStatusData flightStatus;
			FlightStatusGet(&flightStatus);

			struct pios_can_flightstatus_message flighstatus = {
				.flight_mode = flightStatus.FlightMode,
				.armed = flightStatus.Armed
			};

			PIOS_CAN_TxData(pios_can_id, PIOS_CAN_FLIGHTSTATUS, (uint8_t *) &flighstatus);

		} else if (ev.obj == FlightBatteryStateHandle()) {

			FlightBatteryStateData flightBattery;
			FlightBatteryStateGet(&flightBattery);

			struct pios_can_volt_message volt = {
				.volt = flightBattery.Voltage
			};

			PIOS_CAN_TxData(pios_can_id, PIOS_CAN_BATTERY_VOLT, (uint8_t *) &volt);

			struct pios_can_curr_message curr = {
				.curr = flightBattery.Current,
				.consumed = flightBattery.ConsumedEnergy
			};

			PIOS_CAN_TxData(pios_can_id, PIOS_CAN_BATTERY_CURR, (uint8_t *) &curr);

		} else if (ev.obj == BaroAltitudeHandle()) {

			static uint32_t divider = 0;
			if (divider++ % 100 != 0)
				continue;

			float alt;
			BaroAltitudeAltitudeGet(&alt);

			struct pios_can_alt_message baro = {
				.fc_alt = alt
			};

			PIOS_CAN_TxData(pios_can_id, PIOS_CAN_ALT, (uint8_t *) &baro);

		} else if (ev.obj == RFM22BStatusHandle()) {

			RFM22BStatusData rfm22b;
			RFM22BStatusInstGet(1, &rfm22b);

			struct pios_can_rssi_message rssi = {
				.rssi = rfm22b.LinkQuality
			};

			PIOS_CAN_TxData(pios_can_id, PIOS_CAN_RSSI, (uint8_t *) &rssi);

		}

#endif /* PIOS_INCLUDE_CAN */

	}
}


/**
 * @}
 * @}
 */
