/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup OsdCan OSD CAN bus interface
 * @{
 *
 * @file       osdcan.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2016
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
#include "gpsposition.h"
#include "gpsvelocity.h"
#include "modulesettings.h"
#include "positionactual.h"
#include "manualcontrolcommand.h"
#include "systemalarms.h"
#include "taskinfo.h"
#include "velocityactual.h"


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
static struct pios_queue *queue_battery_volt; // for CAN updates from OSD
static struct pios_thread *taskHandle;

// PIOS CAN driver handle
#if defined(PIOS_INCLUDE_CAN)
extern uintptr_t pios_can_id;
#endif /* PIOS_INCLUDE_CAN */


/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t OsdCanInitialize(void)
{
	module_enabled = (pios_can_id != 0);

	if (!module_enabled)
		return -1;

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

		// Only connect messages when modules are enabled
		uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
		ModuleSettingsAdminStateGet(module_state);

		FlightStatusConnectQueue(queue);
		BaroAltitudeConnectQueue(queue);
		ManualControlCommandConnectQueue(queue);
		if (GPSPositionHandle() && module_state[MODULESETTINGS_ADMINSTATE_GPS] == MODULESETTINGS_ADMINSTATE_ENABLED) {
			GPSPositionConnectQueue(queue);
		}
		if (GPSVelocityHandle() && module_state[MODULESETTINGS_ADMINSTATE_GPS] == MODULESETTINGS_ADMINSTATE_ENABLED) {
			GPSVelocityConnectQueue(queue);
		}
		if (FlightBatteryStateHandle() && module_state[MODULESETTINGS_ADMINSTATE_BATTERY] == MODULESETTINGS_ADMINSTATE_ENABLED) {
			FlightBatteryStateConnectQueue(queue);
		} else {
			// Listen for battery voltage updates from the FC
			FlightBatteryStateInitialize();
			queue_battery_volt = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_BATTERY_VOLT);
		}
		PositionActualConnectQueue(queue);
		SystemAlarmsConnectQueue(queue);


		taskHandle = PIOS_Thread_Create(osdCanTask, "OsdCan", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
		TaskMonitorAdd(TASKINFO_RUNNING_ONSCREENDISPLAYCOM, taskHandle);
	}

	return 0;
}

MODULE_INITCALL(OsdCanInitialize, OsdCanStart)

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

			// If battery monitor is not running on FC then we are getting updates from
			// the OSD and should not echo back
			uint8_t task_running[TASKINFO_RUNNING_NUMELEM];
			TaskInfoRunningGet(task_running);
			if (task_running[TASKINFO_RUNNING_BATTERY] == TASKINFO_RUNNING_FALSE)
				break;

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

		} else if (ev.obj == GPSPositionHandle()) {

			GPSPositionData gps;
			GPSPositionGet(&gps);

			static uint32_t divider = 0;
			if (divider % 3 == 0) {
				struct pios_can_gps_latlon latlon = {
					.lat  = gps.Latitude,
					.lon = gps.Longitude,
				};
				PIOS_CAN_TxData(pios_can_id, PIOS_CAN_GPS_LATLON, (uint8_t *) &latlon);
			} else if (divider % 3 == 1) {
				struct pios_can_gps_alt_speed altspeed = {
					.alt  = gps.Altitude,
					.speed = gps.Groundspeed,
				};
				PIOS_CAN_TxData(pios_can_id, PIOS_CAN_GPS_ALTSPEED, (uint8_t *) &altspeed);
			} else if (divider % 3 == 2) {
				struct pios_can_gps_fix fix = {
					.pdop  = gps.PDOP,
					.sats = gps.Satellites,
					.status = gps.Status,
				};
				PIOS_CAN_TxData(pios_can_id, PIOS_CAN_GPS_FIX, (uint8_t *) &fix);
			}

			divider++;
		} else if (ev.obj == GPSVelocityHandle()) {

			GPSVelocityData gpsVel;
			GPSVelocityGet(&gpsVel);
			struct pios_can_gps_vel vel = {
				.north = gpsVel.North,
				.east = gpsVel.East
			};
			PIOS_CAN_TxData(pios_can_id, PIOS_CAN_GPS_VEL, (uint8_t *) &vel);

		} else if (ev.obj == PositionActualHandle()) {

			static uint32_t divider = 0;
			divider++;

			if (divider % 100 == 0) {
				PositionActualData posActual;
				PositionActualGet(&posActual);
				struct pios_can_pos pos = {
					.north = posActual.North,
					.east = posActual.East
				};
				PIOS_CAN_TxData(pios_can_id, PIOS_CAN_POS, (uint8_t *) &pos);
			}

			if (divider % 100 == 50) {
				float down;
				PositionActualDownGet(&down);
				float sinkrate;
				VelocityActualDownGet(&sinkrate);
				struct pios_can_vert vert = {
					.pos_down = down,
					.rate_down = sinkrate
				};
				PIOS_CAN_TxData(pios_can_id, PIOS_CAN_VERT, (uint8_t *) &vert);
			}

		} else if (ev.obj == ManualControlCommandHandle()) {

			static uint32_t divider = 0;
			if (divider++ % 100 != 0)
				continue;

			int16_t rssi_val;
			ManualControlCommandRssiGet(&rssi_val);

			struct pios_can_rssi_message rssi = {
				.rssi = rssi_val
			};

			PIOS_CAN_TxData(pios_can_id, PIOS_CAN_RSSI, (uint8_t *) &rssi);

		} else if (ev.obj == SystemAlarmsHandle()) {

			uint8_t alarm_status[SYSTEMALARMS_ALARM_NUMELEM];
			SystemAlarmsAlarmGet(alarm_status);

			struct pios_can_alarm_message pios_can_alarm_message = {
				.alarms = {0,0,0,0,0,0,0,0}
			};

			// Pack alarms into 2 bit fields. We collapse error and critical to error
			// as OSD represents them the same.
			for (int32_t i = 0; i < SYSTEMALARMS_ALARM_NUMELEM && i < 32; i++) {
				int32_t idx = i / 4;
				int32_t bit = (i % 4) * 2;

				uint8_t val = 0;
				switch(alarm_status[i]) {
					case SYSTEMALARMS_ALARM_UNINITIALISED:
						val = 0;
						break;
					case SYSTEMALARMS_ALARM_OK:
						val = 1;
						break;
					case SYSTEMALARMS_ALARM_WARNING:
						val = 2;
						break;
					case SYSTEMALARMS_ALARM_ERROR:
					case SYSTEMALARMS_ALARM_CRITICAL:
						val = 3;
						break;

				}
				pios_can_alarm_message.alarms[idx] |= (val << bit);
			}

			PIOS_CAN_TxData(pios_can_id, PIOS_CAN_ALARM, (uint8_t *) &pios_can_alarm_message);

		}

		// If we receive an update here the OSD is in charge of monitoring battery voltage
		struct pios_can_volt_message pios_can_volt_message;
		if (PIOS_Queue_Receive(queue_battery_volt, &pios_can_volt_message, 0) == true) {
			FlightBatteryStateVoltageSet(&pios_can_volt_message.volt);
		}


#endif /* PIOS_INCLUDE_CAN */

	}
}


/**
 * @}
 * @}
 */
