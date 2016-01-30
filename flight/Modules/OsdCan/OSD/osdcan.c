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
#include "manualcontrolcommand.h"
#include "modulesettings.h"
#include "positionactual.h"
#include "systemalarms.h"
#include "taskinfo.h"

// Private constants
#define MAX_QUEUE_SIZE 2

#define STACK_SIZE_BYTES 1312
#define TASK_PRIORITY PIOS_THREAD_PRIO_NORMAL

// Private types

// Private variables
static struct pios_queue *queue_roll_pitch;
static struct pios_queue *queue_yaw;
static struct pios_queue *queue_altitude;
static struct pios_queue *queue_flightstatus;
static struct pios_queue *queue_rssi;
static struct pios_queue *queue_battery_volt;
static struct pios_queue *queue_battery_curr;
static struct pios_queue *queue_gps_latlon;
static struct pios_queue *queue_gps_altspeed;
static struct pios_queue *queue_gps_fix;
static struct pios_queue *queue_gps_vel;
static struct pios_queue *queue_pos;
static struct pios_queue *queue_sysalarm;
static struct pios_thread *taskHandle;

// queue for sending updates to FC from OSD
static struct pios_queue *uavo_update_queue;

// Private functions
static void osdCanTask(void* parameters);
static void enable_battery_module();
static void enable_gps_module();

/**
 * @brief Module initialization
 * @return 0
 */
static int32_t OsdCanStart()
{
	// Start main task
	taskHandle = PIOS_Thread_Create(osdCanTask, "OsdCan", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
	TaskMonitorAdd(TASKINFO_RUNNING_ONSCREENDISPLAYCOM, taskHandle);

	// Only connect messages when modules are enabled
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (FlightBatteryStateHandle() && module_state[MODULESETTINGS_ADMINSTATE_BATTERY] == MODULESETTINGS_ADMINSTATE_ENABLED)
		FlightBatteryStateConnectQueue(uavo_update_queue);

	return 0;
}

extern uintptr_t pios_can_id;

/**
 * @brief Module initialization
 * @return 0
 */
static int32_t OsdCanInitialize()
{
	// Listen for ActuatorDesired updates (Primary input to this module)
	AttitudeActualInitialize();

	// Create object queues
	queue_roll_pitch = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_ATTITUDE_ROLL_PITCH);
	queue_yaw = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_ATTITUDE_YAW);
	queue_flightstatus = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_FLIGHTSTATUS);
	queue_altitude = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_ALT);
	queue_rssi = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_RSSI);
	queue_battery_volt = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_BATTERY_VOLT);
	queue_battery_curr = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_BATTERY_CURR);
	queue_gps_latlon = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_GPS_LATLON);
	queue_gps_altspeed = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_GPS_ALTSPEED);
	queue_gps_fix = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_GPS_FIX);
	queue_gps_vel = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_GPS_VEL);
	queue_pos = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_POS);
	queue_sysalarm = PIOS_CAN_RegisterMessageQueue(pios_can_id, PIOS_CAN_ALARM);

	uavo_update_queue = PIOS_Queue_Create(3, sizeof(UAVObjEvent));

	return 0;
}
MODULE_INITCALL(OsdCanInitialize, OsdCanStart);

/**
 * @brief Gimbal output control task
 */
static void osdCanTask(void* parameters)
{

	// Loop forever
	while (1) {

		// If memory becomes an issue can evanutally just use a common buffer
		// and case it below
		struct pios_can_roll_pitch_message roll_pitch_message;
		struct pios_can_yaw_message pios_can_yaw_message;
		struct pios_can_flightstatus_message pios_can_flightstatus_message;
		struct pios_can_alt_message pios_can_alt_message;
		struct pios_can_rssi_message pios_can_rssi_message;
		struct pios_can_volt_message pios_can_volt_message;
		struct pios_can_curr_message pios_can_curr_message;
		struct pios_can_gps_latlon pios_can_gps_latlon_message;
		struct pios_can_gps_alt_speed pios_can_gps_alt_speed_message;
		struct pios_can_gps_fix pios_can_gps_fix_message;
		struct pios_can_gps_vel pios_can_gps_vel_message;
		struct pios_can_pos pios_can_pos_message;
		uint8_t buf[8];

		// Wait for queue message
		if (PIOS_Queue_Receive(queue_roll_pitch, &roll_pitch_message, 0) == true) {
			AttitudeActualData attitudeActual;
			AttitudeActualGet(&attitudeActual);
			attitudeActual.Roll = roll_pitch_message.fc_roll;
			attitudeActual.Pitch = roll_pitch_message.fc_pitch;
			AttitudeActualSet(&attitudeActual);
		}

		if (PIOS_Queue_Receive(queue_yaw, &pios_can_yaw_message, 0) == true) {
			AttitudeActualYawSet(&pios_can_yaw_message.fc_yaw);
		}

		if (PIOS_Queue_Receive(queue_altitude, &pios_can_alt_message, 0) == true) {
			BaroAltitudeAltitudeSet(&pios_can_alt_message.fc_alt);
		}

		if (PIOS_Queue_Receive(queue_flightstatus, &pios_can_flightstatus_message, 0) == true) {
			FlightStatusData flightStatus;
			FlightStatusGet(&flightStatus);
			flightStatus.FlightMode = pios_can_flightstatus_message.flight_mode;
			flightStatus.Armed = pios_can_flightstatus_message.armed;
			FlightStatusSet(&flightStatus);
		}

		if (PIOS_Queue_Receive(queue_rssi, &pios_can_rssi_message, 0) == true) {
			ManualControlCommandRssiSet(&pios_can_rssi_message.rssi);
		}

		if (PIOS_Queue_Receive(queue_battery_volt, &pios_can_volt_message, 0) == true) {
			enable_battery_module();
			FlightBatteryStateVoltageSet(&pios_can_volt_message.volt);
		}

		if (PIOS_Queue_Receive(queue_battery_curr, &pios_can_curr_message, 0) == true) {
			enable_battery_module();
			FlightBatteryStateCurrentSet(&pios_can_curr_message.curr);
			FlightBatteryStateConsumedEnergySet(&pios_can_curr_message.consumed);
		}

		if (PIOS_Queue_Receive(queue_gps_latlon, &pios_can_gps_latlon_message, 0) == true) {
			enable_gps_module();
			GPSPositionData gpsPosition;
			GPSPositionGet(&gpsPosition);
			gpsPosition.Latitude = pios_can_gps_latlon_message.lat;
			gpsPosition.Longitude = pios_can_gps_latlon_message.lon;
			GPSPositionSet(&gpsPosition);
		}

		if (PIOS_Queue_Receive(queue_gps_altspeed, &pios_can_gps_alt_speed_message, 0) == true) {
			enable_gps_module();
			GPSPositionData gpsPosition;
			GPSPositionGet(&gpsPosition);
			gpsPosition.Altitude = pios_can_gps_alt_speed_message.alt;
			gpsPosition.Groundspeed = pios_can_gps_alt_speed_message.speed;
			GPSPositionSet(&gpsPosition);
		}

		if (PIOS_Queue_Receive(queue_gps_fix, &pios_can_gps_fix_message, 0) == true) {
			enable_gps_module();
			GPSPositionData gpsPosition;
			GPSPositionGet(&gpsPosition);
			gpsPosition.PDOP = pios_can_gps_fix_message.pdop;
			gpsPosition.Satellites = pios_can_gps_fix_message.sats;
			gpsPosition.Status = pios_can_gps_fix_message.status;
			GPSPositionSet(&gpsPosition);
		}

		if (PIOS_Queue_Receive(queue_gps_vel, &pios_can_gps_vel_message, 0) == true) {
			GPSVelocityData vel;
			GPSVelocityGet(&vel);
			vel.North = pios_can_gps_vel_message.north;
			vel.East = pios_can_gps_vel_message.east;
			GPSVelocitySet(&vel);
		}

		if (PIOS_Queue_Receive(queue_pos, &pios_can_pos_message, 0) == true) {
			PositionActualData posActual;
			PositionActualGet(&posActual);
			posActual.North = pios_can_pos_message.north;
			posActual.East = pios_can_pos_message.east;
			PositionActualSet(&posActual);
		}

		if (PIOS_Queue_Receive(queue_sysalarm, buf, 0) == true) {

			uint8_t alarm_status[SYSTEMALARMS_ALARM_NUMELEM];

			struct pios_can_alarm_message *pios_can_alarm_message = (struct pios_can_alarm_message *) buf;

			// Pack alarms into 2 bit fields. We collapse error and critical to error
			// as OSD represents them the same.
			for (int32_t i = 0; i < SYSTEMALARMS_ALARM_NUMELEM && i < 32; i++) {
				int32_t idx = i / 4;
				int32_t bit = (i % 4) * 2;

				uint8_t val = (pios_can_alarm_message->alarms[idx] >> bit) & 0x03;
				switch (val) {
				case 0:
					alarm_status[i] = SYSTEMALARMS_ALARM_UNINITIALISED;
					break;
				case 1:
					alarm_status[i] = SYSTEMALARMS_ALARM_OK;
					break;
				case 2:
					alarm_status[i] = SYSTEMALARMS_ALARM_WARNING;
					break;
				default:
					alarm_status[i] = SYSTEMALARMS_ALARM_ERROR;
					break;
				}
			}

			SystemAlarmsAlarmSet(alarm_status);

		}

		// Check if any updates should be sent to the FC. This could also be used in the future
		// for handshaking if we implement the menu
		UAVObjEvent ev;
		if (PIOS_Queue_Receive(uavo_update_queue, &ev, 0)) {
			if (ev.obj == FlightBatteryStateHandle()) {
				// If battery monitor is not running on OSD then we are getting updates from
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
			}
		}

		PIOS_Thread_Sleep(1);
	}

}

//! Flag this module as enabled when we get the appropriate
//! messages
static void enable_battery_module()
{
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
 	ModuleSettingsAdminStateGet(module_state);

 	if (module_state[MODULESETTINGS_ADMINSTATE_BATTERY] != MODULESETTINGS_ADMINSTATE_ENABLED) {
 		module_state[MODULESETTINGS_ADMINSTATE_BATTERY] = MODULESETTINGS_ADMINSTATE_ENABLED;
 		ModuleSettingsAdminStateSet(module_state);
 	}

}

//! Flag this module as enabled when we get the appropriate
//! messages
static void enable_gps_module()
{
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
 	ModuleSettingsAdminStateGet(module_state);

 	if (module_state[MODULESETTINGS_ADMINSTATE_GPS] != MODULESETTINGS_ADMINSTATE_ENABLED) {
 		module_state[MODULESETTINGS_ADMINSTATE_GPS] = MODULESETTINGS_ADMINSTATE_ENABLED;
 		ModuleSettingsAdminStateSet(module_state);
 	}

}

/**
 * @}
 * @}
 */
