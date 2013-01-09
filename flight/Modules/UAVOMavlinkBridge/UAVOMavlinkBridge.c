/**
 ******************************************************************************
 * @addtogroup OpenPilotModules OpenPilot Modules
 * @{ 
 * @addtogroup UAVOMavlinkBridge UAVO to Mavlink Bridge Module
 * @brief Bridge UAVObjects with MavLink data
 * @{ 
 *
 * @file       UAVOMavlinkBridge.c
 * @author     The TauLabs Team, http://www.taulabs.org Copyright (C) 2013.
 * @brief      Bridges selected UAVObjects to Mavlink
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

// ****************
#define PI					3.14159265358979f

#include "openpilot.h"
#include "modulesettings.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "manualcontrolcommand.h"
#include "attitudeactual.h"
#include "airspeedactual.h"
#include "actuatordesired.h"
#include "flightstatus.h"
#include "mavlink.h"


enum MY_MAV_DATA_STREAM {
	MY_MAV_DATA_STREAM_RAW_SENSORS = 0, /* Enable IMU_RAW, GPS_RAW, GPS_STATUS packets. | */
	MY_MAV_DATA_STREAM_EXTENDED_STATUS, /* Enable GPS_STATUS, CONTROL_STATUS, AUX_STATUS | */
	MY_MAV_DATA_STREAM_RC_CHANNELS, /* Enable RC_CHANNELS_SCALED, RC_CHANNELS_RAW, SERVO_OUTPUT_RAW | */
	MY_MAV_DATA_STREAM_POSITION, /* Enable LOCAL_POSITION, GLOBAL_POSITION/GLOBAL_POSITION_INT messages. | */
	MY_MAV_DATA_STREAM_EXTRA1, /* Dependent on the autopilot | */
	MY_MAV_DATA_STREAM_EXTRA2, /* Dependent on the autopilot | */
};

// ****************
// Private functions

static void uavoMavlingBridgeTask(void *parameters);
static bool stream_trigger(enum MAV_DATA_STREAM stream_num);

// ****************
// Private constants

#define STACK_SIZE_BYTES            600
#define TASK_PRIORITY                   (tskIDLE_PRIORITY + 1)
#define TASK_RATE					10

#define MAXSTREAMS 6
static const uint8_t MAVStreams[MAXSTREAMS] = { MY_MAV_DATA_STREAM_RAW_SENSORS, //2Hz
		MY_MAV_DATA_STREAM_EXTENDED_STATUS, //2Hz
		MY_MAV_DATA_STREAM_RC_CHANNELS, //5Hz
		MY_MAV_DATA_STREAM_POSITION, //2Hz
		MY_MAV_DATA_STREAM_EXTRA1, //5Hz
		MY_MAV_DATA_STREAM_EXTRA2 }; //2Hz
static const uint16_t MAVRates[MAXSTREAMS] = { 0x02, 0x02, 0x05, 0x02, 0x0A,
		0x02 };

// ****************
// Private variables

static xTaskHandle uavoMavlingBridgeTaskHandle;

static uint32_t mavlink_port;

static bool module_enabled = false;

static uint8_t * stream_ticks;

static mavlink_message_t mavMsg;

static uint8_t * serial_buf;

/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t uavoMavlingBridgeStart(void) {
	if (module_enabled) {
		// Start tasks
		xTaskCreate(uavoMavlingBridgeTask, (signed char *) "uavoMavlinkBridge",
				STACK_SIZE_BYTES / 4, NULL, TASK_PRIORITY,
				&uavoMavlingBridgeTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_COM2USBBRIDGE,
				uavoMavlingBridgeTaskHandle);
		return 0;
	}
	return -1;
}
/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t uavoMavlingBridgeInitialize(void) {
	mavlink_port = PIOS_COM_MAVLINK;

	ModuleSettingsInitialize();
	uint8_t module_state[MODULESETTINGS_STATE_NUMELEM];
	ModuleSettingsStateGet(module_state);
	if (mavlink_port
			&& (module_state[MODULESETTINGS_STATE_UAVOMAVLINKBRIDGE]
					== MODULESETTINGS_STATE_ENABLED)) {
		module_enabled = true;
		PIOS_COM_ChangeBaud(mavlink_port, 57600);
#ifdef TEST_GPS
		FlightBatteryStateInitialize();
		FlightBatterySettingsInitialize();
		GPSPositionInitialize();
#endif
		serial_buf = pvPortMalloc(MAVLINK_MAX_PACKET_LEN);
		stream_ticks = pvPortMalloc(MAXSTREAMS);
		for (int x = 0; x < MAXSTREAMS; ++x) {
			stream_ticks[x] = (TASK_RATE / MAVRates[x]);
		}
	} else {
		module_enabled = false;
	}
	return 0;
}
MODULE_INITCALL( uavoMavlingBridgeInitialize, uavoMavlingBridgeStart)

/**
 * Main task. It does not return.
 */

static void uavoMavlingBridgeTask(void *parameters) {
	FlightBatterySettingsData batSettings;
	FlightBatteryStateData batState;
	GPSPositionData gpsPosData;
	ManualControlCommandData manualState;
	AttitudeActualData attActual;
	AirspeedActualData airspeedActual;
	ActuatorDesiredData actDesired;
	FlightStatusData flightStatus;

	if (FlightBatterySettingsHandle() != NULL )
		FlightBatterySettingsGet(&batSettings);

	uint16_t msg_lenght;
	uint8_t armed_mode;
	portTickType lastSysTime;
	// Main task loop
	lastSysTime = xTaskGetTickCount();
	while (1) {
		if (stream_trigger(MY_MAV_DATA_STREAM_EXTENDED_STATUS)) {
			if (FlightBatteryStateHandle() != NULL )
				FlightBatteryStateGet(&batState);
			if (GPSPositionHandle() != NULL )
				GPSPositionGet(&gpsPosData);
			mavlink_msg_sys_status_pack(0, 200, &mavMsg, 0, 0, 0, 0,
					batState.Voltage * 1000, batState.Current * 100,
					batState.ConsumedEnergy / batSettings.Capacity * 100, 0, 0,
					0, 0, 0, 0);
			msg_lenght = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_lenght);
			mavlink_msg_gps_raw_int_pack(0, 200, &mavMsg, 0,
					gpsPosData.Status - 1, gpsPosData.Latitude*10^-7,
					gpsPosData.Longitude*10^-7, gpsPosData.Altitude, 0, 0, 0, 0,
					gpsPosData.Satellites);
			msg_lenght = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_lenght);

			//TODO add waypoint nav stuff
			//wp_target_bearing
			//wp_dist = mavlink_msg_nav_controller_output_get_wp_dist(&msg);
			//alt_error = mavlink_msg_nav_controller_output_get_alt_error(&msg);
			//aspd_error = mavlink_msg_nav_controller_output_get_aspd_error(&msg);
			//xtrack_error = mavlink_msg_nav_controller_output_get_xtrack_error(&msg);
			//mavlink_msg_nav_controller_output_pack
			//wp_number
			//mavlink_msg_mission_current_pack
		}

		if (stream_trigger(MY_MAV_DATA_STREAM_RC_CHANNELS)) {
			ManualControlCommandGet(&manualState);
			FlightStatusGet(&flightStatus);
			//TODO connect with RSSI object and pass in last argument
			mavlink_msg_rc_channels_raw_pack(0, 200, &mavMsg, 0, 0,
					manualState.Channel[0], manualState.Channel[1],
					manualState.Channel[2], manualState.Channel[3],
					manualState.Channel[4], manualState.Channel[5],
					manualState.Channel[6], manualState.Channel[7], 0);
			msg_lenght = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_lenght);
		}

		if (stream_trigger(MY_MAV_DATA_STREAM_EXTRA1)) {
			AttitudeActualGet(&attActual);
			mavlink_msg_attitude_pack(0, 200, &mavMsg, 0,
					attActual.Roll * PI / 180, attActual.Pitch * PI / 180,
					attActual.Yaw * PI / 180, 0, 0, 0);
			msg_lenght = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_lenght);
		}

		if (stream_trigger(MY_MAV_DATA_STREAM_EXTRA2)) {
			if (AirspeedActualHandle() != NULL )
				AirspeedActualGet(&airspeedActual);
			if (GPSPositionHandle() != NULL )
				GPSPositionGet(&gpsPosData);
			ActuatorDesiredGet(&actDesired);

			mavlink_msg_vfr_hud_pack(0, 200, &mavMsg,
					airspeedActual.TrueAirspeed, gpsPosData.Groundspeed,
					gpsPosData.Heading, actDesired.Throttle,
					gpsPosData.Altitude, 0);
			msg_lenght = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_lenght);
			if (flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMED)
				armed_mode = MAV_MODE_FLAG_SAFETY_ARMED;
			else
				armed_mode = 0;
			mavlink_msg_heartbeat_pack(0, 200, &mavMsg, MAV_TYPE_FIXED_WING,
					MAV_AUTOPILOT_GENERIC, armed_mode, 0, 0);
			msg_lenght = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_lenght);
		}
		vTaskDelayUntil(&lastSysTime, (1000 / TASK_RATE) / portTICK_RATE_MS);
	}
}

static bool stream_trigger(enum MAV_DATA_STREAM stream_num) {
	uint8_t rate = (uint8_t) MAVRates[stream_num];

	if (rate == 0) {
		return false;
	}

	if (stream_ticks[stream_num] == 0) {
		// we're triggering now, setup the next trigger point
		if (rate > TASK_RATE) {
			rate = TASK_RATE;
		}
		stream_ticks[stream_num] = (TASK_RATE / rate);
		return true;
	}
	// count down at 50Hz
	stream_ticks[stream_num]--;
	return false;
}
