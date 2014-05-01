/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup UAVOMavlinkBridge UAVO to Mavlink Bridge Module
 * @{ 
 *
 * @file       UAVOMavlinkBridge.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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
#include "openpilot.h"
#include "physical_constants.h"
#include "modulesettings.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "manualcontrolcommand.h"
#include "attitudeactual.h"
#include "airspeedactual.h"
#include "actuatordesired.h"
#include "flightstatus.h"
#include "systemstats.h"
#include "homelocation.h"
#include "baroaltitude.h"
#include "mavlink.h"

// ****************
// Private functions

static void uavoMavlinkBridgeTask(void *parameters);
static bool stream_trigger(enum MAV_DATA_STREAM stream_num);

// ****************
// Private constants

#if defined(PIOS_MAVLINK_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_MAVLINK_STACK_SIZE
#else
#define STACK_SIZE_BYTES 800
#endif

#define TASK_PRIORITY               (tskIDLE_PRIORITY + 1)
#define TASK_RATE_HZ				10

static const uint8_t mav_rates[] =
	 { [MAV_DATA_STREAM_RAW_SENSORS]=0x02, //2Hz
	   [MAV_DATA_STREAM_EXTENDED_STATUS]=0x02, //2Hz
	   [MAV_DATA_STREAM_RC_CHANNELS]=0x05, //5Hz
	   [MAV_DATA_STREAM_POSITION]=0x02, //2Hz
	   [MAV_DATA_STREAM_EXTRA1]=0x0A, //10Hz
	   [MAV_DATA_STREAM_EXTRA2]=0x02 }; //2Hz

#define MAXSTREAMS sizeof(mav_rates)

// ****************
// Private variables

static xTaskHandle uavoMavlinkBridgeTaskHandle;

static uint32_t mavlink_port;

static bool module_enabled = false;

static uint8_t * stream_ticks;

static mavlink_message_t mavMsg;

static uint8_t * serial_buf;

static void updateSettings();

/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t uavoMavlinkBridgeStart(void) {
	if (module_enabled) {
		// Start tasks
		xTaskCreate(uavoMavlinkBridgeTask, (signed char *) "uavoMavlinkBridge",
				STACK_SIZE_BYTES / 4, NULL, TASK_PRIORITY,
				&uavoMavlinkBridgeTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_UAVOMAVLINKBRIDGE,
				uavoMavlinkBridgeTaskHandle);
		return 0;
	}
	return -1;
}
/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t uavoMavlinkBridgeInitialize(void) {
	mavlink_port = PIOS_COM_MAVLINK;

	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);

	if (mavlink_port
			&& (module_state[MODULESETTINGS_ADMINSTATE_UAVOMAVLINKBRIDGE]
					== MODULESETTINGS_ADMINSTATE_ENABLED)) {
		module_enabled = true;
		updateSettings();

		serial_buf = pvPortMalloc(MAVLINK_MAX_PACKET_LEN);
		stream_ticks = pvPortMalloc(MAXSTREAMS);
		for (int x = 0; x < MAXSTREAMS; ++x) {
			stream_ticks[x] = (TASK_RATE_HZ / mav_rates[x]);
		}
	} else {
		module_enabled = false;
	}
	return 0;
}
MODULE_INITCALL( uavoMavlinkBridgeInitialize, uavoMavlinkBridgeStart)

/**
 * Main task. It does not return.
 */

static void uavoMavlinkBridgeTask(void *parameters) {
	FlightBatterySettingsData batSettings;
	FlightBatteryStateData batState;
	GPSPositionData gpsPosData;
	ManualControlCommandData manualState;
	AttitudeActualData attActual;
	AirspeedActualData airspeedActual;
	ActuatorDesiredData actDesired;
	FlightStatusData flightStatus;
	SystemStatsData systemStats;
	HomeLocationData homeLocation;
	BaroAltitudeData baroAltitude;

	if (FlightBatterySettingsHandle() != NULL )
		FlightBatterySettingsGet(&batSettings);
	else {
		batSettings.Capacity = 0;
		batSettings.NbCells = 0;
		batSettings.SensorCalibrationFactor[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONFACTOR_CURRENT] = 0;
		batSettings.SensorCalibrationFactor[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONFACTOR_VOLTAGE] = 0;
		batSettings.SensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_BATTERYCURRENT] = FLIGHTBATTERYSETTINGS_SENSORTYPE_DISABLED;
		batSettings.SensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_BATTERYVOLTAGE] = FLIGHTBATTERYSETTINGS_SENSORTYPE_DISABLED;
		batSettings.Type = FLIGHTBATTERYSETTINGS_TYPE_NONE;
		batSettings.VoltageThresholds[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_WARNING] = 0;
		batSettings.VoltageThresholds[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_ALARM] = 0;
	}

	if (GPSPositionHandle() == NULL ){
		gpsPosData.Altitude = 0;
		gpsPosData.GeoidSeparation = 0;
		gpsPosData.Groundspeed = 0;
		gpsPosData.HDOP = 0;
		gpsPosData.Heading = 0;
		gpsPosData.Latitude = 0;
		gpsPosData.Longitude = 0;
		gpsPosData.PDOP = 0;
		gpsPosData.Satellites = 0;
		gpsPosData.Status = 0;
		gpsPosData.VDOP = 0;
	}

	if (FlightBatteryStateHandle() == NULL ) {
		batState.AvgCurrent = 0;
		batState.BoardSupplyVoltage = 0;
		batState.ConsumedEnergy = 0;
		batState.Current = 0;
		batState.EstimatedFlightTime = 0;
		batState.PeakCurrent = 0;
		batState.Voltage = 0;
	}

	if (AirspeedActualHandle() == NULL ) {
		airspeedActual.CalibratedAirspeed = 0;
		airspeedActual.TrueAirspeed = 0;
		airspeedActual.alpha = 0;
		airspeedActual.beta = 0;
	}

	if (HomeLocationHandle() == NULL ) {
		homeLocation.Set=HOMELOCATION_SET_FALSE;
		homeLocation.Latitude = 0;
		homeLocation.Longitude = 0;
		homeLocation.Altitude = 0;
		homeLocation.Be[0] = 0;
		homeLocation.Be[1] = 0;
		homeLocation.Be[2] = 0;
		homeLocation.GroundTemperature = (STANDARD_AIR_TEMPERATURE - CELSIUS2KELVIN) * 10;
		homeLocation.SeaLevelPressure = STANDARD_AIR_SEA_LEVEL_PRESSURE/1000;
	}

	uint16_t msg_length;
	portTickType lastSysTime;
	// Main task loop
	lastSysTime = xTaskGetTickCount();

	while (1) {
		vTaskDelayUntil(&lastSysTime, MS2TICKS(1000 / TASK_RATE_HZ));

		if (stream_trigger(MAV_DATA_STREAM_EXTENDED_STATUS)) {
			if (FlightBatteryStateHandle() != NULL )
				FlightBatteryStateGet(&batState);
			SystemStatsGet(&systemStats);

			int8_t battery_remaining = 0;
			if (batSettings.Capacity != 0) {
				if (batState.ConsumedEnergy < batSettings.Capacity) {
					battery_remaining = 100 - lroundf(batState.ConsumedEnergy / batSettings.Capacity * 100);
				}
			}

			uint16_t voltage = 0;
			if (batSettings.SensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_BATTERYVOLTAGE] == FLIGHTBATTERYSETTINGS_SENSORTYPE_ENABLED)
				voltage = lroundf(batState.Voltage * 1000);

			uint16_t current = 0;
			if (batSettings.SensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_BATTERYCURRENT] == FLIGHTBATTERYSETTINGS_SENSORTYPE_ENABLED)
				current = lroundf(batState.Current * 100);

			mavlink_msg_sys_status_pack(0, 200, &mavMsg,
					// onboard_control_sensors_present Bitmask showing which onboard controllers and sensors are present. Value of 0: not present. Value of 1: present. Indices: 0: 3D gyro, 1: 3D acc, 2: 3D mag, 3: absolute pressure, 4: differential pressure, 5: GPS, 6: optical flow, 7: computer vision position, 8: laser based position, 9: external ground-truth (Vicon or Leica). Controllers: 10: 3D angular rate control 11: attitude stabilization, 12: yaw position, 13: z/altitude control, 14: x/y position control, 15: motor outputs / control
					0,
					// onboard_control_sensors_enabled Bitmask showing which onboard controllers and sensors are enabled:  Value of 0: not enabled. Value of 1: enabled. Indices: 0: 3D gyro, 1: 3D acc, 2: 3D mag, 3: absolute pressure, 4: differential pressure, 5: GPS, 6: optical flow, 7: computer vision position, 8: laser based position, 9: external ground-truth (Vicon or Leica). Controllers: 10: 3D angular rate control 11: attitude stabilization, 12: yaw position, 13: z/altitude control, 14: x/y position control, 15: motor outputs / control
					0,
					// onboard_control_sensors_health Bitmask showing which onboard controllers and sensors are operational or have an error:  Value of 0: not enabled. Value of 1: enabled. Indices: 0: 3D gyro, 1: 3D acc, 2: 3D mag, 3: absolute pressure, 4: differential pressure, 5: GPS, 6: optical flow, 7: computer vision position, 8: laser based position, 9: external ground-truth (Vicon or Leica). Controllers: 10: 3D angular rate control 11: attitude stabilization, 12: yaw position, 13: z/altitude control, 14: x/y position control, 15: motor outputs / control
					0,
					// load Maximum usage in percent of the mainloop time, (0%: 0, 100%: 1000) should be always below 1000
					(uint16_t)systemStats.CPULoad * 10,
					// voltage_battery Battery voltage, in millivolts (1 = 1 millivolt)
					voltage,
					// current_battery Battery current, in 10*milliamperes (1 = 10 milliampere), -1: autopilot does not measure the current
					current,
					// battery_remaining Remaining battery energy: (0%: 0, 100%: 100), -1: autopilot estimate the remaining battery
					battery_remaining,
					// drop_rate_comm Communication drops in percent, (0%: 0, 100%: 10'000), (UART, I2C, SPI, CAN), dropped packets on all links (packets that were corrupted on reception on the MAV)
					0,
					// errors_comm Communication errors (UART, I2C, SPI, CAN), dropped packets on all links (packets that were corrupted on reception on the MAV)
					0,
					// errors_count1 Autopilot-specific errors
					0,
					// errors_count2 Autopilot-specific errors
					0,
					// errors_count3 Autopilot-specific errors
					0,
					// errors_count4 Autopilot-specific errors
					0);
			msg_length = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_length);
		}

		if (stream_trigger(MAV_DATA_STREAM_RC_CHANNELS)) {
			ManualControlCommandGet(&manualState);
			FlightStatusGet(&flightStatus);
			SystemStatsGet(&systemStats);

			//TODO connect with RSSI object and pass in last argument
			mavlink_msg_rc_channels_raw_pack(0, 200, &mavMsg,
					// time_boot_ms Timestamp (milliseconds since system boot)
					systemStats.FlightTime,
					// port Servo output port (set of 8 outputs = 1 port). Most MAVs will just use one, but this allows to encode more than 8 servos.
					0,
					// chan1_raw RC channel 1 value, in microseconds
					manualState.Channel[0],
					// chan2_raw RC channel 2 value, in microseconds
					manualState.Channel[1],
					// chan3_raw RC channel 3 value, in microseconds
					manualState.Channel[2],
					// chan4_raw RC channel 4 value, in microseconds
					manualState.Channel[3],
					// chan5_raw RC channel 5 value, in microseconds
					manualState.Channel[4],
					// chan6_raw RC channel 6 value, in microseconds
					manualState.Channel[5],
					// chan7_raw RC channel 7 value, in microseconds
					manualState.Channel[6],
					// chan8_raw RC channel 8 value, in microseconds
					manualState.Channel[7],
					// rssi Receive signal strength indicator, 0: 0%, 255: 100%
					manualState.Rssi);
			msg_length = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_length);
		}

		if (stream_trigger(MAV_DATA_STREAM_POSITION)) {
			if (GPSPositionHandle() != NULL )
				GPSPositionGet(&gpsPosData);
			if (HomeLocationHandle() != NULL )
				HomeLocationGet(&homeLocation);
			SystemStatsGet(&systemStats);

			uint8_t gps_fix_type;
			switch (gpsPosData.Status)
			{
			case GPSPOSITION_STATUS_NOGPS:
				gps_fix_type = 0;
				break;
			case GPSPOSITION_STATUS_NOFIX:
				gps_fix_type = 1;
				break;
			case GPSPOSITION_STATUS_FIX2D:
				gps_fix_type = 2;
				break;
			case GPSPOSITION_STATUS_FIX3D:
			case GPSPOSITION_STATUS_DIFF3D:
				gps_fix_type = 3;
				break;
			default:
				gps_fix_type = 0;
				break;
			}

			mavlink_msg_gps_raw_int_pack(0, 200, &mavMsg,
					// time_usec Timestamp (microseconds since UNIX epoch or microseconds since system boot)
					(uint64_t)systemStats.FlightTime * 1000,
					// fix_type 0-1: no fix, 2: 2D fix, 3: 3D fix. Some applications will not use the value of this field unless it is at least two, so always correctly fill in the fix.
					gps_fix_type,
					// lat Latitude in 1E7 degrees
					gpsPosData.Latitude,
					// lon Longitude in 1E7 degrees
					gpsPosData.Longitude,
					// alt Altitude in 1E3 meters (millimeters) above MSL
					gpsPosData.Altitude * 1000,
					// eph GPS HDOP horizontal dilution of position in cm (m*100). If unknown, set to: 65535
					gpsPosData.HDOP * 100,
					// epv GPS VDOP horizontal dilution of position in cm (m*100). If unknown, set to: 65535
					gpsPosData.VDOP * 100,
					// vel GPS ground speed (m/s * 100). If unknown, set to: 65535
					gpsPosData.Groundspeed * 100,
					// cog Course over ground (NOT heading, but direction of movement) in degrees * 100, 0.0..359.99 degrees. If unknown, set to: 65535
					gpsPosData.Heading * 100,
					// satellites_visible Number of satellites visible. If unknown, set to 255
					gpsPosData.Satellites);
			msg_length = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_length);

			mavlink_msg_gps_global_origin_pack(0, 200, &mavMsg,
					// latitude Latitude (WGS84), expressed as * 1E7
					homeLocation.Latitude,
					// longitude Longitude (WGS84), expressed as * 1E7
					homeLocation.Longitude,
					// altitude Altitude(WGS84), expressed as * 1000
					homeLocation.Altitude * 1000);
			msg_length = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_length);

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

		if (stream_trigger(MAV_DATA_STREAM_EXTRA1)) {
			AttitudeActualGet(&attActual);
			SystemStatsGet(&systemStats);

			mavlink_msg_attitude_pack(0, 200, &mavMsg,
					// time_boot_ms Timestamp (milliseconds since system boot)
					systemStats.FlightTime,
					// roll Roll angle (rad)
					attActual.Roll * DEG2RAD,
					// pitch Pitch angle (rad)
					attActual.Pitch * DEG2RAD,
					// yaw Yaw angle (rad)
					attActual.Yaw * DEG2RAD,
					// rollspeed Roll angular speed (rad/s)
					0,
					// pitchspeed Pitch angular speed (rad/s)
					0,
					// yawspeed Yaw angular speed (rad/s)
					0);
			msg_length = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_length);
		}

		if (stream_trigger(MAV_DATA_STREAM_EXTRA2)) {
			if (AirspeedActualHandle() != NULL )
				AirspeedActualGet(&airspeedActual);
			if (GPSPositionHandle() != NULL )
				GPSPositionGet(&gpsPosData);
			if (BaroAltitudeHandle() != NULL )
				BaroAltitudeGet(&baroAltitude);
			ActuatorDesiredGet(&actDesired);
			AttitudeActualGet(&attActual);

			float altitude = 0;
			if (BaroAltitudeHandle() != NULL)
				altitude = baroAltitude.Altitude;
			else if (GPSPositionHandle() != NULL)
				altitude = gpsPosData.Altitude;

			// round attActual.Yaw to nearest int and transfer from (-180 ... 180) to (0 ... 360)
			int16_t heading = lroundf(attActual.Yaw);
			if (heading < 0)
				heading += 360;

			mavlink_msg_vfr_hud_pack(0, 200, &mavMsg,
					// airspeed Current airspeed in m/s
					airspeedActual.TrueAirspeed,
					// groundspeed Current ground speed in m/s
					gpsPosData.Groundspeed,
					// heading Current heading in degrees, in compass units (0..360, 0=north)
					heading,
					// throttle Current throttle setting in integer percent, 0 to 100
					actDesired.Throttle * 100,
					// alt Current altitude (MSL), in meters
					altitude,
					// climb Current climb rate in meters/second
					0);
			msg_length = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_length);

			uint8_t armed_mode = 0;
			if (flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMED)
				armed_mode |= MAV_MODE_FLAG_SAFETY_ARMED;

			mavlink_msg_heartbeat_pack(0, 200, &mavMsg,
					// type Type of the MAV (quadrotor, helicopter, etc., up to 15 types, defined in MAV_TYPE ENUM)
					MAV_TYPE_GENERIC,
					// autopilot Autopilot type / class. defined in MAV_AUTOPILOT ENUM
					MAV_AUTOPILOT_GENERIC,
					// base_mode System mode bitfield, see MAV_MODE_FLAGS ENUM in mavlink/include/mavlink_types.h
					armed_mode,
					// custom_mode A bitfield for use for autopilot-specific flags.
					0,
					// system_status System status flag, see MAV_STATE ENUM
					0);
			msg_length = mavlink_msg_to_send_buffer(serial_buf, &mavMsg);
			PIOS_COM_SendBuffer(mavlink_port, serial_buf, msg_length);
		}
	}
}

static bool stream_trigger(enum MAV_DATA_STREAM stream_num) {
	uint8_t rate = (uint8_t) mav_rates[stream_num];

	if (rate == 0) {
		return false;
	}

	if (stream_ticks[stream_num] == 0) {
		// we're triggering now, setup the next trigger point
		if (rate > TASK_RATE_HZ) {
			rate = TASK_RATE_HZ;
		}
		stream_ticks[stream_num] = (TASK_RATE_HZ / rate);
		return true;
	}

	// count down at 50Hz
	stream_ticks[stream_num]--;
	return false;
}

static void updateSettings()
{
	
	if (mavlink_port) {
		// Retrieve settings
		uint8_t speed;
		ModuleSettingsMavlinkSpeedGet(&speed);

		// Set port speed
		switch (speed) {
		case MODULESETTINGS_MAVLINKSPEED_2400:
			PIOS_COM_ChangeBaud(mavlink_port, 2400);
			break;
		case MODULESETTINGS_MAVLINKSPEED_4800:
			PIOS_COM_ChangeBaud(mavlink_port, 4800);
			break;
		case MODULESETTINGS_MAVLINKSPEED_9600:
			PIOS_COM_ChangeBaud(mavlink_port, 9600);
			break;
		case MODULESETTINGS_MAVLINKSPEED_19200:
			PIOS_COM_ChangeBaud(mavlink_port, 19200);
			break;
		case MODULESETTINGS_MAVLINKSPEED_38400:
			PIOS_COM_ChangeBaud(mavlink_port, 38400);
			break;
		case MODULESETTINGS_MAVLINKSPEED_57600:
			PIOS_COM_ChangeBaud(mavlink_port, 57600);
			break;
		case MODULESETTINGS_MAVLINKSPEED_115200:
			PIOS_COM_ChangeBaud(mavlink_port, 115200);
			break;
		}
	}
}
/**
 * @}
 * @}
 */
