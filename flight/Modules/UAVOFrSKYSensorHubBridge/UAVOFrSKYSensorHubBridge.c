/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup uavoFrSKYSensorHubBridge UAVO to FrSKY Bridge Module
 * @{ 
 *
 * @file       uavoFrSKYSensorHubBridge.c
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
#include "pios.h"
#include "openpilot.h"
#include "physical_constants.h"
#include "modulesettings.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "airspeedactual.h"
#include "baroaltitude.h"
#include "accels.h"

#if defined(PIOS_INCLUDE_FRSKY_SENSOR_HUB)
// ****************
// Private functions

static void uavoFrSKYSensorHubBridgeTask(void *parameters) __attribute__((optimize(0)));

static uint16_t frsky_pack_altitude(
		float altitude,
		uint8_t *serial_buf);

static uint16_t frsky_pack_temperature(
		float temperature_01,
		float temperature_02,
		uint8_t *serial_buf);

static uint16_t frsky_pack_accel(
		float accels_x,
		float accels_y,
		float accels_z,
		uint8_t *serial_buf);

static uint16_t frsky_pack_battery(
		float voltage_01,
		float voltage_02,
		float current,
		uint8_t *serial_buf);

static uint16_t frsky_pack_rpm(
		uint16_t rpm,
		uint8_t *serial_buf);

static uint16_t frsky_pack_gps(
		float course,
		int32_t latitude,
		int32_t longitude,
		float altitude,
		float speed,
		uint8_t *serial_buf);

static uint16_t frsky_pack_fuel(
		float fuel_level,
		uint8_t *serial_buf);

static uint16_t frsky_pack_stop(
		uint8_t *serial_buf);

static int16_t frsky_acceleration_unit(float accel);
static void frsky_serialize_value(uint8_t valueid, uint8_t *value, uint8_t *serial_buf, uint8_t *index);
static void frsky_write_userdata_byte(uint8_t byte, uint8_t *serial_buf, uint8_t *index);

// ****************
// Private constants

#if defined(PIOS_FRSKY_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_FRSKY_SENSOR_HUB_STACK_SIZE
#else
#define STACK_SIZE_BYTES 800
#endif

#define TASK_PRIORITY               (tskIDLE_PRIORITY + 1)
#define TASK_RATE_HZ				10

#define FRSKY_MAX_PACKET_LEN 106

#define FRSKY_FRAME_START 0x5E
#define FRSKY_FRAME_DATA_HEADER 0x5E
#define FRSKY_FRAME_STOP 0x5E

enum FRSKY_FRAME
{
    FRSKY_FRAME_01,
    FRSKY_FRAME_02,
    FRSKY_FRAME_03,
    FRSKY_FRAME_COUNT
};

enum FRSKY_VALUE_ID
{
	FRSKY_GPS_ALTITUDE_INTEGER = 0x01,
	FRSKY_GPS_ALTITUDE_DECIMAL = FRSKY_GPS_ALTITUDE_INTEGER + 8,
	FRSKY_TEMPERATURE_1 = 0x02,
	FRSKY_RPM = 0x03,
	FRSKY_FUEL_LEVEL = 0x04,
	FRSKY_TEMPERATURE_2 = 0x05,
	FRSKY_VOLTAGE = 0x06,
	FRSKY_ALTITUDE_INTEGER = 0x10,
	FRSKY_ALTITUDE_DECIMAL = 0x21,
	FRSKY_GPS_SPEED_INTEGER = 0x11,
	FRSKY_GPS_SPEED_DECIMAL = 0x11 + 8,
	FRSKY_GPS_LONGITUDE_INTEGER = 0x12,
	FRSKY_GPS_LONGITUDE_DECIMAL = FRSKY_GPS_LONGITUDE_INTEGER + 8,
	FRSKY_GPS_E_W = 0x1A + 8,
	FRSKY_GPS_LATITUDE_INTEGER = 0x13,
	FRSKY_GPS_LATITUDE_DECIMAL = FRSKY_GPS_LATITUDE_INTEGER + 8,
	FRSKY_GPS_N_S = 0x1B + 8,
	FRSKY_GPS_COURSE_INTEGER = 0x14,
	FRSKY_GPS_COURSE_DECIMAL = FRSKY_GPS_COURSE_INTEGER + 8,
	FRSKY_DATE_MONTH = 0x15,
	FRSKY_DATE_YEAR = 0x16,
	FRSKY_HOUR_MINUTE = 0x17,
	FRSKY_SECOND = 0x18,
	FRSKY_ACCELERATION_X = 0x24,
	FRSKY_ACCELERATION_Y = 0x25,
	FRSKY_ACCELERATION_Z = 0x26,
	FRSKY_VOLTAGE_AMPERE_SENSOR_INTEGER = 0x3A,
	FRSKY_VOLTAGE_AMPERE_SENSOR_DECIMAL = 0x3B,
	FRSKY_CURRENT = 0x28,
};



static const uint8_t frsky_rates[] =
	 { [FRSKY_FRAME_01]=0x05, //5Hz
	   [FRSKY_FRAME_02]=0x01, //1Hz
	   [FRSKY_FRAME_03]=0x01 }; //2Hz

#define MAXSTREAMS sizeof(frsky_rates)

static bool frame_trigger(enum FRSKY_FRAME frame_num);

// ****************
// Private variables

static xTaskHandle uavoFrSKYSensorHubBridgeTaskHandle;

static uint32_t frsky_port;

static bool module_enabled = false;

static uint8_t * frame_ticks;

static uint8_t * serial_buf;

/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t uavoFrSKYSensorHubBridgeStart(void) {
	if (module_enabled) {
		// Start tasks
		xTaskCreate(uavoFrSKYSensorHubBridgeTask, (signed char *) "uavoFrSKYSensorHubBridge",
				STACK_SIZE_BYTES / 4, NULL, TASK_PRIORITY,
				&uavoFrSKYSensorHubBridgeTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_UAVOFRSKYSBRIDGE,
				uavoFrSKYSensorHubBridgeTaskHandle);
		return 0;
	}
	return -1;
}
/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t uavoFrSKYSensorHubBridgeInitialize(void) {
	frsky_port = PIOS_COM_FRSKY_SENSOR_HUB;

	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);

	if (frsky_port
			&& (module_state[MODULESETTINGS_ADMINSTATE_UAVOFRSKYSENSORHUBBRIDGE]
					== MODULESETTINGS_ADMINSTATE_ENABLED)) {
		module_enabled = true;
		PIOS_COM_ChangeBaud(frsky_port, 9600);

		serial_buf = pvPortMalloc(FRSKY_MAX_PACKET_LEN);
		frame_ticks = pvPortMalloc(MAXSTREAMS);
		for (int x = 0; x < MAXSTREAMS; ++x) {
			frame_ticks[x] = (TASK_RATE_HZ / frsky_rates[x]);
		}
	} else {
		module_enabled = false;
	}
	return 0;
}
MODULE_INITCALL( uavoFrSKYSensorHubBridgeInitialize, uavoFrSKYSensorHubBridgeStart)

/**
 * Main task. It does not return.
 */

static void uavoFrSKYSensorHubBridgeTask(void *parameters) {
	FlightBatterySettingsData batSettings;
	FlightBatteryStateData batState;
	GPSPositionData gpsPosData;
	BaroAltitudeData baroAltitude;
	AccelsData accels;

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

	if (AccelsHandle() != NULL )
		AccelsGet(&accels);
	else {
		accels.x = 0.0;
		accels.y = 0.0;
		accels.z = 0.0;
		accels.temperature = 0.0;
	}

	uint16_t msg_length;
	portTickType lastSysTime;
	// Main task loop
	lastSysTime = xTaskGetTickCount();

	while (1) {
		vTaskDelayUntil(&lastSysTime, MS2TICKS(1000 / TASK_RATE_HZ));

		if (frame_trigger(FRSKY_FRAME_01)) {
			if (FlightBatteryStateHandle() != NULL )
				FlightBatteryStateGet(&batState);

			if (AccelsHandle() != NULL )
				AccelsGet(&accels);

			if (BaroAltitudeHandle() != NULL )
				BaroAltitudeGet(&baroAltitude);

			float voltage = 0.0;
			if (batSettings.SensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_BATTERYVOLTAGE] == FLIGHTBATTERYSETTINGS_SENSORTYPE_ENABLED)
				voltage = batState.Voltage * 1000;

			float current = 0.0;
			if (batSettings.SensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_BATTERYCURRENT] == FLIGHTBATTERYSETTINGS_SENSORTYPE_ENABLED)
				current = batState.Current * 100;

			msg_length += frsky_pack_accel(
					accels.x,
					accels.y,
					accels.z,
					serial_buf + msg_length);

			msg_length += frsky_pack_altitude(
					baroAltitude.Altitude,
					serial_buf + msg_length);

			msg_length += frsky_pack_temperature(
					baroAltitude.Temperature,
					accels.temperature,
					serial_buf + msg_length);

			msg_length += frsky_pack_battery(
					voltage,
					0,
					current,
					serial_buf + msg_length);

			msg_length += frsky_pack_rpm(
					0,
					serial_buf + msg_length);

			msg_length += frsky_pack_stop(serial_buf + msg_length);

			PIOS_COM_SendBuffer(frsky_port, serial_buf, msg_length);
		}

		if (frame_trigger(FRSKY_FRAME_02)) {
			if (GPSPositionHandle() != NULL ){
				GPSPositionGet(&gpsPosData);
			}

			if (gpsPosData.Status == GPSPOSITION_STATUS_FIX2D || gpsPosData.Status == GPSPOSITION_STATUS_FIX3D)
			{
				msg_length += frsky_pack_gps(
					gpsPosData.Heading,
					gpsPosData.Latitude,
					gpsPosData.Longitude,
					gpsPosData.Altitude,
					gpsPosData.Groundspeed,
					serial_buf + msg_length);
			}

			msg_length += frsky_pack_fuel(
					0.0,
					serial_buf + msg_length);

			msg_length += frsky_pack_stop(serial_buf + msg_length);

			PIOS_COM_SendBuffer(frsky_port, serial_buf, msg_length);
		}

		if (frame_trigger(FRSKY_FRAME_03)) {
		}
	}
}

static bool frame_trigger(enum FRSKY_FRAME frame_num) {
	uint8_t rate = (uint8_t) frsky_rates[frame_num];

	if (rate == 0) {
		return false;
	}

	if (frame_ticks[frame_num] == 0) {
		// we're triggering now, setup the next trigger point
		if (rate > TASK_RATE_HZ) {
			rate = TASK_RATE_HZ;
		}
		frame_ticks[frame_num] = (TASK_RATE_HZ / rate);
		return true;
	}

	// count down at 50Hz
	frame_ticks[frame_num]--;
	return false;
}

static uint16_t frsky_pack_altitude(
		float altitude,
		uint8_t *serial_buf){

	uint8_t index = 0;

	float altitudeInteger = 0.0;
	altitude = altitude * 100;
	uint16_t decimalValue = lroundf(modff(altitude, &altitudeInteger)*1000);
	int16_t integerValue = lroundf(altitudeInteger);

	frsky_serialize_value(FRSKY_ALTITUDE_INTEGER, (uint8_t*)&integerValue, serial_buf, &index);
	frsky_serialize_value(FRSKY_ALTITUDE_DECIMAL, (uint8_t*)&decimalValue, serial_buf, &index);

	return index;
}

static uint16_t frsky_pack_temperature(
		float temperature_01,
		float temperature_02,
		uint8_t *serial_buf){

	uint8_t index = 0;

	int16_t temperature = lroundf(temperature_01);
	frsky_serialize_value(FRSKY_TEMPERATURE_1, (uint8_t*)&temperature, serial_buf, &index);

	temperature = lroundf(temperature_02);
	frsky_serialize_value(FRSKY_TEMPERATURE_2, (uint8_t*)&temperature, serial_buf, &index);


	return index;
}

static uint16_t frsky_pack_accel(
		float accels_x,
		float accels_y,
		float accels_z,
		uint8_t *serial_buf){

	uint8_t index = 0;

	int16_t accel = frsky_acceleration_unit(accels_x);
	frsky_serialize_value(FRSKY_ACCELERATION_X, (uint8_t*)&accel, serial_buf, &index);

	accel = frsky_acceleration_unit(accels_y);
	frsky_serialize_value(FRSKY_ACCELERATION_Y, (uint8_t*)&accel, serial_buf, &index);

	accel = frsky_acceleration_unit(accels_z);
	frsky_serialize_value(FRSKY_ACCELERATION_Z, (uint8_t*)&accel, serial_buf, &index);

	return index;
}

static uint16_t frsky_pack_battery(
		float voltage_01,
		float voltage_02,
		float current,
		uint8_t *serial_buf){

	uint8_t index = 0;

	uint16_t uvalue = lroundf(voltage_01 * 100);
	frsky_serialize_value(FRSKY_VOLTAGE, (uint8_t*)&uvalue, serial_buf, &index);

	uvalue = lroundf(current * 10);
	frsky_serialize_value(FRSKY_VOLTAGE, (uint8_t*)&uvalue, serial_buf, &index);

	float voltageInteger = 0.0;
	uint16_t decimalValue = lroundf(modff(voltage_02, &voltageInteger)*10);
	uint16_t integerValue = lroundf(voltageInteger);

	frsky_serialize_value(FRSKY_VOLTAGE_AMPERE_SENSOR_INTEGER, (uint8_t*)&integerValue, serial_buf, &index);
	frsky_serialize_value(FRSKY_VOLTAGE_AMPERE_SENSOR_DECIMAL, (uint8_t*)&decimalValue, serial_buf, &index);

	return index;
}

static uint16_t frsky_pack_rpm(
		uint16_t rpm,
		uint8_t *serial_buf)
{
	uint8_t index = 0;

	frsky_serialize_value(FRSKY_RPM, (uint8_t*)&rpm, serial_buf, &index);

	return index;
}

static uint16_t frsky_pack_fuel(
		float fuel_level,
		uint8_t *serial_buf)
{
	uint8_t index = 0;
	uint16_t level = abs(fuel_level) * 100;
	frsky_serialize_value(FRSKY_FUEL_LEVEL, (uint8_t*)&level, serial_buf, &index);

	return index;
}

static uint16_t frsky_pack_stop(
		uint8_t *serial_buf)
{
	serial_buf[0] = FRSKY_FRAME_STOP;

	return 1;
}

static uint16_t frsky_pack_gps(
		float course,
		int32_t latitude,
		int32_t longitude,
		float altitude,
		float speed,
		uint8_t *serial_buf)
{
	uint8_t index = 0;

	{
		float courseInteger = 0.0;
		uint16_t decimalValue = lroundf(modff(course, &courseInteger) * 100);
		uint16_t integerValue = lroundf(courseInteger);

		frsky_serialize_value(FRSKY_GPS_COURSE_INTEGER, (uint8_t*)&integerValue, serial_buf, &index);
		frsky_serialize_value(FRSKY_GPS_COURSE_DECIMAL, (uint8_t*)&decimalValue, serial_buf, &index);
	}

	// latitude
	{
		uint16_t integerValue = (abs(latitude) / 100000);
		uint16_t decimalValue = (abs(latitude) / 10) % 10000;

		frsky_serialize_value(FRSKY_GPS_LATITUDE_INTEGER, (uint8_t*)&integerValue, serial_buf, &index);
		frsky_serialize_value(FRSKY_GPS_LATITUDE_DECIMAL, (uint8_t*)&decimalValue, serial_buf, &index);

		uint16_t hemisphere = 'N';
		if (latitude < 0) {
			hemisphere = 'S';
		}
		frsky_serialize_value(FRSKY_GPS_N_S, (uint8_t*)&hemisphere, serial_buf, &index);
	}

	// longitude
	{
		uint16_t integerValue = (abs(longitude) / 100000);
		uint16_t decimalValue = (abs(longitude) / 10) % 10000;

		uint16_t hemisphere = 'E';
		if (longitude < 0) {
			hemisphere = 'W';
		}

		frsky_serialize_value(FRSKY_GPS_LONGITUDE_INTEGER, (uint8_t*)&integerValue, serial_buf, &index);
		frsky_serialize_value(FRSKY_GPS_LONGITUDE_DECIMAL, (uint8_t*)&decimalValue, serial_buf, &index);

		frsky_serialize_value(FRSKY_GPS_E_W, (uint8_t*)&hemisphere, serial_buf, &index);
	}

	// speed
	{
		float knots = speed / (float)0.514444;

		float knotsInteger = 0.0;
		uint16_t decimalValue = lroundf(modff(knots, &knotsInteger) * 100);
		int16_t integerValue = lroundf(knotsInteger);

		frsky_serialize_value(FRSKY_GPS_SPEED_INTEGER, (uint8_t*)&integerValue, serial_buf, &index);
		frsky_serialize_value(FRSKY_GPS_SPEED_DECIMAL, (uint8_t*)&decimalValue, serial_buf, &index);
	}

	// altitude
	{
		float altitudeInteger = 0.0;
		uint16_t decimalValue = lroundf(modff(altitude, &altitudeInteger) * 100);
		int16_t integerValue = lroundf(altitudeInteger);

		frsky_serialize_value(FRSKY_GPS_ALTITUDE_INTEGER, (uint8_t*)&integerValue, serial_buf, &index);
		frsky_serialize_value(FRSKY_GPS_ALTITUDE_DECIMAL, (uint8_t*)&decimalValue, serial_buf, &index);
	}

	return index;
}

static int16_t frsky_acceleration_unit(float accel)
{
	accel = accel / (float)9.81274; //convert to gravity
	accel *= 1000;
	return lroundf(accel);
}

static void frsky_serialize_value(uint8_t valueid, uint8_t *value, uint8_t *serial_buf, uint8_t *index)
{
	serial_buf[(*index)++] = FRSKY_FRAME_DATA_HEADER;
	serial_buf[(*index)++] = valueid;

	frsky_write_userdata_byte(value[0], serial_buf, index);
	frsky_write_userdata_byte(value[1], serial_buf, index);
}

static void frsky_write_userdata_byte(uint8_t byte, uint8_t *serial_buf, uint8_t *index)
{
	//** byte stuffing
	if ((byte == 0x5E) || (byte == 0x5D))
	{
		serial_buf[(*index)++] = 0x5D;
		serial_buf[(*index)++] = ~(byte ^ 0x60);
	}
	else
	{
		serial_buf[(*index)++] = byte;
	}
}

#endif
/**
 * @}
 * @}
 */
