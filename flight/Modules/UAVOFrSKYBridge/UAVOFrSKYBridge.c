/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup UAVOFrSKYBridge UAVO to FrSKY Bridge Module
 * @{ 
 *
 * @file       UAVOFrSKYBridge.c
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

#if defined(PIOS_INCLUDE_HOTT)
// ****************
// Private functions
// Frame 1 structure
struct frsky_frame_1{
	uint8_t start;				// start byte
	uint8_t accels_x_id;		//
	int16_t accels_x;
	uint8_t accels_y_header;
	uint8_t accels_y_id;		//
	int16_t accels_y;
	uint8_t accels_z_header;
	uint8_t accels_z_id;		//
	int16_t accels_z;
	uint8_t altitude_integer_header;
	uint8_t altitude_integer_id;		//
	int16_t altitude_integer;
	uint8_t altitude_decimal_header;
	uint8_t altitude_decimal_id;		//
	uint16_t altitude_decimal;
	uint8_t temperature_1_header;
	uint8_t temperature_1_id;		//
	int16_t temperature_1;
	uint8_t temperature_2_header;
	uint8_t temperature_2_id;		//
	int16_t temperature_2;
	uint8_t voltage_header;
	uint8_t voltage_id;		//
	uint16_t voltage;
	uint8_t current_header;
	uint8_t current_id;
	uint16_t current;
	uint8_t voltage_amperesensor_integer_header;
	uint8_t voltage_amperesensor_integer_id;		//
	uint16_t voltage_amperesensorinteger;
	uint8_t voltage_amperesensor_decimal_header;
	uint8_t voltage_amperesensor_decimal_id;		//
	uint16_t voltage_amperesensor_decimal;
	uint8_t rpm_header;
	uint8_t rpm_id;		//
	uint16_t rpm;
	uint8_t stop;				// stop byte
};

static void uavoFrSKYBridgeTask(void *parameters);
static uint16_t frsky_pack_frame_01(
		float accels_x,
		float accels_y,
		float accels_z,
		float altitude,
		float temperature_01,
		float temperature_02,
		float voltage,
		float current,
		uint16_t RPM,
		struct frsky_frame_1 *frame);

// ****************
// Private constants

#if defined(PIOS_FRSKY_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_FRSKY_STACK_SIZE
#else
#define STACK_SIZE_BYTES 800
#endif

#define TASK_PRIORITY               (tskIDLE_PRIORITY + 1)
#define TASK_RATE_HZ				10

#define FRSKY_MAX_PACKET_LEN 53

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

static xTaskHandle uavoFrSKYBridgeTaskHandle;

static uint32_t frsky_port;

static bool module_enabled = false;

static uint8_t * frame_ticks;

static uint8_t * serial_buf;

/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t uavoFrSKYBridgeStart(void) {
	if (module_enabled) {
		// Start tasks
		xTaskCreate(uavoFrSKYBridgeTask, (signed char *) "uavoFrSkyBridge",
				STACK_SIZE_BYTES / 4, NULL, TASK_PRIORITY,
				&uavoFrSKYBridgeTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_UAVOFRSKYSBRIDGE,
				uavoFrSKYBridgeTaskHandle);
		return 0;
	}
	return -1;
}
/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t uavoFrSKYBridgeInitialize(void) {
	frsky_port = PIOS_COM_FRSKY;

	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);

	if (frsky_port
			&& (module_state[MODULESETTINGS_ADMINSTATE_UAVOFRSKYBRIDGE]
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
MODULE_INITCALL( uavoFrSKYBridgeInitialize, uavoFrSKYBridgeStart)

/**
 * Main task. It does not return.
 */

static void uavoFrSKYBridgeTask(void *parameters) {
	FlightBatterySettingsData batSettings;
	FlightBatteryStateData batState;
	//GPSPositionData gpsPosData;
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

//	if (GPSPositionHandle() == NULL ){
//		gpsPosData.Altitude = 0;
//		gpsPosData.GeoidSeparation = 0;
//		gpsPosData.Groundspeed = 0;
//		gpsPosData.HDOP = 0;
//		gpsPosData.Heading = 0;
//		gpsPosData.Latitude = 0;
//		gpsPosData.Longitude = 0;
//		gpsPosData.PDOP = 0;
//		gpsPosData.Satellites = 0;
//		gpsPosData.Status = 0;
//		gpsPosData.VDOP = 0;
//	}

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

			msg_length = frsky_pack_frame_01(
					accels.x,
					accels.y,
					accels.z,
					baroAltitude.Altitude,
					baroAltitude.Temperature,
					0.0,
					voltage,
					current,
					0,
					(struct frsky_frame_1 *)serial_buf);

			PIOS_COM_SendBuffer(frsky_port, serial_buf, msg_length);
		}

		if (frame_trigger(FRSKY_FRAME_02)) {
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

static uint16_t frsky_pack_frame_01(
		float accels_x,
		float accels_y,
		float accels_z,
		float altitude,
		float temperature_01,
		float temperature_02,
		float voltage,
		float current,
		uint16_t rpm,
		struct frsky_frame_1 *frame)
{
	frame->start = FRSKY_FRAME_START;
	frame->accels_x_id = FRSKY_ACCELERATION_X;
	frame->accels_x = lroundf(accels_x * 1000);

	frame->accels_y_header = FRSKY_FRAME_DATA_HEADER;
	frame->accels_y_id = FRSKY_ACCELERATION_X;
	frame->accels_y = lroundf(accels_y * 1000);

	frame->accels_z_header = FRSKY_FRAME_DATA_HEADER;
	frame->accels_z_id = FRSKY_ACCELERATION_X;
	frame->accels_z = lroundf(accels_z * 1000);

	float altitudeInteger = 0.0;
	altitude = altitude * 100;
	frame->altitude_decimal_header = FRSKY_FRAME_DATA_HEADER;
	frame->altitude_decimal_id = FRSKY_ALTITUDE_DECIMAL;
	frame->altitude_decimal = lroundf(modff(altitude, &altitudeInteger));
	frame->altitude_integer_header = FRSKY_FRAME_DATA_HEADER;
	frame->altitude_integer_id = FRSKY_ALTITUDE_INTEGER;
	frame->altitude_integer = lroundf(altitudeInteger);

	frame->temperature_1_header = FRSKY_FRAME_DATA_HEADER;
	frame->temperature_1_id = FRSKY_TEMPERATURE_1;
	frame->temperature_1 = lroundf(temperature_01);
	frame->temperature_2_header = FRSKY_FRAME_DATA_HEADER;
	frame->temperature_2_id = FRSKY_TEMPERATURE_2;
	frame->temperature_2 = lroundf(temperature_02);

	frame->voltage_header = FRSKY_FRAME_DATA_HEADER;
	frame->voltage_id = FRSKY_VOLTAGE;
	frame->voltage = lroundf(current * 100);

	frame->current_header = FRSKY_FRAME_DATA_HEADER;
	frame->current_id = FRSKY_CURRENT;
	frame->current = lroundf(current * 10);

	frame->voltage_amperesensor_integer_header = FRSKY_FRAME_DATA_HEADER;
	frame->voltage_amperesensor_integer_id = FRSKY_VOLTAGE_AMPERE_SENSOR_INTEGER;
	frame->voltage_amperesensorinteger = 0;
	frame->voltage_amperesensor_decimal_header = FRSKY_FRAME_DATA_HEADER;
	frame->voltage_amperesensor_decimal_id = FRSKY_VOLTAGE_AMPERE_SENSOR_DECIMAL;
	frame->voltage_amperesensor_decimal = 0;
	frame->rpm_header = FRSKY_FRAME_DATA_HEADER;
	frame->rpm_id = FRSKY_RPM;
	frame->rpm = rpm;
	frame->stop = FRSKY_FRAME_STOP;
	return sizeof(*frame);
}

#endif
/**
 * @}
 * @}
 */
