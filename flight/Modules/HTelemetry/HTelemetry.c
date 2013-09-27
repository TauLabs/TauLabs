/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup HTelemetry HoTT Telemetry Module
 * @{ 
 *
 * @file       HTelemetry.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      sends telemery data on HoTT request
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
#include "modulesettings.h"
#include "htelemetrysettings.h"
#include "altholdsmoothed.h"
#include "baroaltitude.h"
#include "flightbatterystate.h"
#include "gyros.h"
#include "gpsposition.h"

// Private constants
#define STACK_SIZE_BYTES 800
#define TASK_PRIORITY				(tskIDLE_PRIORITY + 1)
#define HTELE_MAX_MESSAGE_LENGTH 200
#define HTELE_TEXT_ID 0x7f			// Text request
#define HTELE_BINARY_ID 0x80		// Binary request
#define HTELE_VARIO_ID 0x89			// Vario Module ID
#define HTELE_VARIO_TEXT_ID 0x90	// Vario Module TEXT ID
#define HTELE_GPS_ID 0x8a			// GPS Module ID
#define HTELE_GPS_TEXT_ID 0xa0		// GPS Module TEXT ID
#define HTELE_ESC_ID 0x8c			// ESC Module ID
#define HTELE_ESC_TEXT_ID 0xc0		// ESC Module TEXT ID
#define HTELE_GAM_ID 0x8d			// General Air Module ID
#define HTELE_GAM_TEXT_ID 0xd0		// General Air Module TEXT ID
#define HTELE_EAM_ID 0x8e			// Electric Air Module ID
#define HTELE_EAM_TEXT_ID 0xe0		// Electric Air Module TEXT ID
#define HTELE_TEXT_START 0x7b		// Start byte Text mode
#define HTELE_START 0x7c			// Start byte Binary mode
#define HTELE_STOP 0x7d				// End byte
#define HTELE_BUTTON_DEC 0xEB		// minus button
#define HTELE_BUTTON_INC 0xED		// plus button
#define HTELE_BUTTON_SET 0xE9
#define HTELE_BUTTON_NIL 0x0F		// esc button
#define HTELE_BUTTON_NEXT 0xEE
#define HTELE_BUTTON_PREV 0xE7

// Private types

// Private variables
static xTaskHandle HTelemetryTaskHandle;
// static HTelemetrySettingsData HTelemetrySettings;
static uint32_t htelemetry_port;
static bool module_enabled = false;
static uint8_t * tx_buffer;

static AltHoldSmoothedData altState;
static BaroAltitudeData baroState;
static FlightBatteryStateData battState;
static GPSPositionData gpsState;
static GyrosData gyroState;
static float baroAltMin;
static float baroAltMax;

// Private functions
static void HTelemetryTask(void *parameters);
static uint16_t build_VARIO_message(uint8_t *buffer);
static uint16_t build_GPS_message(uint8_t *buffer);
static uint16_t build_GAM_message(uint8_t *buffer);
static uint16_t build_EAM_message(uint8_t *buffer);
static uint16_t build_ESC_message(uint8_t *buffer);
static uint16_t build_TEXT_message(uint8_t *buffer);
static uint8_t calc_checksum(uint8_t *data, uint16_t size);
static void convert_float2byte(float val, float scale, uint16_t offset, uint8_t *lsb);
static void convert_float2word(float val, float scale, uint16_t offset, uint8_t *lsb, uint8_t *msb);
static void convert_long2gps(int32_t val, uint8_t *dir, uint8_t *min_lsb, uint8_t *min_msb, uint8_t *sec_lsb, uint8_t *sec_msb);

/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t HTelemetryStart(void)
{
	if (module_enabled) {
		HTelemetryTaskHandle = NULL;

		// Start tasks
		xTaskCreate(HTelemetryTask, (signed char *) "HTelemetry",
				STACK_SIZE_BYTES / 4, NULL, TASK_PRIORITY,
				&HTelemetryTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_HTELEMETRY,
				HTelemetryTaskHandle);
		return 0;
	}
	return -1;
}

/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t HTelemetryInitialize(void)
{
	htelemetry_port = PIOS_COM_HTELEMETRY;

	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);

	if (htelemetry_port && (module_state[MODULESETTINGS_ADMINSTATE_HTELEMETRY] == MODULESETTINGS_ADMINSTATE_ENABLED)) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}

	if (module_enabled) {
		HTelemetrySettingsInitialize();
		return 0;
	}

	return -1;
}
MODULE_INITCALL( HTelemetryInitialize, HTelemetryStart)

/**
 * Main task. It does not return.
 */
static void HTelemetryTask(void *parameters) {
	uint8_t rx_buffer[3];
	uint16_t message_size = 0;

	static enum htelemetry_state {
		IDLE,
		BINARYMODE,
		TEXTMODE,
		TRANSMIT,
		CLEANUP
	} state = IDLE;

	// HoTT telemetry baudrate is fixed to 19200
	PIOS_COM_ChangeBaud(htelemetry_port, 19200);

	// allocate memory for message buffer
	tx_buffer = pvPortMalloc(HTELE_MAX_MESSAGE_LENGTH);

	// 500ms delay for sensor setup
	vTaskDelay(500);

	// clear some sensor data
	baroState.Altitude = 0;
	baroState.Temperature = 0;
	altState.Velocity = 0;
	gpsState.Latitude = 0;
	gpsState.Longitude = 0;
	gpsState.Altitude = 0;
	gpsState.Heading = 0;
	gpsState.Groundspeed = 0;
	gpsState.Status = GPSPOSITION_STATUS_NOGPS;
	gpsState.Satellites = 0;
	battState.Voltage = 0;
	battState.Current = 0;
	battState.ConsumedEnergy = 0;
	gyroState.temperature = 0;

	// initialize altitude min/max values
	if (BaroAltitudeHandle() != NULL)
		BaroAltitudeGet(&baroState);
	baroAltMin = baroState.Altitude;
	baroAltMax = baroState.Altitude;

	// initialize timer variables
	portTickType lastSysTime = xTaskGetTickCount();
	portTickType idleDelay = MS2TICKS(5);
	portTickType dataDelay = MS2TICKS(2);

	// telemetry state machine. endless loop
	while (1) {
		switch (state) {
			case IDLE:
				// wait for the next byte of telemetry request in 2ms interval
				while (PIOS_COM_ReceiveBuffer(htelemetry_port, rx_buffer, 1, 0) == 0)
					vTaskDelayUntil(&lastSysTime, dataDelay);
				// set start trigger point
				lastSysTime = xTaskGetTickCount();
				// shift receiver buffer for better sync
				rx_buffer[2]= rx_buffer[1];
				rx_buffer[1]= rx_buffer[0];
				// check received byte (TELEMETRY MODE)
					switch (rx_buffer[2]) {
						case HTELE_BINARY_ID:
							state = BINARYMODE;
							break;
						case HTELE_TEXT_ID:
							state = TEXTMODE;
							state = IDLE;
							break;
						default:
							state = IDLE;
					}
				break;
			case BINARYMODE:
				// check received byte (SENSOR ID)
				switch (rx_buffer[1]) {
					case HTELE_VARIO_ID:
						message_size = build_VARIO_message(tx_buffer);
						break;
					case HTELE_GPS_ID:
						message_size = build_GPS_message(tx_buffer);
						break;
					case HTELE_GAM_ID:
						message_size = build_GAM_message(tx_buffer);
						break;
					case HTELE_EAM_ID:
						message_size = build_EAM_message(tx_buffer);
						break;
					case HTELE_ESC_ID:
						message_size = build_ESC_message(tx_buffer);
						break;
					default:
						message_size = 0;
				}
				// setup next state according message size
				state = (message_size > 0) ? TRANSMIT : IDLE;
				break;
			case TEXTMODE:
				// check received byte (upper half == SENSOR ID, lower half == KEY CODE)
				// TODO: fill textmessages with data.
				message_size = build_TEXT_message(tx_buffer);
				// setup next state according message size
				state = (message_size > 0) ? TRANSMIT : IDLE;
				break;
			case TRANSMIT:
				// pause, then check serial buffer
				vTaskDelayUntil(&lastSysTime, idleDelay);
				if (PIOS_COM_ReceiveBuffer(htelemetry_port, rx_buffer, 1, 0) == 0) {
					// nothing received means idle line. ready to transmit the requested message
					for (int i = 0; i < message_size; i++) {
						// send message content with pause between each byte
						PIOS_COM_SendCharNonBlocking(htelemetry_port, tx_buffer[i]);
						vTaskDelayUntil(&lastSysTime, dataDelay);
					}
					state = CLEANUP;
				} else {
					// line is not idle.
					state = IDLE;
				}
				break;
			case CLEANUP:
				// Clear serial buffer after transmit. This is required for a possible loopback data.
				vTaskDelayUntil(&lastSysTime, idleDelay);
				PIOS_COM_ReceiveBuffer(htelemetry_port, tx_buffer, message_size, 0);
				state = IDLE;
			default:
				state = IDLE;
		}
	}
}

/**
 * Build requested answer messages.
 * \return value sets message size
 */
uint16_t build_VARIO_message(uint8_t *buffer) {
	// Vario Module message structure
	struct hott_vario_message {
		uint8_t start;				// start byte
		uint8_t sensor_id;			// VARIO sensor ID
		uint8_t warning;			// 0…= warning beeps
		uint8_t sensor_text_id;		// VARIO sensor text ID
		uint8_t alarm_inverse;		// inverse status
		uint8_t act_altitude_L;		// actual altitude LSB/MSB (meters), offset 500, 500 == 0m
		uint8_t act_altitude_H;
		uint8_t max_altitude_L;		// max. altitude LSB/MSB (meters), 500 == 0m
		uint8_t max_altitude_H;
		uint8_t min_altitude_L;		// min. altitude LSB/MSB (meters), 500 == 0m
		uint8_t min_altitude_H;
		uint8_t climbrate_L;		// climb rate LSB/MSB (0.01m/s), offset 30000, 30000 == 0.00m/s
		uint8_t climbrate_H;
		uint8_t climbrate3s_L;		// climb rate LSB/MSB (0.01m/3s), 30000 == 0.00m/3s
		uint8_t climbrate3s_H;
		uint8_t climbrate10s_L;		// climb rate LSB/MSB (0.01m/10s), 30000 == 0.00m/10s
		uint8_t climbrate10s_H;
		uint8_t ascii[21];			// 21 chars of text
		uint8_t ascii1;				// ASCII Free character [1]
		uint8_t ascii2;				// ASCII Free character [2]
		uint8_t ascii3;				// ASCII Free character [3]
		uint8_t unknown;
		uint8_t version;			// version number
		uint8_t stop;				// stop byte
		uint8_t checksum;			// Lower 8-bits of all bytes summed
	} *msg;
	msg = (struct hott_vario_message *)buffer;
	uint8_t config;

	HTelemetrySettingsVARIOGet(&config);
	if (config == HTELEMETRYSETTINGS_VARIO_DISABLED)
		return 0;

	// clear message buffer
	memset(buffer, 0, sizeof(*msg));

	/* message header */
	msg->start = HTELE_START;
	msg->stop = HTELE_STOP;
	msg->sensor_id = HTELE_VARIO_ID;
	msg->sensor_text_id = HTELE_VARIO_TEXT_ID;

	// update data
	if (AltHoldSmoothedHandle() != NULL)
		AltHoldSmoothedGet(&altState);
	if (BaroAltitudeHandle() != NULL)
		BaroAltitudeGet(&baroState);

	// altitude
	baroAltMin = (baroAltMin < baroState.Altitude) ? baroAltMin : baroState.Altitude;
	baroAltMax = (baroAltMax > baroState.Altitude) ? baroAltMax : baroState.Altitude;
	convert_float2word(baroState.Altitude, 1, 500, &msg->act_altitude_L, &msg->act_altitude_H);
	convert_float2word(baroAltMin, 1, 500, &msg->min_altitude_L, &msg->min_altitude_H);
	convert_float2word(baroAltMax, 1, 500, &msg->max_altitude_L, &msg->max_altitude_H);

	// climbrate
	convert_float2word(altState.Velocity, 100, 30000, &msg->climbrate_L, &msg->climbrate_H);
	convert_float2word(altState.Velocity, 300, 30000, &msg->climbrate3s_L, &msg->climbrate3s_H);
	convert_float2word(altState.Velocity, 1000, 30000, &msg->climbrate10s_L, &msg->climbrate10s_H);

	// text
	snprintf((char *)msg->ascii, sizeof(msg->ascii), "TauLabs Test");

	msg->checksum = calc_checksum(buffer, sizeof(*msg));
	return sizeof(*msg);
}

uint16_t build_GPS_message(uint8_t *buffer) {
	// GPS Module message structure
	struct hott_gps_message {
		uint8_t start;				// start byte
		uint8_t sensor_id;			// GPS sensor ID
		uint8_t warning;			// 0…= warning beeps
		uint8_t sensor_text_id;		// GPS Sensor text mode ID
		uint8_t alarm_inverse1;		// inverse status (1)
		uint8_t alarm_inverse2;		// inverse status (1 = no GPS signal)
		uint8_t flight_direction;	// flight direction (1 = 2°; 0° = north, 90° = east , 180° = south , 270° west)
		uint8_t gps_speed_L;		// GPS speed LSB/MSB in km/h
		uint8_t gps_speed_H;
		uint8_t latitude_ns;		// GPS latitude north/south (0 = N)
		uint8_t latitude_min_L;		// GPS latitude LSB/MSB (min)
		uint8_t latitude_min_H;
		uint8_t latitude_sec_L;		// GPS latitude LSB/MSB (sec)
		uint8_t latitude_sec_H;
		uint8_t longitude_ew;		// GPS longitude east/west (0 = E)
		uint8_t longitude_min_L;	// GPS longitude LSB/MSB (min)
		uint8_t longitude_min_H;
		uint8_t longitude_sec_L;	// GPS longitude LSB/MSB (sec)
		uint8_t longitude_sec_H;
		uint8_t distance_L;			// distance LSB/MSB (meters)
		uint8_t distance_H;
		uint8_t altitude_L;			// altitude LSB/MSB (meters), offset 500, 500 == 0m */
		uint8_t altitude_H;
		uint8_t climbrate_L;		// climb rate LSB/MSB in 0.01m/s, offset of 30000, 30000 = 0.00 m/s
		uint8_t climbrate_H;
		uint8_t climbrate3s;		// climb rate in m/3sec. offset of 120, 120 == 0m/3sec
		uint8_t gps_num_sat;		// GPS number of satelites */
		uint8_t gps_fix_char;		// GPS FixChar (GPS fix character. display, if DGPS, 2D oder 3D)
		uint8_t home_direction;		// home direction (direction from starting point to model position)
		uint8_t angle_x_direction;	// angle x-direction
		uint8_t angle_y_direction;	// angle y-direction
		uint8_t angle_z_direction;	// angle z-direction
		uint8_t gyro_x_L;			// gyro x LSB/MSB
		uint8_t gyro_x_H;
		uint8_t gyro_y_L;			// gyro y LSB/MSB
		uint8_t gyro_y_H;
		uint8_t gyro_z_L;			// gyro z lSB/MSB
		uint8_t gyro_z_H;
		uint8_t vibration;			// vibration
		uint8_t ascii4;				// ASCII Free Character [4]
		uint8_t ascii5;				// ASCII Free Character [5]
		uint8_t ascii6;				// ASCII Free Character [6]
		uint8_t version;			// version number
		uint8_t stop;				// stop byte
		uint8_t checksum;			// Lower 8-bits of all bytes summed
	} *msg;
	msg = (struct hott_gps_message *)buffer;
	uint8_t config;

	HTelemetrySettingsGPSGet(&config);
	if (config == HTELEMETRYSETTINGS_GPS_DISABLED)
		return 0;

	// clear message buffer
	memset(buffer, 0, sizeof(*msg));

	// message header
	msg->start = HTELE_START;
	msg->stop = HTELE_STOP;
	msg->sensor_id = HTELE_GPS_ID;
	msg->sensor_text_id = HTELE_GPS_TEXT_ID;

	msg->alarm_inverse1 = 1;
	msg->alarm_inverse2 = 0;

	// update data
	if (AltHoldSmoothedHandle() != NULL)
		AltHoldSmoothedGet(&altState);
	if (GPSPositionHandle() != NULL)
		GPSPositionGet(&gpsState);

	// gps
	// TODO home dir
	convert_long2gps(gpsState.Latitude, &msg->latitude_ns, &msg->latitude_min_L, &msg->latitude_min_H, &msg->latitude_sec_L, &msg->latitude_sec_H);
	convert_long2gps(gpsState.Longitude, &msg->longitude_ew, &msg->longitude_min_L, &msg->longitude_min_H, &msg->longitude_sec_L, &msg->longitude_sec_H);
	convert_float2word(gpsState.Altitude, 1, 500, &msg->altitude_L, &msg->altitude_H);
	convert_float2byte(gpsState.Heading, 1, 0, &msg->flight_direction);
	convert_float2word(gpsState.Groundspeed, 3.6, 0, &msg->gps_speed_L, &msg->gps_speed_H);
	msg->gps_num_sat = gpsState.Satellites;
	switch (gpsState.Status) {
		case GPSPOSITION_STATUS_NOGPS:
			msg->gps_fix_char = '-';
			break;
		case GPSPOSITION_STATUS_NOFIX:
			msg->gps_fix_char = ' ';
			break;
		case GPSPOSITION_STATUS_FIX2D:
			msg->gps_fix_char = '2';
			break;
		case GPSPOSITION_STATUS_FIX3D:
			msg->gps_fix_char = '3';
			break;
		default:
			msg->gps_fix_char = '?';
	}

	// climbrate
	convert_float2word(altState.Velocity, 100, 30000, &msg->climbrate_L, &msg->climbrate_H);
	convert_float2byte(altState.Velocity, 3, 120, &msg->climbrate3s);

	msg->checksum = calc_checksum(buffer, sizeof(*msg));
	return sizeof(*msg);
}

uint16_t build_GAM_message(uint8_t *buffer) {
	// General Air Module message structure
	struct hott_gam_message {
		uint8_t start;				// start byte
		uint8_t sensor_id;			// GAM sensor ID
		uint8_t warning;			// 0…= warning beeps
		uint8_t sensor_text_id;		// EAM Sensor text mode ID
		uint8_t alarm_inverse1;		// This inverts specific parts of display
		uint8_t alarm_inverse2;
		uint8_t cell1;				// Lipo cell voltages im Volt, 2mV steps, 210 == 4.2V
		uint8_t cell2;
		uint8_t cell3;
		uint8_t cell4;
		uint8_t cell5;
		uint8_t cell6;
		uint8_t batt1_voltage_L;	// Battery 1 voltage LSB/MSB in Volt, 0.1V steps, 50 == 5.5V
		uint8_t batt1_voltage_H;
		uint8_t batt2_voltage_L;	// Battery 2 voltage LSB/MSB
		uint8_t batt2_voltage_H;
		uint8_t temperature1;		// Temperature 1 in °C, offset of 20, 20 == 0°C
		uint8_t temperature2;		// Temperature 2
		uint8_t fuel_procent;		// Fuel capacity in %, values from 0..100
		uint8_t fuel_ml_L;			// Fuel capacity LSB/MSB in ml, values from 0..65535
		uint8_t fuel_ml_H;
		uint8_t rpm_L;				// RPM LSB/MSB, scale factor 10, 300 == 3000rpm
		uint8_t rpm_H;
		uint8_t altitude_L;			// altitude in meters LSB/MSB, offset of 500, 500 == 0m
		uint8_t altitude_H;
		uint8_t climbrate_L;		// climb rate in 0.01m/s, offset of 30000, 30000 = 0.00 m/s
		uint8_t climbrate_H;
		uint8_t climbrate3s;		// climb rate in m/3sec. offset of 120, 120 == 0m/3sec
		uint8_t current_L;			// current LSB/MSB in 0.1A steps
		uint8_t current_H;
		uint8_t main_voltage_L;		// main power LSB/MSB voltage in 0.1V steps
		uint8_t main_voltage_H;
		uint8_t batt_cap_L;			// used battery capacity LSB/MSB in 10mAh steps
		uint8_t batt_cap_H;
		uint8_t speed_L;			// Speed LSB/MSB in km/h
		uint8_t speed_H;
		uint8_t min_cell_volt;		// minimum cell voltage in 2mV steps. 124 == 2.48V
		uint8_t min_cell_volt_num;	// number of the cell with the lowest voltage
		uint8_t rpm2_L;				// RPM2 LSB/MSB in 10 rpm steps, 300 == 3000rpm
		uint8_t rpm2_H;
		uint8_t g_error_number;		// general error number (Voice error == 12)
		uint8_t pressure;			// pressure up to 16bar, 0.1bar steps
		uint8_t version;			// version number
		uint8_t stop;				// stop byte
		uint8_t checksum;			// Lower 8-bits of all bytes summed
	} *msg;
	msg = (struct hott_gam_message *)buffer;
	uint8_t config;

	HTelemetrySettingsGAMGet(&config);
	if (config == HTELEMETRYSETTINGS_GAM_DISABLED)
		return 0;

	// clear message buffer
	memset(buffer, 0, sizeof(*msg));

	// message header
	msg->start = HTELE_START;
	msg->stop = HTELE_STOP;
	msg->sensor_id = HTELE_GAM_ID;
	msg->sensor_text_id = HTELE_GAM_TEXT_ID;

	// update data
	if (AltHoldSmoothedHandle() != NULL)
		AltHoldSmoothedGet(&altState);
	if (BaroAltitudeHandle() != NULL)
		BaroAltitudeGet(&baroState);
	if (FlightBatteryStateHandle() != NULL)
		FlightBatteryStateGet(&battState);
	if (GyrosHandle() != NULL)
		GyrosGet(&gyroState);

	// batterie
	convert_float2word(battState.Voltage, .1, 0, &msg->batt1_voltage_L, &msg->batt1_voltage_H);
	convert_float2word(battState.Voltage, .1, 0, &msg->batt2_voltage_L, &msg->batt2_voltage_H);
	convert_float2word(battState.Voltage, .1, 0, &msg->main_voltage_L, &msg->main_voltage_H);

	// temperatures
	convert_float2byte(gyroState.temperature, 1, 20, &msg->temperature1);
	convert_float2byte(baroState.Temperature, 1, 20, &msg->temperature2);

	// altitude
	convert_float2word(baroState.Altitude, 1, 500, &msg->altitude_L, &msg->altitude_H);

	// climbrate
	convert_float2word(altState.Velocity, 100, 30000, &msg->climbrate_L, &msg->climbrate_H);
	convert_float2byte(altState.Velocity, 3, 120, &msg->climbrate3s);

	msg->checksum = calc_checksum(buffer, sizeof(*msg));
	return sizeof(*msg);
}

uint16_t build_EAM_message(uint8_t *buffer) {
	// Electric Air Module message structure
	struct hott_eam_message {
		uint8_t start;				// Start byte
		uint8_t sensor_id;			// EAM sensor id
		uint8_t warning;
		uint8_t sensor_text_id;		// EAM Sensor text mode ID
		uint8_t alarm_inverse1;
		uint8_t alarm_inverse2;
		uint8_t cell1_L;			// Lipo cell voltages
		uint8_t cell2_L;
		uint8_t cell3_L;
		uint8_t cell4_L;
		uint8_t cell5_L;
		uint8_t cell6_L;
		uint8_t cell7_L;
		uint8_t cell1_H;
		uint8_t cell2_H;
		uint8_t cell3_H;
		uint8_t cell4_H;
		uint8_t cell5_H;
		uint8_t cell6_H;
		uint8_t cell7_H;
		uint8_t batt1_voltage_L;	// Battery 1 voltage, lower 8-bits in steps of 0.02V
		uint8_t batt1_voltage_H;
		uint8_t batt2_voltage_L;	// Battery 2 voltage, lower 8-bits in steps of 0.02V
		uint8_t batt2_voltage_H;
		uint8_t temperature1;		// Temperature sensor 1. 20 = 0 degrees
		uint8_t temperature2;
		uint8_t altitude_L;			// Attitude (meters). 500 = 0 meters
		uint8_t altitude_H;
		uint8_t current_L;			// Current (A) in steps of 0.1A
		uint8_t current_H;
		uint8_t main_voltage_L;		// Main power voltage in steps of 0.1V
		uint8_t main_voltage_H;
		uint8_t battery_capacity_L;	// Used battery capacity in steps of 10mAh
		uint8_t battery_capacity_H;
		uint8_t climbrate_L;		// Climb rate in 0.01m/s. 0m/s = 30000
		uint8_t climbrate_H;
		uint8_t climbrate3s;		// Climb rate in m/3sec. 0m/3sec = 120
		uint8_t rpm_L;				// RPM Lower 8-bits In steps of 10 rpm
		uint8_t rpm_H;
		uint8_t electric_min;		// Flight time in minutes.
		uint8_t electric_sec;		// Flight time in seconds.
		uint8_t speed_L;			// Airspeed in km/h in steps of 1 km/h
		uint8_t speed_H;
		uint8_t stop;				// Stop byte
		uint8_t checksum;			// Lower 8-bits of all bytes summed.
	} *msg;
	msg = (struct hott_eam_message *)buffer;
	uint8_t config;

	HTelemetrySettingsEAMGet(&config);
	if (config == HTELEMETRYSETTINGS_EAM_DISABLED)
		return 0;

	// clear message buffer
	memset(buffer, 0, sizeof(*msg));

	// message header
	msg->start = HTELE_START;
	msg->stop = HTELE_STOP;
	msg->sensor_id = HTELE_EAM_ID;
	msg->sensor_text_id = HTELE_EAM_TEXT_ID;

	// update data
	if (AltHoldSmoothedHandle() != NULL)
		AltHoldSmoothedGet(&altState);
	if (BaroAltitudeHandle() != NULL)
		BaroAltitudeGet(&baroState);
	if (FlightBatteryStateHandle() != NULL)
		FlightBatteryStateGet(&battState);
	if (GyrosHandle() != NULL)
		GyrosGet(&gyroState);

	// batterie
	convert_float2word(battState.Voltage, .1, 0, &msg->batt1_voltage_L, &msg->batt1_voltage_H);
	convert_float2word(battState.Voltage, .1, 0, &msg->batt2_voltage_L, &msg->batt2_voltage_H);
	convert_float2word(battState.Voltage, .1, 0, &msg->main_voltage_L, &msg->main_voltage_H);
	convert_float2word(battState.Current, -.1, 0, &msg->current_L, &msg->current_H);
	convert_float2word(battState.ConsumedEnergy, -.1, 0, &msg->battery_capacity_L, &msg->battery_capacity_H);

	// temperatures
	convert_float2byte(gyroState.temperature, 1, 20, &msg->temperature1);
	convert_float2byte(baroState.Temperature, 1, 20, &msg->temperature2);

	// altitude
	convert_float2word(baroState.Altitude, 1, 500, &msg->altitude_L, &msg->altitude_H);

	// climbrate
	convert_float2word(altState.Velocity, 100, 30000, &msg->climbrate_L, &msg->climbrate_H);
	convert_float2byte(altState.Velocity, 3, 120, &msg->climbrate3s);

	msg->checksum = calc_checksum(buffer, sizeof(*msg));
	return sizeof(*msg);
}

uint16_t build_ESC_message(uint8_t *buffer) {
	// ESC Module message structure
	struct hott_esc_message {
		uint8_t start;				// Start byte
		uint8_t sensor_id;			// EAM sensor id
		uint8_t warning;
		uint8_t sensor_text_id;		// ESC Sensor text mode ID
		uint8_t alarm_inverse1;
		uint8_t alarm_inverse2;
		uint8_t batt_max_voltage_L;	// Battery maximum voltage, lower 8-bits in steps of 0.01V
		uint8_t batt_max_voltage_H;
		uint8_t batt_voltage_L;		// Battery voltage, lower 8-bits in steps of 0.01V
		uint8_t batt_voltage_H;
		uint8_t batt_capacity_L;	// Used battery capacity in steps of 10mAh
		uint8_t batt_capacity_H;
		uint8_t temperature1;		// Temperature sensor 1. 20 = 0 degrees
		uint8_t temperature2;
		uint8_t current_L;			// Current (A) in steps of 0.1A
		uint8_t current_H;
		uint8_t current_max_L;		// maximal Current (A) in steps of 0.1A
		uint8_t current_max_H;
		uint8_t rpm_L;				// RPM Lower 8-bits In steps of 10 rpm
		uint8_t rpm_H;
		uint8_t rpm_max_L;			// maxmimal RPM Lower 8-bits In steps of 10 rpm
		uint8_t rpm_max_H;
		uint8_t dn_L;
		uint8_t dn_H;
		uint8_t dn2_L;
		uint8_t dn2_H;
		uint8_t dummy[16];			// 16 dummy bytes
		uint8_t stop;				// Stop byte
		uint8_t checksum;			// Lower 8-bits of all bytes summed.
	} *msg;
	msg = (struct hott_esc_message *)buffer;
	uint8_t config;

	HTelemetrySettingsESCGet(&config);
	if (config == HTELEMETRYSETTINGS_ESC_DISABLED)
		return 0;

	// clear message buffer
	memset(buffer, 0, sizeof(*msg));

	// message header
	msg->start = HTELE_START;
	msg->stop = HTELE_STOP;
	msg->sensor_id = HTELE_ESC_ID;
	msg->sensor_text_id = HTELE_ESC_TEXT_ID;

	// dummy values
	msg->dn_L = 0x7a;
	msg->dn2_L = 0x14;
	msg->dn2_H = 0x14;

	msg->checksum = calc_checksum(buffer, sizeof(*msg));
	return sizeof(*msg);
}

uint16_t build_TEXT_message(uint8_t *buffer) {
	// textmode message structure
	struct hott_text_message {
		uint8_t start;				// Start byte
		uint8_t sensor_id;			// EAM sensor id
		uint8_t warning;
		uint8_t text[21][8];		// text field 21 columns and 8 rows (bit 7=1 for inverse display)
		uint8_t stop;				// Stop byte
		uint8_t checksum;			// Lower 8-bits of all bytes summed.
	} *msg;
	msg = (struct hott_text_message *)buffer;

	// message header
	msg->start = HTELE_START;
	msg->stop = HTELE_STOP;
	msg->sensor_id = HTELE_TEXT_ID;

	msg->checksum = calc_checksum(buffer, sizeof(*msg));
	return sizeof(*msg);
}

/**
 * calculate checksum of data buffer
 */
uint8_t calc_checksum(uint8_t *data, uint16_t size) {
	uint16_t sum = 0;
	for(int i = 0; i < size; i++)
		sum += data[i];
	return sum;
}

/**
 * convert float value with scale and offset to byte and write result to given pointer.
 */
void convert_float2byte(float val, float scale, uint16_t offset, uint8_t *lsb) {
	uint16_t temp = (uint16_t)((val * scale) + offset);
	*lsb = (uint8_t)temp & 0xff;
}

/**
 * convert float value with scale and offset to word and write result to given lsb/msb pointer.
 */
void convert_float2word(float val, float scale, uint16_t offset, uint8_t *lsb, uint8_t *msb) {
	uint16_t temp = (uint16_t)((val * scale) + offset);
	*lsb = (uint8_t)temp & 0xff;
	*msb = (uint8_t)(temp >> 8) & 0xff;
}

/**
 * convert dword gps value into HoTT gps format and write result to given pointer.
 */
void convert_long2gps(int32_t val, uint8_t *dir, uint8_t *min_lsb, uint8_t *min_msb, uint8_t *sec_lsb, uint8_t *sec_msb) {
	//convert gps decigrad value into degrees, minutes and seconds
	uint32_t absval = abs(val);
	uint16_t deg = (absval / 10000000);
	uint32_t sec = (absval - deg * 10000000) * 6;
	uint16_t min = sec / 1000000;
	sec %= 1000000;
	sec = sec / 100;
	uint16_t degmin = deg * 100 + min;
	// write results
	*dir = (val < 0) ? 1 : 0;
	*min_lsb = (uint8_t)degmin & 0xff;
	*min_msb = (uint8_t)(degmin >> 8) & 0xff;
	*sec_lsb = (uint8_t)sec & 0xff;
	*sec_msb = (uint8_t)(sec >> 8) & 0xff;
}


/**
 * @}
 * @}
 */
