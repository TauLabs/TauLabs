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
#include "attitudeactual.h"
#include "altholdsmoothed.h"
#include "baroaltitude.h"
#include "flightbatterystate.h"
#include "flightstatus.h"
#include "gyros.h"
#include "gpsposition.h"
#include "gpstime.h"
#include "homelocation.h"
#include "positionactual.h"
#include "systemalarms.h"

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

#define StatusSize 21

// Private structures and types
typedef struct {
		uint8_t l;
		uint8_t h;
} uword_t;

struct telemetryData{
	HTelemetrySettingsData Settings;
	AltHoldSmoothedData Altitude;
	AttitudeActualData Attitude;
	BaroAltitudeData Baro;
	FlightBatteryStateData Battery;
	FlightStatusData FlightStatus;
	GPSPositionData GPS;
	GPSTimeData GPStime;
	GyrosData Gyro;
	HomeLocationData Home;
	PositionActualData Position;
	SystemAlarmsData SysAlarms;
	char StatusLine[StatusSize];
};

// Private variables
static xTaskHandle HTelemetryTaskHandle;
static uint32_t htelemetry_port;
static bool module_enabled = false;
static uint8_t *tx_buffer;
static struct telemetryData *teleState;
static float climbrate1s;
static float climbrate3s;
static float climbrate10s;
static float altitude1s;
static float altitudeOffset;
static float altitudeGround;
static float altitudeMin;
static float altitudeMax;
static float altitudeLast;
static float homedistance;
static float homecourse;
static uint8_t LastArmed;

// Private functions
static void HTelemetryTask(void *parameters);
static uint16_t build_VARIO_message(uint8_t *buffer);
static uint16_t build_GPS_message(uint8_t *buffer);
static uint16_t build_GAM_message(uint8_t *buffer);
static uint16_t build_EAM_message(uint8_t *buffer);
static uint16_t build_ESC_message(uint8_t *buffer);
static uint16_t build_TEXT_message(uint8_t *buffer);
static uint8_t calc_checksum(uint8_t *data, uint16_t size);
static uint8_t generate_warning();
static void update_telemetrydata();
static void convert_long2gps(int32_t value, uint8_t *dir, uword_t *min, uword_t *sec);
static uint8_t scale_float2uint8(float value, float scale, float offset);
static int8_t scale_float2int8(float value, float scale, float offset);
static uword_t scale_float2uword(float value, float scale, float offset);

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

	// allocate memory for telemetry data
	teleState = (struct telemetryData *)pvPortMalloc(sizeof(*teleState));

	// check buffer pointer for security
	if ((tx_buffer == NULL) || (teleState == NULL))
		return;

	// clear all state values
	memset(teleState, 0, sizeof(*teleState));

	// 500ms delay for sensor setup
	vTaskDelay(500);

	// first update of telemetry data
	update_telemetrydata(teleState);

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
				if (state != IDLE) {
					// get actual HTelemetrysettings before generate messages
					update_telemetrydata();
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
		uint8_t alarm_inverse;		// this inverts specific parts of display
		uword_t altitude;			// altitude (meters), offset 500, 500 == 0m
		uword_t max_altitude;		// max. altitude (meters)
		uword_t min_altitude;		// min. altitude (meters)
		uword_t climbrate;			// climb rate (0.01m/s), offset 30000, 30000 == 0.00m/s
		uword_t climbrate3s;		// climb rate (0.01m/3s)
		uword_t climbrate10s;		// climb rate (0.01m/10s)
		uint8_t ascii[21];			// 21 chars of text
		uint8_t ascii1;				// ASCII Free character [1] (next to Alt)
		uint8_t ascii2;				// ASCII Free character [2] (next to Dir)
		uint8_t ascii3;				// ASCII Free character [3] (next to I)
		int8_t  compass;			// 1=2°, -1=-2°
		uint8_t version;			// version number
		uint8_t stop;				// stop byte
		uint8_t checksum;			// Lower 8-bits of all bytes summed
	} *msg;
	msg = (struct hott_vario_message *)buffer;

	if (teleState->Settings.Sensor[HTELEMETRYSETTINGS_SENSOR_VARIO] == HTELEMETRYSETTINGS_SENSOR_DISABLED)
		return 0;

	// clear message buffer
	memset(buffer, 0, sizeof(*msg));

	// message header
	msg->start = HTELE_START;
	msg->stop = HTELE_STOP;
	msg->sensor_id = HTELE_VARIO_ID;
	msg->warning = generate_warning();
	msg->sensor_text_id = HTELE_VARIO_TEXT_ID;

	// alarm inverse bits. invert display areas on limits
	msg->alarm_inverse |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXHEIGHT] < altitudeGround) ? (1<<1) : 0;
	msg->alarm_inverse |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINHEIGHT] > altitudeGround) ? (1<<2) : 0;
	msg->alarm_inverse |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_NEGDIFFERENCE1] > climbrate1s) ? (1<<3) : 0;
	msg->alarm_inverse |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_POSDIFFERENCE1] < climbrate1s) ? (1<<3) : 0;
	msg->alarm_inverse |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_NEGDIFFERENCE2] > climbrate3s) ? (1<<4) : 0;
	msg->alarm_inverse |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_POSDIFFERENCE2] < climbrate3s) ? (1<<4) : 0;
	msg->alarm_inverse |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_NEGDIFFERENCE2] > climbrate10s) ? (1<<5) : 0;
	msg->alarm_inverse |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_POSDIFFERENCE2] < climbrate10s) ? (1<<5) : 0;

	// altitude relative to ground
	msg->altitude = scale_float2uword(altitudeGround, 1, 500);
	msg->min_altitude = scale_float2uword(altitudeMin, 1, 500);
	msg->max_altitude = scale_float2uword(altitudeMax, 1, 500);

	// climbrate
	msg->climbrate = scale_float2uword(climbrate1s, 100, 30000);
	msg->climbrate3s = scale_float2uword(climbrate3s, 100, 30000);
	msg->climbrate10s = scale_float2uword(climbrate10s, 100, 30000);

	// compass
	msg->compass = scale_float2int8(teleState->Attitude.Yaw, 0.5, 0);

	// statusline
	memcpy(msg->ascii, teleState->StatusLine, sizeof(msg->ascii));

	// free display characters
	msg->ascii1 = 0;
	msg->ascii2 = 0;
	msg->ascii3 = 0;

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
		uint8_t alarm_inverse1;		// this inverts specific parts of display
		uint8_t alarm_inverse2;
		uint8_t flight_direction;	// flight direction (1 = 2°; 0° = north, 90° = east , 180° = south , 270° west)
		uword_t gps_speed;			// GPS speed (km/h)
		uint8_t latitude_ns;		// GPS latitude north/south (0 = N)
		uword_t latitude_min;		// GPS latitude (min)
		uword_t latitude_sec;		// GPS latitude (sec)
		uint8_t longitude_ew;		// GPS longitude east/west (0 = E)
		uword_t longitude_min;		// GPS longitude (min)
		uword_t longitude_sec;		// GPS longitude (sec)
		uword_t distance;			// distance from home location (meters)
		uword_t altitude;			// altitude (meters), offset 500, 500 == 0m 
		uword_t climbrate;			// climb rate (0.01m/s), offset of 30000, 30000 = 0.00 m/s
		uint8_t climbrate3s;		// climb rate in (m/3s). offset of 120, 120 == 0m/3sec
		uint8_t gps_num_sat;		// GPS number of satelites
		uint8_t gps_fix_char;		// GPS FixChar ('D'=DGPS, '2'=2D, '3'=3D)
		uint8_t home_direction;		// home direction (1=2°, direction from starting point to model position)
		int8_t  angle_roll;			// angle x-direction (roll 1=2°, 255=-2°)
		int8_t  angle_nick;			// angle y-direction (nick)
		int8_t  angle_compass;		// angle z-direction (yaw)
		uint8_t gps_hour;			// GPS time hours
		uint8_t gps_min;			// GPS time minutes
		uint8_t gps_sec;			// GPS time seconds
		uint8_t gps_msec;			// GPS time .sss
		uword_t msl;				// MSL or NN altitude
		uint8_t vibration;			// vibration
		uint8_t ascii4;				// ASCII Free Character [4] (next to home distance)
		uint8_t ascii5;				// ASCII Free Character [5] (next to home direction)
		uint8_t ascii6;				// ASCII Free Character [6] (next to number of sats)
		uint8_t version;			// version number (0=gps, 1=gyro, 255=multicopter)
		uint8_t stop;				// stop byte
		uint8_t checksum;			// Lower 8-bits of all bytes summed
	} *msg;
	msg = (struct hott_gps_message *)buffer;

	if (teleState->Settings.Sensor[HTELEMETRYSETTINGS_SENSOR_GPS] == HTELEMETRYSETTINGS_SENSOR_DISABLED)
		return 0;

	// clear message buffer
	memset(buffer, 0, sizeof(*msg));

	// message header
	msg->start = HTELE_START;
	msg->stop = HTELE_STOP;
	msg->sensor_id = HTELE_GPS_ID;
	msg->warning = generate_warning();
	msg->sensor_text_id = HTELE_GPS_TEXT_ID;

	// alarm inverse bits. invert display areas on limits
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXDISTANCE] < homedistance) ? (1<<0) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINSPEED] > teleState->GPS.Groundspeed) ? (1<<1) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXSPEED] < teleState->GPS.Groundspeed) ? (1<<1) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINHEIGHT] > altitudeGround) ? (1<<2) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXHEIGHT] < altitudeGround) ? (1<<2) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_NEGDIFFERENCE1] > climbrate1s) ? (1<<3) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_POSDIFFERENCE1] < climbrate1s) ? (1<<3) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_NEGDIFFERENCE2] > climbrate3s) ? (1<<4) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_POSDIFFERENCE2] < climbrate3s) ? (1<<4) : 0;
	msg->alarm_inverse2 |= (teleState->SysAlarms.Alarm[SYSTEMALARMS_ALARM_GPS] != SYSTEMALARMS_ALARM_OK) ? (1<<0) : 0;

	// gps direction, groundspeed and postition
	msg->flight_direction = scale_float2uint8(teleState->GPS.Heading, 0.5, 0);
	msg->gps_speed = scale_float2uword(teleState->GPS.Groundspeed, 3.6, 0);
	convert_long2gps(teleState->GPS.Latitude, &msg->latitude_ns, &msg->latitude_min, &msg->latitude_sec);
	convert_long2gps(teleState->GPS.Longitude, &msg->longitude_ew, &msg->longitude_min, &msg->longitude_sec);

	// homelocation distance, course and state
	msg->distance = scale_float2uword(homedistance, 1, 0);
	msg->home_direction = scale_float2uint8(homecourse, 0.5, 0);
	msg->ascii5 = (teleState->Home.Set ? 'H' : '-');

	// altitude relative to ground and climb rate
	msg->altitude = scale_float2uword(altitudeGround, 1, 500);
	msg->climbrate = scale_float2uword(climbrate1s, 100, 30000);
	msg->climbrate3s = scale_float2uint8(climbrate3s, 1, 120);

	// number of satellites,gps fix and state
	msg->gps_num_sat = teleState->GPS.Satellites;
	switch (teleState->GPS.Status) {
		case GPSPOSITION_STATUS_FIX2D:
			msg->gps_fix_char = '2';
		case GPSPOSITION_STATUS_FIX3D:
			msg->gps_fix_char = '3';
		default:
			msg->gps_fix_char = 0;
	}
	switch (teleState->SysAlarms.Alarm[SYSTEMALARMS_ALARM_GPS]) {
		case SYSTEMALARMS_ALARM_UNINITIALISED:
			msg->ascii6 = 0;
			break;
		case SYSTEMALARMS_ALARM_OK:
			msg->ascii6 = '.';
			break;
		case SYSTEMALARMS_ALARM_WARNING:
			msg->ascii6 = '?';
			break;
		case SYSTEMALARMS_ALARM_ERROR:
		case SYSTEMALARMS_ALARM_CRITICAL:
			msg->ascii6 = '!';
			break;
		default:
			msg->ascii6 = 0;
	}

	// model angles
	msg->angle_roll = scale_float2int8(teleState->Attitude.Roll, 0.5, 0);
	msg->angle_nick = scale_float2int8(teleState->Attitude.Pitch, 0.5, 0);
	msg->angle_compass = scale_float2int8(teleState->Attitude.Yaw, 0.5, 0);

	// gps time
	msg->gps_hour = teleState->GPStime.Hour;
	msg->gps_min = teleState->GPStime.Minute;
	msg->gps_sec = teleState->GPStime.Second;
	msg->gps_msec = 0;

	// gps MSL (NN) altitude MSL
	msg->msl = scale_float2uword(teleState->GPS.Altitude, 1, 0);

	// free display chararacter
	msg->ascii4 = 0;

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
		uint8_t alarm_inverse1;		// this inverts specific parts of display
		uint8_t alarm_inverse2;
		uint8_t cell1;				// lipo cell voltages in 0.02V steps, 210 == 4.2V
		uint8_t cell2;
		uint8_t cell3;
		uint8_t cell4;
		uint8_t cell5;
		uint8_t cell6;
		uword_t batt1_voltage;		// battery sensor 1 in 0.1V steps, 50 == 5.5V
		uword_t batt2_voltage;		// battery sensor 2 voltage
		uint8_t temperature1;		// temperature 1 in °C, offset of 20, 20 == 0°C
		uint8_t temperature2;		// temperature 2
		uint8_t fuel_procent;		// fuel capacity in %, values from 0..100
		uword_t fuel_ml;			// fuel capacity in ml, values from 0..65535
		uword_t rpm;				// rpm, scale factor 10, 300 == 3000rpm
		uword_t altitude;			// altitude in meters, offset of 500, 500 == 0m
		uword_t climbrate;			// climb rate (0.01m/s), offset of 30000, 30000 = 0.00 m/s
		uint8_t climbrate3s;		// climb rate (m/3sec). offset of 120, 120 == 0m/3sec
		uword_t current;			// current in 0.1A steps
		uword_t voltage;			// main power voltage in 0.1V steps
		uword_t capacity;			// used battery capacity in 10mAh steps
		uword_t speed;				// speed in km/h
		uint8_t min_cell_volt;		// lowest cell voltage in 20mV steps. 124 == 2.48V
		uint8_t min_cell_volt_num;	// number of the cell with the lowest voltage
		uword_t rpm2;				// rpm2 in 10 rpm steps, 300 == 3000rpm
		uint8_t g_error_number;		// general error number (Voice error == 12)
		uint8_t pressure;			// pressure up to 15bar, 0.1bar steps
		uint8_t version;			// version number
		uint8_t stop;				// stop byte
		uint8_t checksum;			// Lower 8-bits of all bytes summed
	} *msg;
	msg = (struct hott_gam_message *)buffer;

	if (teleState->Settings.Sensor[HTELEMETRYSETTINGS_SENSOR_GAM] == HTELEMETRYSETTINGS_SENSOR_DISABLED)
		return 0;

	// clear message buffer
	memset(buffer, 0, sizeof(*msg));

	// message header
	msg->start = HTELE_START;
	msg->stop = HTELE_STOP;
	msg->sensor_id = HTELE_GAM_ID;
	msg->warning = generate_warning();
	msg->sensor_text_id = HTELE_GAM_TEXT_ID;

	// alarm inverse bits. invert display areas on limits
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINHEIGHT] > altitudeGround) ? (1<<7) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXHEIGHT] < altitudeGround) ? (1<<7) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXCURRENT] < teleState->Battery.Current) ? (1<<0) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINPOWERVOLTAGE] > teleState->Battery.Voltage) ? (1<<1) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXPOWERVOLTAGE] < teleState->Battery.Voltage) ? (1<<1) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINHEIGHT] > altitudeGround) ? (1<<2) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXHEIGHT] < altitudeGround) ? (1<<2) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_NEGDIFFERENCE1] > climbrate1s) ? (1<<3) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_POSDIFFERENCE1] < climbrate1s) ? (1<<3) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_NEGDIFFERENCE2] > climbrate3s) ? (1<<4) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_POSDIFFERENCE2] < climbrate3s) ? (1<<4) : 0;

	// temperatures
	msg->temperature1 = scale_float2uint8(teleState->Gyro.temperature, 1, 20);
	msg->temperature2 = scale_float2uint8(teleState->Baro.Temperature, 1, 20);

	// altitude
	msg->altitude = scale_float2uword(altitudeGround, 1, 500);

	// climbrate
	msg->climbrate = scale_float2uword(climbrate1s, 100, 30000);
	msg->climbrate3s = scale_float2uint8(climbrate3s, 1, 120);

	// main battery
	float voltage = (teleState->Battery.Voltage > 0) ? teleState->Battery.Voltage : 0;
	float current = (teleState->Battery.Current > 0) ? teleState->Battery.Current : 0;
	float energy = (teleState->Battery.ConsumedEnergy > 0) ? teleState->Battery.ConsumedEnergy : 0;
	msg->voltage = scale_float2uword(voltage, 10, 0);
	msg->current = scale_float2uword(current, 10, 0);
	msg->capacity = scale_float2uword(energy, 1, 0);

	// pressure kPa to 0.1Bar
	msg->pressure = scale_float2uint8(teleState->Baro.Pressure, 0.1, 0);

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
		uint8_t alarm_inverse1;		// this inverts specific parts of display
		uint8_t alarm_inverse2;
		uint8_t cell1_L;			// Lipo cell voltages of lower one
		uint8_t cell2_L;
		uint8_t cell3_L;
		uint8_t cell4_L;
		uint8_t cell5_L;
		uint8_t cell6_L;
		uint8_t cell7_L;
		uint8_t cell1_H;			// Lipo cell voltages of higher one
		uint8_t cell2_H;
		uint8_t cell3_H;
		uint8_t cell4_H;
		uint8_t cell5_H;
		uint8_t cell6_H;
		uint8_t cell7_H;
		uword_t batt1_voltage;		// battery sensor 1 voltage, in steps of 0.02V
		uword_t batt2_voltage;		// battery sensor 2 voltage, in steps of 0.02V
		uint8_t temperature1;		// temperature sensor 1. 20 = 0 degrees
		uint8_t temperature2;		// temperature sensor 2. 20 = 0 degrees
		uword_t altitude;			// altitude (meters). 500 = 0 meters
		uword_t current;			// current (A) in steps of 0.1A
		uword_t voltage;			// main power voltage in steps of 0.1V
		uword_t capacity;			// used battery capacity in steps of 10mAh
		uword_t climbrate;			// climb rate in 0.01m/s. 0m/s = 30000
		uint8_t climbrate3s;		// climb rate in m/3sec. 0m/3sec = 120
		uword_t rpm;				// rpm in steps of 10 rpm
		uint8_t electric_min;		// estaminated flight time in minutes.
		uint8_t electric_sec;		// estaminated flight time in seconds.
		uword_t speed;				// speed in km/h in steps of 1 km/h
		uint8_t stop;				// Stop byte
		uint8_t checksum;			// Lower 8-bits of all bytes summed.
	} *msg;
	msg = (struct hott_eam_message *)buffer;

	if (teleState->Settings.Sensor[HTELEMETRYSETTINGS_SENSOR_EAM] == HTELEMETRYSETTINGS_SENSOR_DISABLED)
		return 0;

	// clear message buffer
	memset(buffer, 0, sizeof(*msg));

	// message header
	msg->start = HTELE_START;
	msg->stop = HTELE_STOP;
	msg->sensor_id = HTELE_EAM_ID;
	msg->warning = generate_warning();
	msg->sensor_text_id = HTELE_EAM_TEXT_ID;

	// alarm inverse bits. invert display areas on limits
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXUSEDCAPACITY] < teleState->Battery.ConsumedEnergy) ? (1<<0) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINHEIGHT] > altitudeGround) ? (1<<5) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXHEIGHT] < altitudeGround) ? (1<<5) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXCURRENT] < teleState->Battery.Current) ? (1<<6) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINPOWERVOLTAGE] > teleState->Battery.Voltage) ? (1<<7) : 0;
	msg->alarm_inverse1 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXPOWERVOLTAGE] < teleState->Battery.Voltage) ? (1<<7) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_NEGDIFFERENCE2] > climbrate3s) ? (1<<0) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_POSDIFFERENCE2] < climbrate3s) ? (1<<0) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_NEGDIFFERENCE1] > climbrate1s) ? (1<<1) : 0;
	msg->alarm_inverse2 |= (teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_POSDIFFERENCE1] < climbrate1s) ? (1<<1) : 0;

	// main battery
	float voltage = (teleState->Battery.Voltage > 0) ? teleState->Battery.Voltage : 0;
	float current = (teleState->Battery.Current > 0) ? teleState->Battery.Current : 0;
	float energy = (teleState->Battery.ConsumedEnergy > 0) ? teleState->Battery.ConsumedEnergy : 0;
	msg->voltage = scale_float2uword(voltage, 10, 0);
	msg->current = scale_float2uword(current, 10, 0);
	msg->capacity = scale_float2uword(energy, 1, 0);

	// temperatures
	msg->temperature1 = scale_float2uint8(teleState->Gyro.temperature, 1, 20);
	msg->temperature2 = scale_float2uint8(teleState->Baro.Temperature, 1, 20);

	// altitude
	msg->altitude = scale_float2uword(altitudeGround, 1, 500);

	// climbrate
	msg->climbrate = scale_float2uword(climbrate1s, 100, 30000);
	msg->climbrate3s = scale_float2uint8(climbrate3s, 1, 120);

	// flight time
	msg->electric_min = teleState->Battery.EstimatedFlightTime / 60;
	msg->electric_sec = teleState->Battery.EstimatedFlightTime - 60 * msg->electric_min;

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
		uword_t batt_voltage;		// battery voltage in steps of 0.1V
		uword_t min_batt_voltage;	// min battery voltage
		uword_t batt_capacity;		// used battery capacity in steps of 10mAh
		uint8_t temperatureESC;		// temperature of ESC . 20 = 0 degrees
		uint8_t max_temperatureESC;	// max temperature of ESC
		uword_t current;			// Current in steps of 0.1A
		uword_t max_current;		// maximal current
		uword_t rpm;				// rpm in steps of 10 rpm
		uword_t max_rpm;			// max rpm
		uint8_t temperatureMOT;		// motor temperature
		uint8_t max_temperatureMOT;	// maximal motor temperature
		uint8_t dummy[19];			// 19 dummy bytes
		uint8_t stop;				// Stop byte
		uint8_t checksum;			// Lower 8-bits of all bytes summed.
	} *msg;
	msg = (struct hott_esc_message *)buffer;

	if (teleState->Settings.Sensor[HTELEMETRYSETTINGS_SENSOR_ESC] == HTELEMETRYSETTINGS_SENSOR_DISABLED)
		return 0;

	// clear message buffer
	memset(buffer, 0, sizeof(*msg));

	// message header
	msg->start = HTELE_START;
	msg->stop = HTELE_STOP;
	msg->sensor_id = HTELE_ESC_ID;
	msg->warning = 0;
	msg->sensor_text_id = HTELE_ESC_TEXT_ID;

	// main batterie
	float voltage = (teleState->Battery.Voltage > 0) ? teleState->Battery.Voltage : 0;
	float current = (teleState->Battery.Current > 0) ? teleState->Battery.Current : 0;
	float energy = (teleState->Battery.ConsumedEnergy > 0) ? teleState->Battery.ConsumedEnergy : 0;
	msg->batt_voltage = scale_float2uword(voltage, 10, 0);
	msg->current = scale_float2uword(current, 10, 0);
	msg->batt_capacity = scale_float2uword(energy, 10, 0);

	// temperatures
	msg->temperatureESC = scale_float2uint8(teleState->Gyro.temperature, 1, 20);
	msg->max_temperatureESC = scale_float2uint8(0, 1, 20);
	msg->temperatureMOT = scale_float2uint8(teleState->Baro.Temperature, 1, 20);
	msg->max_temperatureMOT = scale_float2uint8(0, 1, 20);

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
 * update telemetry data
 * this is called on every telemetry request
 * calling intervall is 200ms depending on TX
 * 200ms telemetry request is used as time base for timed calculations (5Hz interval)
*/
void update_telemetrydata () {
	// update all available data
	if (HTelemetrySettingsHandle() != NULL)
		HTelemetrySettingsGet(&teleState->Settings);
	if (AltHoldSmoothedHandle() != NULL)
		AltHoldSmoothedGet(&teleState->Altitude);
	if (AttitudeActualHandle() != NULL)
		AttitudeActualGet(&teleState->Attitude);
	if (BaroAltitudeHandle() != NULL)
		BaroAltitudeGet(&teleState->Baro);
	if (FlightBatteryStateHandle() != NULL)
		FlightBatteryStateGet(&teleState->Battery);
	if (FlightStatusHandle() != NULL)
		FlightStatusGet(&teleState->FlightStatus);
	if (GPSPositionHandle() != NULL)
		GPSPositionGet(&teleState->GPS);
	if (GPSTimeHandle() != NULL)
		GPSTimeGet(&teleState->GPStime);
	if (GyrosHandle() != NULL)
		GyrosGet(&teleState->Gyro);
	if (HomeLocationHandle() != NULL)
		HomeLocationGet(&teleState->Home);
	if (PositionActualHandle() != NULL)
		PositionActualGet(&teleState->Position);
	if (SystemAlarmsHandle() != NULL)
		SystemAlarmsGet(&teleState->SysAlarms);

	// climbrate and altitude calculation TODO: change this with loop buffer
	climbrate1s = teleState->Altitude.Velocity;
	climbrate3s = 3 * teleState->Altitude.Velocity;
	climbrate10s = 10 * teleState->Altitude.Velocity;
	altitude1s = teleState->Baro.Altitude;

	// set altitude offset and clear min/max values when arming
	if ((teleState->FlightStatus.Armed == FLIGHTSTATUS_ARMED_ARMING) || ((LastArmed != FLIGHTSTATUS_ARMED_ARMED) && (teleState->FlightStatus.Armed == FLIGHTSTATUS_ARMED_ARMED))) {
		altitudeOffset = altitude1s;
		altitudeMin = 0;
		altitudeMax = 0;
	}
	LastArmed = teleState->FlightStatus.Armed;

	// calculate altitude relative to start position
	altitudeGround = altitude1s - altitudeOffset;

	// check and set min/max values when armed.
	if (teleState->FlightStatus.Armed == FLIGHTSTATUS_ARMED_ARMED) {
		if (altitudeMin > altitudeGround)
			altitudeMin = altitudeGround;
		if (altitudeMax < altitudeGround)
			altitudeMax = altitudeGround;
	}

	// gps home position and course
	homedistance = sqrtf(teleState->Position.North * teleState->Position.North + teleState->Position.East * teleState->Position.East);
	homecourse = acosf(- teleState->Position.North / homedistance) / 3.14159265f * 180;
	if (teleState->Position.East > 0)
		homecourse = 360 - homecourse;

	// statusline
	switch (teleState->FlightStatus.FlightMode) {
		case FLIGHTSTATUS_FLIGHTMODE_MANUAL:
			snprintf(teleState->StatusLine, sizeof(teleState->StatusLine), "Manual");
			break;
		case FLIGHTSTATUS_FLIGHTMODE_STABILIZED1:
			snprintf(teleState->StatusLine, sizeof(teleState->StatusLine), "Stabilized1");
			break;
		case FLIGHTSTATUS_FLIGHTMODE_STABILIZED2:
			snprintf(teleState->StatusLine, sizeof(teleState->StatusLine), "Stabilized2");
			break;
		case FLIGHTSTATUS_FLIGHTMODE_STABILIZED3:
			snprintf(teleState->StatusLine, sizeof(teleState->StatusLine), "Stabilized3");
			break;
		case FLIGHTSTATUS_FLIGHTMODE_AUTOTUNE:
			snprintf(teleState->StatusLine, sizeof(teleState->StatusLine), "Autotune");
			break;
		case FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD:
			snprintf(teleState->StatusLine, sizeof(teleState->StatusLine), "AltitudeHold");
			break;
		case FLIGHTSTATUS_FLIGHTMODE_VELOCITYCONTROL:
			snprintf(teleState->StatusLine, sizeof(teleState->StatusLine), "VelocityControl");
			break;
		case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
			snprintf(teleState->StatusLine, sizeof(teleState->StatusLine), "PositionHold");
			break;
		case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:
			snprintf(teleState->StatusLine, sizeof(teleState->StatusLine), "ReturnToHome");
			break;
		case FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER:
			snprintf(teleState->StatusLine, sizeof(teleState->StatusLine), "PathPlanner");
			break;
		default:
			snprintf(teleState->StatusLine, sizeof(teleState->StatusLine), "unkown");
	}
}

/**
 * generate warning beeps or announcements
*/
uint8_t generate_warning() {
	// set warning tone with hardcoded priority
	//tone a
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MINSPEED] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINSPEED] > teleState->GPS.Groundspeed * 3.6f))
		return 1;
	//tone b
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_NEGDIFFERENCE2] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_NEGDIFFERENCE2] > climbrate3s))
		return 2;
	// tone c
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_NEGDIFFERENCE1] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_NEGDIFFERENCE2] > climbrate1s))
		return 3;
	// tone d
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MAXDISTANCE] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXDISTANCE] < homedistance))
		return 4;
	// tone e (unused)

	// tone f
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MINSENSOR1TEMP] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINSENSOR1TEMP] > teleState->Gyro.temperature))
		return 6;
	// tone g
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MINSENSOR2TEMP] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINSENSOR2TEMP] > teleState->Baro.Temperature))
		return 7;
	// tone h
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MAXSENSOR1TEMP] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXSENSOR1TEMP] < teleState->Gyro.temperature))
		return 8;
	// tone i
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MAXSENSOR2TEMP] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXSENSOR2TEMP] < teleState->Baro.Temperature))
		return 9;
	// tone j (10, MAXSENSOR1VOLTAGE)

	// tone k (11, MAXSENSOR2VOLTAGE)

	//tone l
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MAXSPEED] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXSPEED] < teleState->GPS.Groundspeed * 3.6f))
		return 12;
	//tone m
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_POSDIFFERENCE2] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_POSDIFFERENCE2] > climbrate3s))
		return 13;
	// tone n
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_POSDIFFERENCE1] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_POSDIFFERENCE1] > climbrate1s))
		return 14;
	// tone o
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MINHEIGHT] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINHEIGHT] > altitudeGround))
		return 15;
	// tone p
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MINPOWERVOLTAGE] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MINPOWERVOLTAGE] > teleState->Battery.Voltage))
		return 16;
	// tone q (17, MINCELLVOLTAGE)

	// tone r (18, MINSENSOR1VOLTAGE)

	// tone s (19, MINSENSOR2VOLTAGE)

	// tone t (20, MINRPM)

	// tone u (21, MINFUEL)

	// tone v
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MAXUSEDCAPACITY] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXUSEDCAPACITY] < teleState->Battery.ConsumedEnergy))
		return 22;
	// tone w
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MAXCURRENT] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXCURRENT] < teleState->Battery.Current))
		return 23;
	// tone x
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MAXPOWERVOLTAGE] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXPOWERVOLTAGE] < teleState->Battery.Voltage))
		return 24;
	// tone y (25, MAXRPM)

	// tone z
	if ((teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_MAXHEIGHT] == HTELEMETRYSETTINGS_WARNING_ENABLED) &&
		(teleState->Settings.Limit[HTELEMETRYSETTINGS_LIMIT_MAXHEIGHT] < altitudeGround))
		return 26;

	// altitude beeps at 20,40,60,80,100,200,400,600,800 and 1000 meters
	if (teleState->Settings.Warning[HTELEMETRYSETTINGS_WARNING_ALTITUDEBEEP] == HTELEMETRYSETTINGS_WARNING_ENABLED) {
		// update altitude when checked for beeps
		float last = altitudeLast;
		float actual = altitudeGround;
		altitudeLast = altitudeGround;
		if (((last < 20) && (actual > 20)) || ((last > 20) && (actual < 20)))
			return 37;
		if (((last < 40) && (actual > 40)) || ((last > 40) && (actual < 40)))
			return 38;
		if (((last < 60) && (actual > 60)) || ((last > 60) && (actual < 60)))
			return 39;
		if (((last < 80) && (actual > 80)) || ((last > 80) && (actual < 80)))
			return 40;
		if (((last < 100) && (actual > 100)) || ((last > 100) && (actual < 100)))
			return 41;
		if (((last < 200) && (actual > 200)) || ((last > 200) && (actual < 200)))
			return 46;
		if (((last < 400) && (actual > 400)) || ((last > 400) && (actual < 400)))
			return 47;
		if (((last < 600) && (actual > 600)) || ((last > 600) && (actual < 600)))
			return 48;
		if (((last < 800) && (actual > 800)) || ((last > 800) && (actual < 800)))
			return 49;
		if (((last < 1000) && (actual > 1000)) || ((last > 1000) && (actual < 1000)))
			return 50;
	}

	// there is no warning
	return 0;
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
 * scale float value with scale and offset to unsigned byte
 */
uint8_t scale_float2uint8(float value, float scale, float offset) {
	uint16_t temp = (uint16_t)roundf(value * scale + offset);
	uint8_t result;
	result = (uint8_t)temp & 0xff;
	return result;
}

/**
 * scale float value with scale and offset to signed byte (int8_t)
 */
int8_t scale_float2int8(float value, float scale, float offset) {
	int8_t result = (int8_t)roundf(value * scale + offset);
	return result;
}

/**
 * scale float value with scale and offset to word
 */
uword_t scale_float2uword(float value, float scale, float offset) {
	uint16_t temp = (uint16_t)roundf(value * scale + offset);
	uword_t result;
	result.l = (uint8_t)temp & 0xff;
	result.h = (uint8_t)(temp >> 8) & 0xff;
	return result;
}

/**
 * convert dword gps value into HoTT gps format and write result to given pointers
 */
void convert_long2gps(int32_t value, uint8_t *dir, uword_t *min, uword_t *sec) {
	//convert gps decigrad value into degrees, minutes and seconds
	uword_t temp;
	uint32_t absvalue = abs(value);
	uint16_t degrees = (absvalue / 10000000);
	uint32_t seconds = (absvalue - degrees * 10000000) * 6;
	uint16_t minutes = seconds / 1000000;
	seconds %= 1000000;
	seconds = seconds / 100;
	uint16_t degmin = degrees * 100 + minutes;
	// write results
	*dir = (value < 0) ? 1 : 0;
	temp.l = (uint8_t)degmin & 0xff;
	temp.h = (uint8_t)(degmin >> 8) & 0xff;
	*min = temp;
	temp.l = (uint8_t)seconds & 0xff;
	temp.h = (uint8_t)(seconds >> 8) & 0xff;
	*sec = temp;
}

/**
 * @}
 * @}
 */
