/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup UAVOHoTTBridge HoTT Telemetry Module
 * @{ 
 *
 * @file       uavohottbridge.h
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
#include "hottsettings.h"
#include "attitudeactual.h"
#include "baroaltitude.h"
#include "flightbatterystate.h"
#include "flightstatus.h"
#include "gyros.h"
#include "gpsposition.h"
#include "gpstime.h"
#include "homelocation.h"
#include "positionactual.h"
#include "systemalarms.h"
#include "velocityactual.h"

// timing variables
#define IDLE_TIME 10	// idle line delay to prevent data crashes on telemetry line.
#define DATA_TIME 3		// time between 2 transmitted bytes

// sizes and lengths
#define climbratesize 50			// defines size of ring buffer for climbrate calculation
#define statussize 21				// number of characters in status line
#define HOTT_MAX_MESSAGE_LENGTH 200

// scale factors
#define M_TO_CM				100		// scale m to cm or m/s to cm/s 
#define MS_TO_KMH			3.6f	// scale m/s to km/h
#define DEG_TO_UINT			0.5f	// devide degrees by 2. then the value fits into a byte.

// offsets. used to make transmitted values unsigned.
#define OFFSET_ALTITUDE		500		// 500m
#define OFFSET_CLIMBRATE	30000	// 30000cm/s or 300m/s
#define OFFSET_CLIMBRATE3S	120		// 120m/s
#define OFFSET_TEMPERATURE	20		// 20 degrees

// Bits to invert display areas or values
#define VARIO_INVERT_ALT (1<<0)			// altitude
#define VARIO_INVERT_MAX (1<<1)			// max altitude
#define VARIO_INVERT_MIN (1<<2)			// min altitude
#define VARIO_INVERT_CR1S (1<<3)		// climbrate 1s
#define VARIO_INVERT_CR3S (1<<4)		// climbrate 3s
#define VARIO_INVERT_CR10S (1<<5)		// climbrate 10s

#define GPS_INVERT_HDIST (1<<0)			// home distance
#define GPS_INVERT_SPEED (1<<1)			// speed (kmh)
#define GPS_INVERT_ALT (1<<2)			// altitude
#define GPS_INVERT_CR1S (1<<3)			// climbrate 1s
#define GPS_INVERT_CR3S (1<<4)			// climbrate 3s
#define GPS_INVERT2_POS (1<<0)			// GPS postion values

#define GAM_INVERT_CELL (1<<0)			// cell voltage
#define GAM_INVERT_BATT1 (1<<1)			// battery 1 voltage
#define GAM_INVERT_BATT2 (1<<2)			// battery 1 voltage
#define GAM_INVERT_TEMP1 (1<<3)			// temperature 1
#define GAM_INVERT_TEMP2 (1<<4)			// temperature 2
#define GAM_INVERT_FUEL (1<<5)			// fuel
#define GAM_INVERT2_CURRENT (1<<0)		// current
#define GAM_INVERT2_VOLTAGE (1<<1)		// voltage
#define GAM_INVERT2_ALT (1<<2)			// altitude
#define GAM_INVERT2_CR1S (1<<3)			// climbrate 1s
#define GAM_INVERT2_CR3S (1<<4)			// climbrate 3s

#define EAM_INVERT_CAPACITY (1<<0)		// capacity
#define EAM_INVERT_BATT1 (1<<1)			// battery 1 voltage
#define EAM_INVERT_BATT2 (1<<2)			// battery 1 voltage
#define EAM_INVERT_TEMP1 (1<<3)			// temperature 1
#define EAM_INVERT_TEMP2 (1<<4)			// temperature 2
#define EAM_INVERT_ALT (1<<5)			// altitude
#define EAM_INVERT_CURRENT (1<<6)		// current
#define EAM_INVERT_VOLTAGE (1<<7)		// voltage
#define EAM_INVERT2_ALT (1<<2)			// altitude
#define EAM_INVERT2_CR1S (1<<3)			// climbrate 1s
#define EAM_INVERT2_CR3S (1<<4)			// climbrate 3s

#define ESC_INVERT_VOLTAGE (1<<0)		// voltage
#define ESC_INVERT_TEMP1 (1<<1)			// temperature 1
#define ESC_INVERT_TEMP2 (1<<2)			// temperature 2
#define ESC_INVERT_CURRENT (1<<3)		// current
#define ESC_INVERT_RPM (1<<4)			// rpm 
#define ESC_INVERT_CAPACITY (1<<5)		// capacity
#define ESC_INVERT_MAXCURRENT (1<<6)	// maximum current

// message codes
#define HOTT_TEXT_ID 0x7f			// Text request
#define HOTT_BINARY_ID 0x80			// Binary request
#define HOTT_VARIO_ID 0x89			// Vario Module ID
#define HOTT_VARIO_TEXT_ID 0x90		// Vario Module TEXT ID
#define HOTT_GPS_ID 0x8a			// GPS Module ID
#define HOTT_GPS_TEXT_ID 0xa0		// GPS Module TEXT ID
#define HOTT_ESC_ID 0x8c			// ESC Module ID
#define HOTT_ESC_TEXT_ID 0xc0		// ESC Module TEXT ID
#define HOTT_GAM_ID 0x8d			// General Air Module ID
#define HOTT_GAM_TEXT_ID 0xd0		// General Air Module TEXT ID
#define HOTT_EAM_ID 0x8e			// Electric Air Module ID
#define HOTT_EAM_TEXT_ID 0xe0		// Electric Air Module TEXT ID
#define HOTT_TEXT_START 0x7b		// Start byte Text mode
#define HOTT_START 0x7c				// Start byte Binary mode
#define HOTT_STOP 0x7d				// End byte
#define HOTT_BUTTON_DEC 0xEB		// minus button
#define HOTT_BUTTON_INC 0xED		// plus button
#define HOTT_BUTTON_SET 0xE9		// set button
#define HOTT_BUTTON_NIL 0x0F		// esc button
#define HOTT_BUTTON_NEXT 0xEE		// next button
#define HOTT_BUTTON_PREV 0xE7		// previous button

// prefined signal tones or spoken announcments 
#define HOTT_TONE_A		1	// minimum speed
#define HOTT_TONE_B		2	// sink rate 3 seconds
#define HOTT_TONE_C		3	// sink rate 1 second
#define HOTT_TONE_D		4	// maximum distance
#define HOTT_TONE_E		5	// -
#define HOTT_TONE_F		6	// minimum temperature sensor 1
#define HOTT_TONE_G		7	// minimum temperature sensor 2
#define HOTT_TONE_H		8	// maximum temperature sensor 1
#define HOTT_TONE_I		9	// maximum temperature sensor 2 
#define HOTT_TONE_J		10	// overvoltage sensor 1
#define HOTT_TONE_K		11	// overvoltage sensor 2
#define HOTT_TONE_L		12	// maximum speed
#define HOTT_TONE_M		13	// climb rate 3 seconds
#define HOTT_TONE_N		14	// climb rate 1 second
#define HOTT_TONE_O		15	// minimum height
#define HOTT_TONE_P		16	// minimum input voltage
#define HOTT_TONE_Q		17	// minimum cell voltage
#define HOTT_TONE_R		18	// undervoltage sensor 1
#define HOTT_TONE_S		19	// undervoltage sensor 2
#define HOTT_TONE_T		20	// minimum rpm
#define HOTT_TONE_U		21	// fuel reserve
#define HOTT_TONE_V		22	// capacity
#define HOTT_TONE_W		23	// maximum current
#define HOTT_TONE_X		24	// maximum input voltage
#define HOTT_TONE_Y		25	// maximum rpm
#define HOTT_TONE_Z		26	// maximum height
#define HOTT_TONE_20M	37	// 20 meters
#define HOTT_TONE_40M	38	// 40 meters
#define HOTT_TONE_60M	39	// 60 meters
#define HOTT_TONE_80M	40	// 80 meters
#define HOTT_TONE_100M	41	// 100 meters
#define HOTT_TONE_42	42	// receiver voltage
#define HOTT_TONE_43	43	// receiver temperature
#define HOTT_TONE_200M	46	// 200 meters
#define HOTT_TONE_400M	47	// 400 meters
#define HOTT_TONE_600M	48	// 600 meters
#define HOTT_TONE_800M	49	// 800 meters
#define HOTT_TONE_1000M	50	// 10000 meters
#define HOTT_TONE_51	51	// maximum servo temperature
#define HOTT_TONE_52	52	// maximum servo position difference


// Private types
typedef struct {
		uint8_t l;
		uint8_t h;
} uword_t;

// Private structures
struct telemetrydata{
	HoTTSettingsData Settings;
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
	VelocityActualData Velocity;
	float climbratebuffer[climbratesize];
	uint8_t climbrate_pointer;
	float altitude;
	float min_altitude;
	float max_altitude;
	float altitude_last;
	float climbrate1s;
	float climbrate3s;
	float climbrate10s;
	float homedistance;
	float homecourse;
	uint8_t last_armed;
	char statusline[statussize];
};

// VARIO Module message structure
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
};

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
};

// General Air Module message structure
struct hott_gam_message {
	uint8_t start;				// start byte
	uint8_t sensor_id;			// GAM sensor ID
	uint8_t warning;			// 0…= warning beeps
	uint8_t sensor_text_id;		// EAM Sensor text mode ID
	uint8_t alarm_inverse1;		// this inverts specific parts of display
	uint8_t alarm_inverse2;
	uint8_t cell1;				// cell voltages in 0.02V steps, 210 == 4.2V
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
};

// Electric Air Module message structure
struct hott_eam_message {
	uint8_t start;				// Start byte
	uint8_t sensor_id;			// EAM sensor id
	uint8_t warning;
	uint8_t sensor_text_id;		// EAM Sensor text mode ID
	uint8_t alarm_inverse1;		// this inverts specific parts of display
	uint8_t alarm_inverse2;
	uint8_t cell1_L;			// cell voltages of the lower battery
	uint8_t cell2_L;
	uint8_t cell3_L;
	uint8_t cell4_L;
	uint8_t cell5_L;
	uint8_t cell6_L;
	uint8_t cell7_L;
	uint8_t cell1_H;			// cell voltages of higher battery
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
};

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
};

// TEXT message structure
struct hott_text_message {
	uint8_t start;				// Start byte
	uint8_t sensor_id;			// TEXT id
	uint8_t warning;
	uint8_t text[21][8];		// text field 21 columns and 8 rows (bit 7=1 for inverse display)
	uint8_t stop;				// Stop byte
	uint8_t checksum;			// Lower 8-bits of all bytes summed.
};


/**
 * @}
 * @}
 */