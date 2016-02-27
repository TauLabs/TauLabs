/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 *
 * @file       msplib.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015-2016
 * @brief      Library for handling MSP protocol communications
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

#ifndef MSPLIB_H
#define MSPLIB_H

#include "openpilot.h"
#include "pios.h"
#include "flightstatus.h"

#define MSP_MAX_PACKET_SIZE 16

/* MSP constants */

#define MSP_SENSOR_ACC 1
#define MSP_SENSOR_BARO 2
#define MSP_SENSOR_MAG 4
#define MSP_SENSOR_GPS 8

// Magic numbers copied from mwosd
#define  MSP_IDENT      100 // multitype + multiwii version + protocol version + capability variable
#define  MSP_STATUS     101 // cycletime & errors_count & sensor present & box activation & current setting number
#define  MSP_RAW_IMU    102 // 9 DOF
#define  MSP_SERVO      103 // 8 servos
#define  MSP_MOTOR      104 // 8 motors
#define  MSP_RC         105 // 8 rc chan and more
#define  MSP_RAW_GPS    106 // fix, numsat, lat, lon, alt, speed, ground course
#define  MSP_COMP_GPS   107 // distance home, direction home
#define  MSP_ATTITUDE   108 // 2 angles 1 heading
#define  MSP_ALTITUDE   109 // altitude, variometer
#define  MSP_ANALOG     110 // vbat, powermetersum, rssi if available on RX
#define  MSP_RC_TUNING  111 // rc rate, rc expo, rollpitch rate, yaw rate, dyn throttle PID
#define  MSP_PID        112 // P I D coeff (9 are used currently)
#define  MSP_BOX        113 // BOX setup (number is dependant of your setup)
#define  MSP_MISC       114 // powermeter trig
#define  MSP_MOTOR_PINS 115 // which pins are in use for motors & servos, for GUI
#define  MSP_BOXNAMES   116 // the aux switch names
#define  MSP_PIDNAMES   117 // the PID names
#define  MSP_BOXIDS     119 // get the permanent IDs associated to BOXes
#define  MSP_NAV_STATUS 121 // Returns navigation status
#define  MSP_CELLS      130 // FrSky SPort Telemtry
#define  MSP_ALARMS     242 // Alarm request

/* Data formats for sending and receiving */

//! MSP flight modes
typedef enum {
	MSP_BOX_ARM,
	MSP_BOX_ANGLE,
	MSP_BOX_HORIZON,
	MSP_BOX_BARO,
	MSP_BOX_VARIO,
	MSP_BOX_MAG,
	MSP_BOX_GPSHOME,
	MSP_BOX_GPSHOLD,
	MSP_BOX_LAST,
} msp_box_t;

struct msp_packet_status {
	uint16_t cycleTime;
	uint16_t i2cErrors;
	uint16_t sensors;
	uint32_t flags;
	uint8_t setting;
} __attribute__((packed));

struct msp_packet_attitude {
	int16_t x;
	int16_t y;
	int16_t h;
} __attribute__((packed));

struct msp_packet_analog {
	uint8_t vbat;
	uint16_t powerMeterSum;
	uint16_t rssi;
	uint16_t current;
} __attribute__((packed));

struct msp_packet_altitude {
	int32_t alt; // cm
	uint16_t vario; // cm/s
} __attribute__((packed));

struct msp_packet_command {
	uint16_t channels[8];
} __attribute__((packed));

struct msp_packet_boxids {
	uint8_t boxes[MSP_BOX_LAST];
} __attribute__((packed));

struct msp_packet_rawgps {
	uint8_t  fix;                 // 0 or 1
	uint8_t  num_sat;
	int32_t lat;                  // 1 / 10 000 000 deg
	int32_t lon;                  // 1 / 10 000 000 deg
	uint16_t alt;                 // meter
	uint16_t speed;               // cm/s
	int16_t ground_course;        // degree * 10
} __attribute__((packed));

struct msp_packet_compgps {
	uint16_t distance_to_home;     // meter
	int16_t  direction_to_home;    // degree [-180:180]
	uint8_t  home_position_valid;  // 0 = Invalid
} __attribute__((packed));

union msp_data {
	uint8_t data[MSP_MAX_PACKET_SIZE];
	struct msp_packet_status status;
	struct msp_packet_attitude attitude;
	struct msp_packet_analog analog;
	struct msp_packet_altitude altitude;
	struct msp_packet_command command;
	struct msp_packet_boxids boxids;
	struct msp_packet_rawgps rawgps;
	struct msp_packet_compgps compgps;
};

//! Map between our flight modes
const static struct {
	msp_box_t mode;
	uint8_t mwboxid;
	FlightStatusFlightModeOptions tlmode;
} msp_boxes[] = {
	{ MSP_BOX_ARM, 0, 0 },
	{ MSP_BOX_ANGLE, 1, FLIGHTSTATUS_FLIGHTMODE_LEVELING},
	{ MSP_BOX_HORIZON, 2, FLIGHTSTATUS_FLIGHTMODE_HORIZON},
	{ MSP_BOX_BARO, 3, FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD},
	{ MSP_BOX_VARIO, 4, 0},
	{ MSP_BOX_MAG, 5, 0},
	{ MSP_BOX_GPSHOME, 10, FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME},
	{ MSP_BOX_GPSHOLD, 11, FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD},
	{ MSP_BOX_LAST, 0xff, 0},
};

//! MSP parser states
typedef enum {
	MSP_IDLE,
	MSP_HEADER_START,
	MSP_HEADER_M,
	/* These states mean we are in a request or command packet */
	MSP_HEADER_C_SIZE,
	MSP_HEADER_C_CMD,
	MSP_C_FILLBUF,
	MSP_C_CHECKSUM,
	/* These states mean we are in a response packet */
	MSP_HEADER_R_SIZE,
	MSP_HEADER_R_CMD,
	MSP_R_FILLBUF,
	MSP_R_CHECKSUM,
	/* Throw away any bytes we can't consume */
	MSP_DISCARD,
} msp_state;

typedef bool (*msp_cb_store)(void *msp, uint8_t cmd, const uint8_t *data, size_t len);

//! Track all the state information for the parser
struct msp_bridge {
	uintptr_t com;

	msp_state state;

	uint8_t cmd_size;
	uint8_t cmd_id;
	uint8_t cmd_i;
	uint8_t checksum;

	msp_cb_store response_cb;
	msp_cb_store request_cb;

	union msp_data cmd_data;
};

// We do a little dance here to have a nice clean function prototype for the outside
// API but internally we have to have the first parameter as void* in order to avoid
// a circular definition
typedef bool (*msp_cb)(struct msp_bridge *msp, uint8_t cmd, const uint8_t *data, size_t len);

//! Allocate memory for an MSP bridge
struct msp_bridge * msp_init(uintptr_t com);

//! Send response to a data request
void msp_send_response(struct msp_bridge *m, uint8_t cmd, const uint8_t *data, size_t len);

//! Send a request for a data update
void msp_send_request(struct msp_bridge *m, uint8_t type);

//! Consume and process bytes received by the parser
bool msp_receive_byte(struct msp_bridge *m, uint8_t b);

void msp_set_response_cb(struct msp_bridge *m, msp_cb response_cb);
void msp_set_request_cb(struct msp_bridge *m, msp_cb request_cb);

#endif /* MSPLIB_H */

/**
 * @}
 */
