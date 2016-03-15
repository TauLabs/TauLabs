/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_CAN PiOS CAN interface layer
 * @brief CAN interface for PiOS
 * @{
 *
 * @file       pios_can.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2016
 * @brief      PiOS CAN interface header
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

#if !defined(PIOS_CAN_H)
#define PIOS_CAN_H

#include "pios_queue.h"

//! The set of CAN messages
enum pios_can_messages {
	PIOS_CAN_ERROR = -1,
	PIOS_CAN_GIMBAL = 0,
	PIOS_CAN_ATTITUDE_ROLL_PITCH = 1,
	PIOS_CAN_ATTITUDE_YAW = 2,
	PIOS_CAN_BATTERY_VOLT = 3,
	PIOS_CAN_BATTERY_CURR = 4,
	PIOS_CAN_RSSI = 5,
	PIOS_CAN_ALT = 6,
	PIOS_CAN_FLIGHTSTATUS = 7,
	PIOS_CAN_GPS_LATLON = 8,
	PIOS_CAN_GPS_ALTSPEED = 9,
	PIOS_CAN_GPS_FIX = 10,
	PIOS_CAN_GPS_VEL = 11,
	PIOS_CAN_POS = 12,
	PIOS_CAN_VERT = 13, // carries smoothed altitude and climbrate
	PIOS_CAN_ALARM = 14,
	PIOS_CAN_LAST
};
// Note: new messages must be defined in both
//    pios_can_message_stdid
//    get_message_size
// in pios_can.c

//! Message to tell gimbal the desired setpoint and FC state
struct pios_can_gimbal_message {
	int8_t fc_roll;
	int8_t fc_pitch;
	uint8_t fc_yaw;
	int8_t setpoint_roll;
	int8_t setpoint_pitch;
	uint8_t setpoint_yaw;
}  __attribute__((packed));

//! Message to pass attitude information
struct pios_can_roll_pitch_message {
	float fc_roll;
	float fc_pitch;
}  __attribute__((packed));

//! Message to pass attitude information
struct pios_can_yaw_message {
	float fc_yaw;
}  __attribute__((packed));

//! Message to pass attitude information
struct pios_can_volt_message {
	float volt;
}  __attribute__((packed));

//! Message to pass attitude information
struct pios_can_curr_message {
	float curr;
	float consumed;
}  __attribute__((packed));

//! Message to pass rssi information
struct pios_can_rssi_message {
	int16_t rssi;
}  __attribute__((packed));

//! Message to pass altitude information
struct pios_can_alt_message {
	float fc_alt;
}  __attribute__((packed));

//! Message to pass rssi information
struct pios_can_flightstatus_message {
	uint8_t flight_mode;
	uint8_t armed;
}  __attribute__((packed));

//! Message to pass GPS lat and lon information
struct pios_can_gps_latlon {
	uint32_t lat;
	uint32_t lon;
}  __attribute__((packed));

//! Message to pass GPS lat and lon information
struct pios_can_gps_alt_speed {
	float alt;
	float speed;
}  __attribute__((packed));

struct pios_can_gps_fix {
	float pdop;
	uint8_t sats;
	uint8_t status;
}  __attribute__((packed));

struct pios_can_gps_vel {
	float north;
	float east;
}  __attribute__((packed));

struct pios_can_pos {
	float north;
	float east;
}  __attribute__((packed));

struct pios_can_vert {
	float pos_down;
	float rate_down;
}  __attribute__((packed));

struct pios_can_alarm_message {
	uint8_t alarms[8];
}  __attribute__((packed));

//! Transmit a data message with a particular message ID
int32_t PIOS_CAN_TxData(uintptr_t id, enum pios_can_messages, uint8_t *data);

//! Get a queue to receive messages of a particular message ID
struct pios_queue * PIOS_CAN_RegisterMessageQueue(uintptr_t id, enum pios_can_messages msg_id);

#endif /* PIOS_CAN_H */

/**
 * @}
 * @}
 */
