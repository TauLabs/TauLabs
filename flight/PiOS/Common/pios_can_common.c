/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_CAN PiOS CAN interface layer
 * @brief CAN interface for PiOS
 * @{
 *
 * @file       pios_can_common.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      PiOS CAN common methods across platforms
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


#include "pios_can.h"

//! The mapping of message types to CAN BUS StdID
const uint32_t pios_can_message_stdid[PIOS_CAN_LAST] = {
	[PIOS_CAN_GIMBAL] = 0x130,
	[PIOS_CAN_ATTITUDE_ROLL_PITCH] = 0x185,
	[PIOS_CAN_ATTITUDE_YAW] = 0x196,
	[PIOS_CAN_BATTERY_VOLT] = 0x2A0,
	[PIOS_CAN_BATTERY_CURR] = 0x2A1,
	[PIOS_CAN_RSSI] = 0x2A2,
	[PIOS_CAN_ALT] = 0x2A3,
	[PIOS_CAN_FLIGHTSTATUS] = 0x2A4,
	[PIOS_CAN_GPS_LATLON] = 0x2B1,
	[PIOS_CAN_GPS_ALTSPEED] = 0x2B2,
	[PIOS_CAN_GPS_FIX] = 0x2B3,
	[PIOS_CAN_GPS_VEL] = 0x2B4,
	[PIOS_CAN_POS] = 0x2B5,
	[PIOS_CAN_VERT] = 0x2B6,
	[PIOS_CAN_ALARM] = 0x2B7,
};

//! Map between message IDs and structures
int32_t get_message_size(uint32_t msg_id) {
	switch(msg_id) {
	case PIOS_CAN_GIMBAL:
		return sizeof(struct pios_can_gimbal_message);
	case PIOS_CAN_ATTITUDE_ROLL_PITCH:
		return sizeof(struct pios_can_roll_pitch_message);
	case PIOS_CAN_ATTITUDE_YAW:
		return sizeof(struct pios_can_yaw_message);
	case PIOS_CAN_BATTERY_VOLT:
		return sizeof(struct pios_can_volt_message);
	case PIOS_CAN_BATTERY_CURR:
		return sizeof(struct pios_can_curr_message);
	case PIOS_CAN_RSSI:
		return sizeof(struct pios_can_rssi_message);
	case PIOS_CAN_ALT:
		return sizeof(struct pios_can_alt_message);
	case PIOS_CAN_FLIGHTSTATUS:
		return sizeof(struct pios_can_flightstatus_message);
	case PIOS_CAN_GPS_LATLON:
		return sizeof(struct pios_can_gps_latlon);
	case PIOS_CAN_GPS_ALTSPEED:
		return sizeof(struct pios_can_gps_alt_speed);
	case PIOS_CAN_GPS_FIX:
		return sizeof(struct pios_can_gps_fix);
	case PIOS_CAN_GPS_VEL:
		return sizeof(struct pios_can_gps_vel);
	case PIOS_CAN_POS:
		return sizeof(struct pios_can_pos);
	case PIOS_CAN_VERT:
		return sizeof(struct pios_can_vert);
	case PIOS_CAN_ALARM:
		return sizeof(struct pios_can_alarm_message);
	default:
		return -1;
	}
}
