/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_CAN PiOS CAN interface layer
 * @brief CAN interface for PiOS
 * @{
 *
 * @file       pios_can.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
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
 
#if defined(PIOS_INCLUDE_FREERTOS)

//! The set of CAN messages
enum pios_can_messages {
	PIOS_CAN_GIMBAL = 0,
	PIOS_CAN_LAST = 1
};

//! Message to tell gimbal the desired setpoint and FC state
struct pios_can_gimbal_message {
	int8_t fc_roll;
	int8_t fc_pitch;
	uint8_t fc_yaw;
	int8_t setpoint_roll;
	int8_t setpoint_pitch;
	uint8_t setpoint_yaw;
};

//! Transmit a data message with a particular message ID
int32_t PIOS_CAN_TxData(uintptr_t id, enum pios_can_messages, uint8_t *data);

//! Get a queue to receive messages of a particular message ID
struct pios_queue * PIOS_CAN_RegisterMessageQueue(uintptr_t id, enum pios_can_messages msg_id);

#endif /* PIOS_INCLUDE_FREERTOS */

#endif /* PIOS_CAN_H */

/**
 * @}
 * @}
 */
