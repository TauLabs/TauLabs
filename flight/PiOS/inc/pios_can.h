/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_CAN PiOS CAN interface layer
 * @brief CAN interface for PiOS
 * @{
 *
 * @file       pios_can.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

//! Transmit a data message with a particular message ID
int32_t PIOS_CAN_TxData(uintptr_t id, uint32_t std_id, uint8_t *data, uint32_t bytes);

//! Fetch the last data message with a particular message ID
int32_t PIOS_CAN_RxData(uintptr_t id, uint32_t std_id, uint8_t *data);

#endif /* PIOS_CAN_H */

/**
 * @}
 * @}
 */
