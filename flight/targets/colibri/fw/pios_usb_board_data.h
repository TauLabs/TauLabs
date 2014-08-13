/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup Colibri Colibri support files
 * @{
 *
 * @file       pios_usb_board_data.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Defines for board specific usb information
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

#ifndef PIOS_USB_BOARD_DATA_H
#define PIOS_USB_BOARD_DATA_H

#define PIOS_USB_BOARD_CDC_DATA_LENGTH 64
#define PIOS_USB_BOARD_CDC_MGMT_LENGTH 32
#define PIOS_USB_BOARD_HID_DATA_LENGTH 64

#define PIOS_USB_BOARD_EP_NUM 4

#include "pios_usb_defs.h" 	/* USB_* macros */


#define PIOS_USB_BOARD_VENDOR_ID 0x0fda		//Team Black Sheep
#define PIOS_USB_BOARD_PRODUCT_ID 0x0100	//colibri flight control rev. 1
#define PIOS_USB_BOARD_DEVICE_VER USB_OP_DEVICE_VER(0, USB_OP_BOARD_MODE_FW)
#define PIOS_USB_BOARD_SN_SUFFIX "+FW"

#endif	/* PIOS_USB_BOARD_DATA_H */
