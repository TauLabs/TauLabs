/**
 ******************************************************************************
 * @addtogroup TauLabsBootloader Tau Labs Bootloaders
 * @{
 * @addtogroup FlyingF3BL FlyingF3 bootloader
 * @{
 *
 * @file       pios_board.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Board specific initialization for the bootloader
 * @see        The GNU Public License (GPL) Version 3
 *
 *****************************************************************************//* 
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

/* Pull in the board-specific static HW definitions.
 * Including .c files is a bit ugly but this allows all of
 * the HW definitions to be const and static to limit their
 * scope.  
 *
 * NOTE: THIS IS THE ONLY PLACE THAT SHOULD EVER INCLUDE THIS FILE
 */
#include "board_hw_defs.c"

#include <pios_board_info.h>
#include <pios.h>

uintptr_t pios_com_telem_usb_id;

void PIOS_Board_Init()
{
	/* Delay system */
	PIOS_DELAY_Init();

#if defined(PIOS_INCLUDE_LED)
	PIOS_LED_Init(&pios_led_cfg);
#endif	/* PIOS_INCLUDE_LED */

#if defined(PIOS_INCLUDE_FLASH)
	/* Inititialize all flash drivers */
	PIOS_Flash_Internal_Init(&pios_internal_flash_id, &flash_internal_cfg);

	/* Register the partition table */
	PIOS_FLASH_register_partition_table(pios_flash_partition_table, NELEMENTS(pios_flash_partition_table));
#endif	/* PIOS_INCLUDE_FLASH */

#if defined(PIOS_INCLUDE_USB)
	/* Initialize board specific USB data */
	PIOS_USB_BOARD_DATA_Init();

	/* Activate the HID-only USB configuration */
	PIOS_USB_DESC_HID_ONLY_Init();

	uintptr_t pios_usb_id;
	PIOS_USB_Init(&pios_usb_id, &pios_usb_main_cfg);

#if defined(PIOS_INCLUDE_USB_HID) && defined(PIOS_INCLUDE_COM_MSG)
	uintptr_t pios_usb_hid_id;
	if (PIOS_USB_HID_Init(&pios_usb_hid_id, &pios_usb_hid_cfg, pios_usb_id)) {
		PIOS_Assert(0);
	}
	if (PIOS_COM_MSG_Init(&pios_com_telem_usb_id, &pios_usb_hid_com_driver, pios_usb_hid_id)) {
		PIOS_Assert(0);
	}
#endif	/* PIOS_INCLUDE_USB_HID && PIOS_INCLUDE_COM_MSG */

#endif	/* PIOS_INCLUDE_USB */
}
