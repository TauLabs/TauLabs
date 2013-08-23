/**
 ******************************************************************************
 * @addtogroup TauLabsBootloader Tau Labs Bootloaders
 * @{
 * @addtogroup BootloaderUpdate Update the bootloader stored in a target
 * @{
 *
 * @file       main.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Starting point for the code
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

/* Bootloader Includes */
#include <pios.h>
#include "pios_board_info.h"

/* Prototype of PIOS_Board_Init() function */
extern void PIOS_Board_Init(void);
void error(int);

/* The ADDRESSES of the _bu_payload_* symbols are the important data */
extern const uint32_t _bu_payload_start;
extern const uint32_t _bu_payload_end;
extern const uint32_t _bu_payload_size;

int main(void) {

	PIOS_SYS_Init();
	PIOS_Board_Init();
	PIOS_LED_On(PIOS_LED_HEARTBEAT);
	PIOS_DELAY_WaitmS(3000);
	PIOS_LED_Off(PIOS_LED_HEARTBEAT);

	/*
	 * Make sure the bootloader we're carrying is for the same
	 * board type and board revision as the one we're running on.
	 *
	 * Assume the bootloader in flash and the bootloader contained in
	 * the updater both carry a board_info_blob at the end of the image.
	 */

	/* Calculate how far the board_info_blob is from the beginning of the bootloader */
	uint32_t board_info_blob_offset = (uint32_t)&pios_board_info_blob - (uint32_t)0x08000000;

	/* Use the same offset into our embedded bootloader image */
	struct pios_board_info * new_board_info_blob = (struct pios_board_info *)
		((uintptr_t)&_bu_payload_start + board_info_blob_offset);

	/* Compare the two board info blobs to make sure they're for the same HW revision */
	if ((pios_board_info_blob.magic != new_board_info_blob->magic) ||
		(pios_board_info_blob.board_type != new_board_info_blob->board_type) ||
		(pios_board_info_blob.board_rev != new_board_info_blob->board_rev)) {
		error(PIOS_LED_HEARTBEAT);
	}

	/* Embedded bootloader looks like it's the right one for this HW, proceed... */

	uintptr_t bl_partition_id;
	if (PIOS_FLASH_find_partition_id(FLASH_PARTITION_LABEL_BL, &bl_partition_id) != 0)
		error(PIOS_LED_HEARTBEAT);

	/* Erase the partition */
	PIOS_LED_On(PIOS_LED_HEARTBEAT);
	PIOS_FLASH_start_transaction(bl_partition_id);
	PIOS_FLASH_erase_partition(bl_partition_id);
	PIOS_FLASH_end_transaction(bl_partition_id);
	PIOS_LED_Off(PIOS_LED_HEARTBEAT);

	PIOS_DELAY_WaitmS(500);

	/* Write in the new bootloader */
	PIOS_LED_On(PIOS_LED_HEARTBEAT);
	PIOS_FLASH_start_transaction(bl_partition_id);
	PIOS_FLASH_write_data(bl_partition_id, 0, (uint8_t *)&_bu_payload_start, _bu_payload_size);
	PIOS_FLASH_end_transaction(bl_partition_id);
	PIOS_LED_Off(PIOS_LED_HEARTBEAT);

	PIOS_DELAY_WaitmS(500);

	/* Invalidate the FW partition so that we don't run the updater again on next boot */
	uintptr_t fw_partition_id;
	if (PIOS_FLASH_find_partition_id(FLASH_PARTITION_LABEL_FW, &fw_partition_id) != 0)
		error(PIOS_LED_HEARTBEAT);

	PIOS_LED_On(PIOS_LED_HEARTBEAT);
	const uint32_t zero = 0;
	PIOS_FLASH_start_transaction(fw_partition_id);
	PIOS_FLASH_write_data(fw_partition_id, 0, (uint8_t *)&zero, sizeof(zero));
	PIOS_FLASH_end_transaction(fw_partition_id);
	PIOS_LED_Off(PIOS_LED_HEARTBEAT);

	PIOS_DELAY_WaitmS(1000);

	/* Flash the LED to indicate finished */
	for (uint8_t x = 0; x < 5; ++x) {
			PIOS_LED_On(PIOS_LED_HEARTBEAT);
			PIOS_DELAY_WaitmS(1000);
			PIOS_LED_Off(PIOS_LED_HEARTBEAT);
			PIOS_DELAY_WaitmS(1000);
	}

	while (1) {
		PIOS_DELAY_WaitmS(1000);
	}
}

void error(int led) {
	for (;;) {
		PIOS_LED_On(led);
		PIOS_DELAY_WaitmS(500);
		PIOS_LED_Off(led);
		PIOS_DELAY_WaitmS(500);
	}
}

/**
 * @}
 * @}
 */
