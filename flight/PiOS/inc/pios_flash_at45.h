/**
 ******************************************************************************
 * @file       pios_flash_at45.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_FLASH Flash Driver API Definition
 * @{
 * @brief Flash Driver API Definition
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

#ifndef PIOS_FLASH_AT45_H_
#define PIOS_FLASH_AT45_H_

#include <stdint.h>

struct pios_flash_chunk {
	uint8_t * addr;
	uint32_t len;
};

struct pios_flash_driver {
	int32_t (*start_transaction)(uintptr_t flash_id);
	int32_t (*end_transaction)(uintptr_t flash_id);
	int32_t (*erase_chip)(uintptr_t flash_id);
	int32_t (*erase_sector)(uintptr_t flash_id, uint32_t addr);
	int32_t (*write_data)(uintptr_t flash_id, uint16_t addr, uint8_t * data, uint16_t len);
	int32_t (*write_buffer)(uintptr_t flash_id, uint16_t addr,  uint8_t * data, uint16_t len);
	int32_t (*buffer_to_page)(uintptr_t flash_id, uint16_t addr);
	int32_t (*page_to_buffer)(uintptr_t flash_id, uint16_t addr);
	int32_t (*read_data)(uintptr_t flash_id, uint32_t addr, uint16_t offset, uint8_t * data, uint16_t len);
};

#endif	/* PIOS_FLASH_AT45_H_ */
