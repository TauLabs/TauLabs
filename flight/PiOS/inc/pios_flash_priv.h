/**
 ******************************************************************************
 * @file       pios_flash_priv.h
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

#ifndef PIOS_FLASH_PRIV_H_
#define PIOS_FLASH_PRIV_H_

#include <stdint.h>
#include "pios_flash.h"

struct pios_flash_driver {
	int32_t (*start_transaction)(uintptr_t chip_id);
	int32_t (*end_transaction)(uintptr_t chip_id);

	int32_t (*erase_sector)(uintptr_t chip_id, uint32_t chip_sector, uint32_t chip_offset);
	int32_t (*write_data)(uintptr_t chip_id, uint32_t chip_offset, const uint8_t * data, uint16_t len);
	int32_t (*read_data)(uintptr_t chip_id, uint32_t chip_offset, uint8_t * data, uint16_t len);
};

struct pios_flash_sector_range {
	uint16_t base_sector;
	uint16_t last_sector;
	uint32_t sector_size;
};

struct pios_flash_chip {
	const struct pios_flash_driver * driver;
	uintptr_t * chip_id;
	uint32_t page_size;
	const struct pios_flash_sector_range * sector_blocks;
	uint32_t num_blocks;
};

struct pios_flash_partition {
	enum pios_flash_partition_labels label;
	const struct pios_flash_chip * chip_desc;
	uint16_t first_sector;
	uint16_t last_sector;
};

extern void PIOS_FLASH_register_partition_table(const struct pios_flash_partition partition_table[], uint8_t num_partitions);

#endif	/* PIOS_FLASH_PRIV_H_ */
