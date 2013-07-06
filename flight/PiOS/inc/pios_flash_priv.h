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

/**
 * Describes the API that must be implemented by each flash chip driver.
 */
struct pios_flash_driver {
	int32_t (*start_transaction)(uintptr_t chip_id);
	int32_t (*end_transaction)(uintptr_t chip_id);

	int32_t (*erase_sector)(uintptr_t chip_id, uint32_t chip_sector, uint32_t chip_offset);
	int32_t (*write_data)(uintptr_t chip_id, uint32_t chip_offset, const uint8_t *data, uint16_t len);
	int32_t (*read_data)(uintptr_t chip_id, uint32_t chip_offset, uint8_t *data, uint16_t len);
};

/**
 * Describes a block of sectors within a chip that all have the same sector size.
 */
struct pios_flash_sector_range {
	uint16_t base_sector;
	uint16_t last_sector;
	uint32_t sector_size;
};

/**
 * Describes all of the attributes of a single, physical flash device.
 *
 * driver provides all of the functions to allow access to the device.
 * chip_id points to the run-time context for this device after it has been initialized.
 * page_size is the largest unit that can be written in a single write request.
 *    These pages are aligned to the start of a sector and the entire write operation must
 *    be within a single page.
 * sector_blocks points to an array of blocks of sector ranges.
 *    This is effectively a run-length-encoded list of all sectors grouped by sector size
 * num_blocks is the number of elements in the sector_blocks array
 */
struct pios_flash_chip {
	const struct pios_flash_driver *driver;
	uintptr_t *chip_id;
	uint32_t page_size;
	const struct pios_flash_sector_range *sector_blocks;
	uint32_t num_blocks;
};

/**
 * A partition is a contiguous block of sectors within a single underlying flash device.
 *
 * label is used to describe what the partition is used for and to attach the proper filesystem
 * chip_desc refers to the underlying chip device descriptor
 * first_sector is the sector number (within the chip) where the partition begins
 * last_sector is the last sector number (within the chip) that is within the partition
 */
struct pios_flash_partition {
	enum pios_flash_partition_labels label;
	const struct pios_flash_chip *chip_desc;
	uint16_t first_sector;
	uint16_t last_sector;
	uint32_t chip_offset;
	uint32_t size;
};

#define FLASH_SECTOR_1KB   (  1 * 1024)
#define FLASH_SECTOR_2KB   (  2 * 1024)
#define FLASH_SECTOR_4KB   (  4 * 1024)
#define FLASH_SECTOR_8KB   (  8 * 1024)
#define FLASH_SECTOR_16KB  ( 16 * 1024)
#define FLASH_SECTOR_32KB  ( 32 * 1024)
#define FLASH_SECTOR_64KB  ( 64 * 1024)
#define FLASH_SECTOR_128KB (128 * 1024)

extern void PIOS_FLASH_register_partition_table(const struct pios_flash_partition partition_table[], uint8_t num_partitions);

#endif	/* PIOS_FLASH_PRIV_H_ */
