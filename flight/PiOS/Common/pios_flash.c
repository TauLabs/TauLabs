/**
 ******************************************************************************
 * @file       pios_flash.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_FLASH Flash Partition Abstraction
 * @{
 * @brief Flash Partition Abstraction to hide details of underlying flash device details
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

/* Project Includes */
#include "pios_config.h"

#if defined(PIOS_INCLUDE_FLASH)

#include "pios_flash_priv.h"	/* External API definition */

#include <stdbool.h>		/* bool */
#include <stdlib.h>		/* NULL */

static struct pios_flash_partition const *partitions;
static uint8_t num_partitions;

#define PIOS_Assert(x) do { } while (!(x))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

void PIOS_FLASH_register_partition_table(const struct pios_flash_partition partition_table[], uint8_t partition_table_len)
{
	/* TODO: could check integrity of the table being passed in.  maybe a waste of code space. */

	PIOS_Assert(partition_table);

	for (uint8_t i = 0; i < num_partitions; i++) {
		const struct pios_flash_partition *partition = &partition_table[i];
		PIOS_Assert(partition->label < FLASH_PARTITION_NUM_LABELS);
		PIOS_Assert(partition->chip_desc);
		PIOS_Assert(partition->chip_desc->driver);
		PIOS_Assert(partition->chip_desc->chip_id);
		PIOS_Assert(partition->chip_desc->sector_blocks);
		PIOS_Assert(partition->chip_desc->num_blocks > 0);
		PIOS_Assert(partition->last_sector >= partition->first_sector);
	}

	partitions     = partition_table;
	num_partitions = partition_table_len;
}

/* Lookup flash partition_id from a partition label (BL, FW, SETTINGS, etc.) */
int32_t PIOS_FLASH_find_partition_id(enum pios_flash_partition_labels label, uintptr_t *partition_id)
{
	PIOS_Assert(partition_id);
	
	for (uint8_t i = 0; i < num_partitions; i++) {
		if (partitions[i].label == label) {
			*partition_id = (uintptr_t) &partitions[i];
			return 0;
		}
	}

	return -1;
	
}

uint16_t PIOS_FLASH_get_num_partitions(void)
{
	return num_partitions;
}

static bool PIOS_FLASH_validate_partition(const struct pios_flash_partition *partition)
{
	return ((partition >= &partitions[0]) && (partition <= &partitions[num_partitions]));
}

struct pios_flash_sector_desc {
	/* Context */
	uint16_t block_id;
	uint32_t sector;

	/* User information */
	uint32_t chip_offset;
	uint32_t partition_offset;
	uint32_t sector_size;
};

static bool pios_flash_get_partition_first_sector(const struct pios_flash_partition *partition, struct pios_flash_sector_desc *curr)
{
	if (!PIOS_FLASH_validate_partition(partition))
		return false;

	/* Find the beginning of the partition */
	uint32_t chip_offset = 0;
	for (uint16_t block_id = 0; block_id < partition->chip_desc->num_blocks; block_id++) {
		const struct pios_flash_sector_range *block = &partition->chip_desc->sector_blocks[block_id];

		if ((partition->first_sector >= block->base_sector) && (partition->first_sector <= block->last_sector)) {
			/* Sector is in this block.  Compute offset within this block */
			chip_offset += (partition->first_sector - block->base_sector) *block->sector_size;

			curr->block_id         = block_id;
			curr->sector           = partition->first_sector;

			curr->chip_offset      = chip_offset;
			curr->sector_size      = block->sector_size;
			curr->partition_offset = 0;

			return true;
		} else {
			/* Not this block.  Skip forward to the next block. */
			uint32_t num_sectors_in_range = (block->last_sector - block->base_sector + 1);
			chip_offset += num_sectors_in_range *block->sector_size;
		}
	}

	return false;
}

static bool pios_flash_get_partition_next_sector(const struct pios_flash_partition *partition, struct pios_flash_sector_desc *curr)
{
	if (!PIOS_FLASH_validate_partition(partition))
		return false;

	/* Are we past the end of the device? */
	if (curr->block_id >= partition->chip_desc->num_blocks)
		return false;

	const struct pios_flash_sector_range *block = &partition->chip_desc->sector_blocks[curr->block_id];

	/* Is the current sector within the current block? */
	if ((curr->sector < block->base_sector) || (curr->sector > block->last_sector))
		return false;

	/* Is the current sector within the current partition? */
	if ((curr->sector < partition->first_sector) || (curr->sector > partition->last_sector))
		return false;

	/*
	 * We've been given a valid context, find the next sector in the partition
	 */

	/* Accumulate the size of the current sector into the offsets */
	curr->chip_offset      += block->sector_size;
	curr->partition_offset += block->sector_size;

	curr->sector++;

	/* Are we still in the partition? */
	if (curr->sector > partition->last_sector)
		return false;

	/* Are we now beyond the end of the current block? */
	if (curr->sector > block->last_sector) {
		/* Move to the next block */
		curr->block_id++;

		/* Are we still within the chip boundaries? */
		if (curr->block_id >= partition->chip_desc->num_blocks)
			return false;

		block = &partition->chip_desc->sector_blocks[curr->block_id];
	}

	curr->sector_size = block->sector_size;

	return true;
}

static bool pios_flash_partition_get_chip_extents(const struct pios_flash_partition *partition, uint32_t *chip_start_offset, uint32_t *chip_end_offset)
{
	if (!PIOS_FLASH_validate_partition(partition))
		return false;

	struct pios_flash_sector_desc sector_desc;
	if (!pios_flash_get_partition_first_sector(partition, &sector_desc))
		return false;

	if (chip_start_offset)
		*chip_start_offset = sector_desc.chip_offset;

	/* Traverse the current partition to find the end of it */
	do {
		;
	} while (pios_flash_get_partition_next_sector(partition, &sector_desc));

	if (chip_end_offset)
		*chip_end_offset = sector_desc.chip_offset - 1;

	return true;
}

int32_t PIOS_FLASH_get_partition_size(uintptr_t partition_id, uint32_t *partition_size)
{
	PIOS_Assert(partition_size);

	struct pios_flash_partition *partition = (struct pios_flash_partition *)partition_id;

	if (!PIOS_FLASH_validate_partition(partition))
		return -20;

	uint32_t chip_start_offset;
	uint32_t chip_end_offset;
	if (!pios_flash_partition_get_chip_extents(partition, &chip_start_offset, &chip_end_offset))
		return -21;

	*partition_size = (chip_end_offset + 1) - chip_start_offset;

	return 0;
}

int32_t PIOS_FLASH_start_transaction(uintptr_t partition_id)
{
	struct pios_flash_partition *partition = (struct pios_flash_partition *)partition_id;

	if (!PIOS_FLASH_validate_partition(partition))
		return -20;

	if (partition->chip_desc->driver->start_transaction)
		return partition->chip_desc->driver->start_transaction(*partition->chip_desc->chip_id);

	/* If the underlying driver doesn't support this, just return OK */
	return 0;
}

int32_t PIOS_FLASH_end_transaction(uintptr_t partition_id)
{
	struct pios_flash_partition *partition = (struct pios_flash_partition *)partition_id;

	if (!PIOS_FLASH_validate_partition(partition))
		return -20;

	if (partition->chip_desc->driver->end_transaction)
		return partition->chip_desc->driver->end_transaction(*partition->chip_desc->chip_id);

	/* If the underlying driver doesn't support this, just return OK */
	return 0;
}

int32_t PIOS_FLASH_erase_range(uintptr_t partition_id, uint32_t start_offset, uint32_t size)
{
	struct pios_flash_partition *partition = (struct pios_flash_partition *)partition_id;

	if (!PIOS_FLASH_validate_partition(partition))
		return -20;

	if (!partition->chip_desc->driver->erase_sector)
		return -21;

	struct pios_flash_sector_desc sector_desc;
	if (!pios_flash_get_partition_first_sector(partition, &sector_desc))
		return -22;

	/* Traverse the current partition and erase sectors within the requested range */
	uint32_t erase_offset = start_offset;
	do {
		/* Have we finished erasing? */
		if (size == 0)
			break;

		/* Is our start offset in the current sector? */
		if ((erase_offset >= sector_desc.partition_offset) &&
		        (erase_offset < sector_desc.partition_offset + sector_desc.sector_size)) {
			/* Make sure we're erasing on a sector boundary */
			if (erase_offset != sector_desc.partition_offset)
				return -23;

			/* Make sure we're intending to erase the entire sector */
			if (size < sector_desc.sector_size)
				return -24;

			/* Erase the sector */
			if (partition->chip_desc->driver->erase_sector(*partition->chip_desc->chip_id,
									sector_desc.sector,
									sector_desc.chip_offset) != 0) {
				return -25;
			}

			/* Update our accounting */
			erase_offset += sector_desc.sector_size;
			size         -= sector_desc.sector_size;
		}
	} while (pios_flash_get_partition_next_sector(partition, &sector_desc));

	return 0;
}

int32_t PIOS_FLASH_erase_partition(uintptr_t partition_id)
{
	struct pios_flash_partition *partition = (struct pios_flash_partition *)partition_id;

	if (!PIOS_FLASH_validate_partition(partition))
		return -20;

	if (!partition->chip_desc->driver->erase_sector)
		return -21;

	struct pios_flash_sector_desc sector_desc;
	if (!pios_flash_get_partition_first_sector(partition, &sector_desc))
		return -22;

	/* Traverse the current partition and erase sectors within the requested range */
	do {
		/* Erase the sector */
		if (partition->chip_desc->driver->erase_sector(*partition->chip_desc->chip_id,
								sector_desc.sector,
								sector_desc.chip_offset) != 0) {
			return -23;
		}
	} while (pios_flash_get_partition_next_sector(partition, &sector_desc));

	return 0;
}

int32_t PIOS_FLASH_write_data(uintptr_t partition_id, uint32_t partition_offset, const uint8_t *data, uint16_t len)
{
	struct pios_flash_partition *partition = (struct pios_flash_partition *)partition_id;

	if (!PIOS_FLASH_validate_partition(partition))
		return -20;

	if (!partition->chip_desc->driver->write_data)
		return -21;

	/* Find the start and end of the partition within the chip's address space */
	uint32_t chip_start_of_partition;
	uint32_t chip_end_of_partition;
	if (!pios_flash_partition_get_chip_extents(partition,
							&chip_start_of_partition,
							&chip_end_of_partition))
		return -22;

	/* Are we writing past the end of the partition? */
	if ((partition_offset + len) > ((chip_end_of_partition + 1) - chip_start_of_partition))
		return -23;

	/* Fragment the writes to chip's page size */
	uint32_t page_size = partition->chip_desc->page_size;
	while (len > 0) {
		/* Individual writes must fit entirely within a single page buffer. */
		uint32_t page_remaining = page_size - (partition_offset % page_size);
		uint16_t write_size = MIN(len, page_remaining);

		int32_t rc = partition->chip_desc->driver->write_data(*partition->chip_desc->chip_id,
								chip_start_of_partition + partition_offset,
								data,
								write_size);
		if (rc != 0) {
			/* Failed to write the data to the underlying flash */
			return rc;
		}

		/* Update our accounting */
		data             += write_size;
		partition_offset += write_size;
		len              -= write_size;
	}

	return 0;
}

int32_t PIOS_FLASH_read_data(uintptr_t partition_id, uint32_t partition_offset, uint8_t *data, uint16_t len)
{
	struct pios_flash_partition *partition = (struct pios_flash_partition *)partition_id;

	if (!PIOS_FLASH_validate_partition(partition))
		return -20;

	if (!partition->chip_desc->driver->read_data)
		return -21;

	/* Find the start and end of the partition within the chip's address space */
	uint32_t chip_start_of_partition;
	uint32_t chip_end_of_partition;
	if (!pios_flash_partition_get_chip_extents(partition,
							&chip_start_of_partition,
							&chip_end_of_partition))
		return -22;

	/* Are we reading past the end of the partition? */
	if ((partition_offset + len) > ((chip_end_of_partition + 1) - chip_start_of_partition))
		return -23;

	return partition->chip_desc->driver->read_data(*partition->chip_desc->chip_id,
						chip_start_of_partition + partition_offset,
						data,
						len);
}

#endif	/* PIOS_INCLUDE_FLASH */

/**
 * @}
 * @}
 */
