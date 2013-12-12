/**
 ******************************************************************************
 * @file       pios_flash.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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

#ifndef PIOS_FLASH_H_
#define PIOS_FLASH_H_

#include <stdint.h>

enum pios_flash_partition_labels {
	FLASH_PARTITION_LABEL_BL,
	FLASH_PARTITION_LABEL_FW,
	FLASH_PARTITION_LABEL_EE,
	FLASH_PARTITION_LABEL_SETTINGS,
	FLASH_PARTITION_LABEL_WAYPOINTS,
	FLASH_PARTITION_LABEL_LOG,
	FLASH_PARTITION_LABEL_OTA,

	FLASH_PARTITION_NUM_LABELS, /* Must be last */
};

extern int32_t PIOS_FLASH_find_partition_id(enum pios_flash_partition_labels label, uintptr_t *partition_id);
extern uint16_t PIOS_FLASH_get_num_partitions(void);
extern int32_t PIOS_FLASH_get_partition_size(uintptr_t partition_id, uint32_t *partition_size);

extern int32_t PIOS_FLASH_start_transaction(uintptr_t partition_id);
extern int32_t PIOS_FLASH_end_transaction(uintptr_t partition_id);
extern int32_t PIOS_FLASH_erase_partition(uintptr_t partition_id);
extern int32_t PIOS_FLASH_erase_range(uintptr_t partition_id, uint32_t start_offset, uint32_t size);
extern int32_t PIOS_FLASH_write_data(uintptr_t partition_id, uint32_t offset, const uint8_t *data, uint16_t len);
extern int32_t PIOS_FLASH_read_data(uintptr_t partition_id, uint32_t offset, uint8_t *data, uint16_t len);

#endif	/* PIOS_FLASH_H_ */
