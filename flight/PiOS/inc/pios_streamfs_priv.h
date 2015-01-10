/**
 ******************************************************************************
 * @file       pios_streamfs_priv.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_FLASHFS Flash Filesystem Function
 * @{
 * @brief Streaming circular buffer for external NOR Flash
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

#ifndef PIOS_FLASHFS_STREAMFS_PRIV_H_
#define PIOS_FLASHFS_STREAMFS_PRIV_H_

#include <stdint.h>
#include "pios_flash.h"

/**
 * Configuration for a streamfs filesystem
 */
struct streamfs_cfg {
	uint32_t fs_magic;
	uint32_t arena_size; /* The size chunk that is erased (must equal sector size) */
	uint32_t write_size;  /* The size to buffer between writes */
};

int32_t PIOS_STREAMFS_Init(uintptr_t *fs_id, const struct streamfs_cfg *cfg, enum pios_flash_partition_labels partition_label);

extern const struct pios_com_driver pios_streamfs_com_driver;

#endif	/* PIOS_FLASHFS_STREAMFS_PRIV_H_ */
