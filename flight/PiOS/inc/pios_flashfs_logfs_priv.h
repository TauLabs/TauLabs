/**
 ******************************************************************************
 * @file       pios_flashfs_logfs_priv.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_FLASHFS Flash Filesystem Function
 * @{
 * @brief Log Structured Filesystem for internal or external NOR Flash
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

#ifndef PIOS_FLASHFS_LOGFS_PRIV_H_
#define PIOS_FLASHFS_LOGFS_PRIV_H_

#include <stdint.h>
#include "pios_flash.h"		/* struct pios_flash_driver */

/**
 * Configuration for a logfs filesystem
 *
 * Note: a filesystem requires room for at least 2 arenas within its partition.
 * Note: a filesystem requires room for at least 2 slots per arena.  The first slot is reserved.
 */
struct flashfs_logfs_cfg {
	uint32_t fs_magic;
	uint32_t arena_size;	/* Max size of one generation of the filesystem */
	uint32_t slot_size;	/* Max size of a "file" within the filesystem */
};

int32_t PIOS_FLASHFS_Logfs_Init(uintptr_t * fs_id, const struct flashfs_logfs_cfg * cfg, enum pios_flash_partition_labels partition_label);

int32_t PIOS_FLASHFS_Logfs_Destroy(uintptr_t fs_id);

#endif	/* PIOS_FLASHFS_LOGFS_PRIV_H_ */
