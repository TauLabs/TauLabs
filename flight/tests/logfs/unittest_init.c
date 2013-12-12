/* 
 * These need to be defined in a .c file so that we can use
 * designated initializer syntax which c++ doesn't support (yet).
 */

#define NELEMENTS(x) (sizeof(x) / sizeof(*(x)))

#include "pios_flashfs_logfs_priv.h"

const struct flashfs_logfs_cfg flashfs_config_settings = {
	.fs_magic      = 0x89abceef,
	.arena_size    = 0x00010000, /* 256 * slot size */
	.slot_size     = 0x00000100, /* 256 bytes */
};

const struct flashfs_logfs_cfg flashfs_config_waypoints = {
	.fs_magic      = 0x89abceef,
	.arena_size    = 0x00010000, /* 64 * slot size */
	.slot_size     = 0x00000400, /* 256 bytes */
};

#include "pios_flash_posix_priv.h"

#include "pios_flash_priv.h"

const struct pios_flash_posix_cfg flash_config = {
	.size_of_flash  = 3 * 1024 * 1024,
	.size_of_sector = FLASH_SECTOR_64KB,
};

static const struct pios_flash_sector_range posix_flash_sectors[] = {
	{
		.base_sector = 0,
		.last_sector = 47,
		.sector_size = FLASH_SECTOR_64KB,
	},
};

uintptr_t pios_posix_flash_id;
static const struct pios_flash_chip pios_flash_chip_posix = {
	.driver        = &pios_posix_flash_driver,
	.chip_id       = &pios_posix_flash_id,
	.page_size     = 256,
	.sector_blocks = posix_flash_sectors,
	.num_blocks    = NELEMENTS(posix_flash_sectors),
};

const struct pios_flash_partition pios_flash_partition_table[] = {
	{
		.label        = FLASH_PARTITION_LABEL_SETTINGS,
		.chip_desc    = &pios_flash_chip_posix,
		.first_sector = 0,
		.last_sector  = 31,
		.chip_offset  = 0,
		.size         = (31 - 0 + 1) * FLASH_SECTOR_64KB,
	},

	{
		.label        = FLASH_PARTITION_LABEL_WAYPOINTS,
		.chip_desc    = &pios_flash_chip_posix,
		.first_sector = 32,
		.last_sector  = 47,
		.chip_offset  = (32 * 64 * 1024),
		.size         = (47 - 32 + 1) * FLASH_SECTOR_64KB,
	},
};

uint32_t pios_flash_partition_table_size = NELEMENTS(pios_flash_partition_table);
