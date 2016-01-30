/**
 ******************************************************************************
 * @file       pios_flashfs_streamfs.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014-2016
 * @author     dRonin, http://dronin.org Copyright (C) 2015
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

/* Project Includes */
#include "pios.h"

#include "pios_flash.h"		     /* PIOS_FLASH_* */
#include "pios_streamfs_priv.h" /* Internal API */

#include <stdbool.h>
#include <stddef.h>		/* NULL */

#define MIN(x,y) ((x) < (y) ? (x) : (y))

/**
 * @Note
 * This file system provides the ability to create numbered files
 * and stream to flash. These buffers will wrap around the available
 * flash partition and the file system makes no attempt to prevent
 * that happening. It is intended for applications of storing
 * large files or logging information
 *
 * Files are written into continuous sectors of the flash chip. Each
 * sector has a footer to indicate the file id and the sector id.
 *
 * Arenas map onto sectors. 
 */

#include <pios_com.h>

/* Provide a COM driver */
static void PIOS_STREAMFS_RegisterRxCallback(uintptr_t fs_id, pios_com_callback rx_in_cb, uintptr_t context);
static void PIOS_STREAMFS_RegisterTxCallback(uintptr_t fs_id, pios_com_callback tx_out_cb, uintptr_t context);
static void PIOS_STREAMFS_TxStart(uintptr_t fs_id, uint16_t tx_bytes_avail);
static void PIOS_STREAMFS_RxStart(uintptr_t fs_id, uint16_t rx_bytes_avail);

const struct pios_com_driver pios_streamfs_com_driver = {
	.tx_start   = PIOS_STREAMFS_TxStart,
	.rx_start   = PIOS_STREAMFS_RxStart,
	.bind_tx_cb = PIOS_STREAMFS_RegisterTxCallback,
	.bind_rx_cb = PIOS_STREAMFS_RegisterRxCallback,
};

/*
 * Filesystem state data tracked in RAM
 */

enum pios_flashfs_streamfs_dev_magic {
	PIOS_FLASHFS_STREAMFS_DEV_MAGIC = 0x93A40F82,
};

struct streamfs_state {
	enum pios_flashfs_streamfs_dev_magic magic;
	const struct streamfs_cfg *cfg;


	/* pios_com interface */
	pios_com_callback rx_in_cb;
	uintptr_t rx_in_context;
	pios_com_callback tx_out_cb;
	uintptr_t tx_out_context;
	uint8_t *com_buffer;

	/* Information for current file handle */
	bool file_open_writing;
	bool file_open_reading;
	int32_t active_file_id;
	int32_t active_file_segment;
	int32_t active_file_arena;
	int32_t active_file_arena_offset;

	/* Information about file system contents */
	int32_t min_file_id;
	int32_t max_file_id;

	/* Underlying flash partition handle */
	uintptr_t partition_id;
	uint32_t partition_size;
	uint32_t partition_arenas;
};

/*
 * Internal Utility functions
 */

/**
 * @brief Return the offset in flash of a particular slot within an arena
 * @return address of the requested slot
 */
static uintptr_t streamfs_get_addr(const struct streamfs_state *streamfs, uint32_t arena_id, uint16_t arena_offset)
{
	PIOS_Assert(arena_id < (streamfs->partition_size / streamfs->cfg->arena_size));
	PIOS_Assert(arena_offset < streamfs->cfg->arena_size);

	return (arena_id * streamfs->cfg->arena_size) + arena_offset;
}

struct streamfs_footer {
	uint32_t magic;
	uint32_t written_bytes;
	uint32_t file_id;
	uint16_t file_segment;
} __attribute__((packed));


/****************************************
 * Arena life-cycle transition functions
 ****************************************/

/**
 * @brief Erases all sectors within the given arena and sets arena to erased state.
 * @return 0 if success, < 0 on failure
 * @note Must be called while holding the flash transaction lock
 */
static int32_t streamfs_erase_arena(const struct streamfs_state *streamfs, uint32_t arena_id)
{
	uintptr_t arena_addr = streamfs_get_addr(streamfs, arena_id, 0);

	/* Erase all of the sectors in the arena */
	if (PIOS_FLASH_erase_range(streamfs->partition_id, arena_addr, streamfs->cfg->arena_size) != 0) {
		return -1;
	}

	/* Arena is ready to be written to */
	return 0;
}

/**
 * @brief Erases all arenas available to this filesystem instance
 * @return 0 if success, < 0 on failure
 * @note Must be called while holding the flash transaction lock
 */
static int32_t streamfs_erase_all_arenas(const struct streamfs_state *streamfs)
{
	uint32_t num_arenas = streamfs->partition_size / streamfs->cfg->arena_size;

	for (uint32_t arena = 0; arena < num_arenas; arena++) {
		if (streamfs_erase_arena(streamfs, arena) != 0)
			return -1;
	}

	return 0;
}

static bool streamfs_validate(const struct streamfs_state *streamfs)
{
	return (streamfs && (streamfs->magic == PIOS_FLASHFS_STREAMFS_DEV_MAGIC));
}

static struct streamfs_state *streamfs_alloc(void)
{
	struct streamfs_state *streamfs;

	streamfs = (struct streamfs_state *)PIOS_malloc_no_dma(sizeof(*streamfs));
	if (!streamfs) return (NULL);

	streamfs->magic = PIOS_FLASHFS_STREAMFS_DEV_MAGIC;
	return(streamfs);
}

static void streamfs_free(struct streamfs_state *streamfs)
{
	/* Invalidate the magic */
	streamfs->magic = ~PIOS_FLASHFS_STREAMFS_DEV_MAGIC;
	PIOS_free(streamfs);
}

/**
 * Write footer to current sector and reset pointers for writing to
 * next sector
 */
/* NOTE: Must be called while holding the flash transaction lock */ 
static int32_t streamfs_new_sector(struct streamfs_state *streamfs)
{
	struct streamfs_footer footer;
	footer.magic = streamfs->cfg->fs_magic;
	footer.written_bytes = streamfs->active_file_arena_offset;
	footer.file_id = streamfs->active_file_id;
	footer.file_segment = streamfs->active_file_segment;

	uint32_t start_address = streamfs_get_addr(streamfs, streamfs->active_file_arena,
			                                   streamfs->cfg->arena_size - sizeof(footer));

	if (PIOS_FLASH_write_data(streamfs->partition_id, start_address, (uint8_t *) &footer, sizeof(footer)) != 0) {
		return -1;
	}

	// Reset pointers for writing to next sector
	streamfs->active_file_arena = (streamfs->active_file_arena + 1) % streamfs->partition_arenas;
	streamfs->active_file_arena_offset = 0;
	streamfs->active_file_segment++;

	// Test whether the sector has already been erased by checking the footer
	start_address = streamfs_get_addr(streamfs, streamfs->active_file_arena,
			                          streamfs->cfg->arena_size - sizeof(footer));
	if (PIOS_FLASH_read_data(streamfs->partition_id, start_address, (uint8_t *) &footer, sizeof(footer)) != 0) {
		return -2;
	}

	for (int i=0; i < sizeof(footer); i++) {
		if (((uint8_t*)&footer)[i] != 0xFF) {
			if (streamfs_erase_arena(streamfs, streamfs->active_file_arena) != 0) {
				return -3;
			}
			break;
		}
	}

	return 0;
}

/**
 * Close this sector by writing footer. Does not prepare next sector.
 */
/* NOTE: Must be called while holding the flash transaction lock */
static int32_t streamfs_close_sector(struct streamfs_state *streamfs)
{
	struct streamfs_footer footer;
	footer.magic = streamfs->cfg->fs_magic;
	footer.written_bytes = streamfs->active_file_arena_offset;
	footer.file_id = streamfs->active_file_id;
	footer.file_segment = streamfs->active_file_segment;

	uint32_t start_address = streamfs_get_addr(streamfs, streamfs->active_file_arena,
			                                   streamfs->cfg->arena_size - sizeof(footer));

	if (PIOS_FLASH_write_data(streamfs->partition_id, start_address, (uint8_t *) &footer, sizeof(footer)) != 0) {
		return -1;
	}

	return 0;
}


/**
 * Find the first arena for a file
 * @param[in] streamfs the file system handle
 * @param[in] file_id the file to find
 * @return the sector number if found, or negative if there was an error
 *
 * @NOTE: Must be called while holding the flash transaction lock
 */
static int32_t streamfs_find_first_arena(struct streamfs_state *streamfs, int32_t file_id)
{
	uint16_t num_arenas = streamfs->partition_size / streamfs->cfg->arena_size;

	bool found_file = false;
	uint32_t min_segment = 0xFFFFFFFF;
	uint32_t sector = 0xFFFFFFFF;

	for (uint16_t arena = 0; arena < num_arenas; arena++) {
		// Read footer for each arena
		struct streamfs_footer footer;
		uint32_t start_address = streamfs_get_addr(streamfs, arena,
				                                   streamfs->cfg->arena_size - sizeof(footer));
		if (PIOS_FLASH_read_data(streamfs->partition_id, start_address, (uint8_t *) &footer, sizeof(footer)) != 0) {
			return -1;
		}

		if (footer.magic == streamfs->cfg->fs_magic && footer.file_id == file_id) {
			found_file = true;
			if (footer.file_segment < min_segment) {
				min_segment = footer.file_segment;
				sector = arena;
			}
		}
	}

	if (found_file) {
		return sector;
	}

	return -2;
}

/**
 * Find the last arena for a file
 * @param[in] streamfs the file system handle
 * @param[in] file_id the file to find
 * @return the sector number if found, or negative if there was an error
 *
 * @NOTE: Must be called while holding the flash transaction lock
 */
static int32_t streamfs_find_last_arena(struct streamfs_state *streamfs, int32_t file_id)
{
	uint16_t num_arenas = streamfs->partition_size / streamfs->cfg->arena_size;

	bool found_file = false;
	int32_t max_segment = -1;
	uint32_t sector = 0;

	for (uint16_t arena = 0; arena < num_arenas; arena++) {
		// Read footer for each arena
		struct streamfs_footer footer;
		uint32_t start_address = streamfs_get_addr(streamfs, arena,
				                                   streamfs->cfg->arena_size - sizeof(footer));
		if (PIOS_FLASH_read_data(streamfs->partition_id, start_address, (uint8_t *) &footer, sizeof(footer)) != 0) {
			return -3;
		}

		if (footer.magic == streamfs->cfg->fs_magic && footer.file_id == file_id) {
			found_file = true;
			if (footer.file_segment > max_segment) {
				max_segment = footer.file_segment;
				sector = arena;
			}
		}
	}

	if (found_file) {
		return sector;
	}

	return -4;
}

/**
 * Find the first sector for a file
 * @param[in] streamfs the file system handle
 * @return the sector for the new file
 *
 * @NOTE: Must be called while holding the flash transaction lock
 */
static int32_t streamfs_find_new_sector(struct streamfs_state *streamfs)
{
	// No files on file system
	if (streamfs->max_file_id < 0) {
		return 0;
	}

	int32_t last_sector = streamfs_find_last_arena(streamfs, streamfs->max_file_id);
	PIOS_Assert(last_sector >= 0);

	uint16_t num_arenas = streamfs->partition_size / streamfs->cfg->arena_size;
	return (last_sector + 1) % num_arenas;
}

/* NOTE: Must be called while holding the flash transaction lock */
static int32_t streamfs_append_to_file(struct streamfs_state *streamfs, uint8_t *data, uint32_t len)
{
	if (!streamfs->file_open_writing)
		return -1;

	if (streamfs->file_open_reading)
		return -2;

	uint32_t total_written = 0;

	while (len > 0) {
		uint32_t start_address = streamfs_get_addr(streamfs, streamfs->active_file_arena,
			                                                 streamfs->active_file_arena_offset);

		// Make sure not to write into the space for the footer
		uint32_t bytes_to_write = len;
		if ((streamfs->active_file_arena_offset + bytes_to_write) > (streamfs->cfg->arena_size - sizeof(struct streamfs_footer))) {
			bytes_to_write = streamfs->cfg->arena_size - sizeof(struct streamfs_footer) - streamfs->active_file_arena_offset;
		}

		if (PIOS_FLASH_write_data(streamfs->partition_id, start_address, data, bytes_to_write) != 0) {
			return -3;
		}

		// Increment pointers
		streamfs->active_file_arena_offset += bytes_to_write;
		len -= bytes_to_write;
		total_written += bytes_to_write;
		data = &data[bytes_to_write];


		if (streamfs->active_file_arena_offset >= (streamfs->cfg->arena_size - sizeof(struct streamfs_footer))) {
			if (streamfs_new_sector(streamfs) != 0) {
				return -4;
			}
		}
	}

	return total_written;
}

/* NOTE: Must be called while holding the flash transaction lock */
static int32_t streamfs_read_from_file(struct streamfs_state *streamfs, uint8_t *data, uint32_t len)
{
	if (streamfs->file_open_writing)
		return -1;

	if (!streamfs->file_open_reading)
		return -2;

	uint32_t total_read_len = 0;
	int32_t current_segment = 0;
	while (len > 0) {
		struct streamfs_footer footer;
		uint32_t start_address = streamfs_get_addr(streamfs, streamfs->active_file_arena,
				                                   streamfs->cfg->arena_size - sizeof(footer));

		PIOS_Assert(start_address < streamfs->partition_size);
		if (PIOS_FLASH_read_data(streamfs->partition_id, start_address, (uint8_t *) &footer, sizeof(footer)) != 0) {
			return -3;
		}

		// Return error if at the end of the file
		if (footer.magic != streamfs->cfg->fs_magic && footer.file_id != streamfs->active_file_id) {
			return total_read_len;
		}

		// Detected wrap around of file
		if (footer.file_segment < current_segment) {
			return total_read_len;
		}

		// End of file
		if (streamfs->active_file_arena_offset == footer.written_bytes) {
			return total_read_len;
		}

		start_address = streamfs_get_addr(streamfs, streamfs->active_file_arena,
			                                        streamfs->active_file_arena_offset);

		// Read either remaining bytes or until the footer
		int32_t bytes_to_read = len;
		if ((streamfs->active_file_arena_offset + bytes_to_read) > (streamfs->cfg->arena_size - sizeof(struct streamfs_footer))) {
			bytes_to_read = streamfs->cfg->arena_size - sizeof(struct streamfs_footer) - streamfs->active_file_arena_offset;
		}

		// Do not read more than valid bytes
		if ((streamfs->active_file_arena_offset + bytes_to_read) > footer.written_bytes) {
			bytes_to_read = footer.written_bytes - streamfs->active_file_arena_offset;
		}

		if (PIOS_FLASH_read_data(streamfs->partition_id, start_address, data, bytes_to_read) != 0) {
			return -3;
		}

		// Increment pointers
		len -= bytes_to_read;
		total_read_len += bytes_to_read;
		data = &data[bytes_to_read];

		streamfs->active_file_arena_offset += bytes_to_read;
		PIOS_Assert(streamfs->active_file_arena_offset <= (streamfs->cfg->arena_size - sizeof(struct streamfs_footer)));
		if (streamfs->active_file_arena_offset == streamfs->cfg->arena_size - sizeof(struct streamfs_footer)) {
			uint16_t num_arenas = streamfs->partition_size / streamfs->cfg->arena_size;
			streamfs->active_file_arena = (streamfs->active_file_arena + 1) % num_arenas;
			streamfs->active_file_arena_offset = 0;
		}
	}

	return total_read_len;
}

/* NOTE: Must be called while holding the flash transaction lock */
static int32_t streamfs_scan_filesystem(struct streamfs_state *streamfs)
{
	// Don't try and read while actively writing
	if (streamfs->file_open_writing)
		return -1;
	if (streamfs->file_open_reading)
		return -2;

	uint16_t num_arenas = streamfs->partition_size / streamfs->cfg->arena_size;
	streamfs->min_file_id = -1;
	streamfs->max_file_id = 0;

	bool found_file = false;

	for (uint16_t arena = 0; arena < num_arenas; arena++) {
		// Read footer for each arena
		struct streamfs_footer footer;
		uint32_t start_address = streamfs_get_addr(streamfs, arena,
				                                   streamfs->cfg->arena_size - sizeof(footer));
		if (PIOS_FLASH_read_data(streamfs->partition_id, start_address, (uint8_t *) &footer, sizeof(footer)) != 0) {
			return -3;
		}

		if (footer.magic == streamfs->cfg->fs_magic) {
			found_file = true;
			if (footer.file_id < streamfs->min_file_id)
				streamfs->min_file_id = footer.file_id;
			if (footer.file_id > streamfs->max_file_id)
				streamfs->max_file_id = footer.file_id;
		}
	}

	if (!found_file) {
		streamfs->min_file_id = -1;
		streamfs->max_file_id = -1;
	}

	return 0;
}
/**********************************
 *
 * Public API
 *
 *********************************/


/**
 * @brief Initialize the flash object setting FS
 * @return 0 if success, -1 if failure
 */
int32_t PIOS_STREAMFS_Init(uintptr_t *fs_id, const struct streamfs_cfg *cfg, enum pios_flash_partition_labels partition_label)
{
	PIOS_Assert(cfg);

	/* Find the partition id for the requested partition label */
	uintptr_t partition_id;
	if (PIOS_FLASH_find_partition_id(partition_label, &partition_id) != 0) {
		return -1;
	}

	/* Query the total partition size */
	uint32_t partition_size;
	if (PIOS_FLASH_get_partition_size(partition_id, &partition_size) != 0) {
		return -1;
	}

	/* sector_size must exactly divide the partition size */
	PIOS_Assert((partition_size % cfg->arena_size) == 0);

	/* sector_size must exceed write_size */
	PIOS_Assert(cfg->arena_size > cfg->write_size);

	int8_t rc;

	struct streamfs_state *streamfs;

	streamfs = (struct streamfs_state *) streamfs_alloc();
	if (!streamfs) {
		rc = -1;
		goto out_exit;
	}

	streamfs->com_buffer = (uint8_t *)PIOS_malloc(cfg->write_size);
	if (!streamfs->com_buffer) {
		PIOS_free(streamfs);
		return -1;
	}

	/* Bind configuration parameters to this filesystem instance */
	streamfs->cfg            = cfg;	/* filesystem configuration */
	streamfs->partition_id   = partition_id; /* underlying partition */
	streamfs->partition_size = partition_size; /* size of underlying partition */
	streamfs->partition_arenas = partition_size / cfg->arena_size;

	streamfs->file_open_writing        = false;
	streamfs->file_open_reading        = false;
	streamfs->active_file_id           = 0;
	streamfs->active_file_arena        = 0;
	streamfs->active_file_arena_offset = 0;

	if (PIOS_FLASH_start_transaction(streamfs->partition_id) != 0) {
		rc = -1;
		goto out_exit;
	}

	// TODO: find the first sector after the last file
	// TODO: validate that the partition is valid for streaming (magic?)

	// Scan filesystem contents
	streamfs_scan_filesystem(streamfs);

	rc = 0;

	*fs_id = (uintptr_t) streamfs;

//out_end_trans:
	PIOS_FLASH_end_transaction(streamfs->partition_id);

out_exit:
	return rc;
}

int32_t PIOS_STREAMFS_Destroy(uintptr_t fs_id)
{
	int32_t rc;

	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;

	if (!streamfs_validate(streamfs)) {
		rc = -1;
		goto out_exit;
	}

	streamfs_free(streamfs);
	rc = 0;

out_exit:
	return rc;
}

/**
 * @brief Erases all filesystem arenas and activate the first arena
 * @param[in] fs_id The filesystem to use for this action
 * @return 0 if success or error code
 * @retval -1 if fs_id is not a valid filesystem instance
 * @retval -2 if failed to start transaction
 * @retval -3 if failed to erase all arenas
 * @retval -4 if failed to activate arena 0
 * @retval -5 if failed to mount arena 0
 */
int32_t PIOS_STREAMFS_Format(uintptr_t fs_id)
{
	int32_t rc;

	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;

	if (!streamfs_validate(streamfs)) {
		rc = -1;
		goto out_exit;
	}

	if (PIOS_FLASH_start_transaction(streamfs->partition_id) != 0) {
		rc = -2;
		goto out_exit;
	}

	if (streamfs_erase_all_arenas(streamfs) != 0) {
		rc = -3;
		goto out_end_trans;
	}

	/* Chip erased and log remounted successfully */
	rc = 0;

out_end_trans:
	PIOS_FLASH_end_transaction(streamfs->partition_id);

out_exit:
	return rc;
}


/*
 * File opening and closing utilities
 */

/**
 * Create a new file for streaming to
 *
 * @param[in] fs_id the streaming device handle
 * @returns 0 if successful, <0 if not
 */
int32_t PIOS_STREAMFS_OpenWrite(uintptr_t fs_id)
{
	int32_t rc;

	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;

	if (!streamfs_validate(streamfs)) {
		rc = -1;
		goto out_exit;
	}

	if (streamfs->file_open_writing) {
		rc = -2;
		goto out_exit;
	}

	if (streamfs->file_open_reading) {
		rc = -3;
		goto out_exit;
	}

	if (PIOS_FLASH_start_transaction(streamfs->partition_id) != 0) {
		rc = -4;
		goto out_exit;
	}

	// TODO: use clever scheme to find where to start a new file
	streamfs->active_file_id = streamfs->max_file_id + 1;
	streamfs->active_file_segment = 0;
	streamfs->active_file_arena = streamfs_find_new_sector(streamfs);
	streamfs->active_file_arena_offset = 0;
	streamfs->file_open_writing = true;

	// Erase this sector to prepare for streaming
	if (streamfs_erase_arena(streamfs, streamfs->active_file_arena) != 0) {
		rc = -5;
		goto out_end_trans;
	}

	rc = 0;

out_end_trans:
	PIOS_FLASH_end_transaction(streamfs->partition_id);

out_exit:
	return rc;
}

int32_t PIOS_STREAMFS_OpenRead(uintptr_t fs_id, uint32_t file_id)
{
	int32_t rc;

	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;

	if (!streamfs_validate(streamfs)) {
		rc = -1;
		goto out_exit;
	}

	if (streamfs->file_open_writing) {
		rc = -2;
		goto out_exit;
	}

	if (streamfs->file_open_reading) {
		rc = -3;
		goto out_exit;
	}

	if (PIOS_FLASH_start_transaction(streamfs->partition_id) != 0) {
		rc = -4;
		goto out_exit;
	}

	// Find start of file
	streamfs->active_file_arena = streamfs_find_first_arena(streamfs, file_id);
	if (streamfs->active_file_arena >= 0) {
		streamfs->active_file_id = file_id;
		streamfs->active_file_segment = 0;
		streamfs->active_file_arena_offset = 0;
		streamfs->file_open_reading = true;
	} else {
		streamfs->active_file_arena = 0;
		rc = -5;
		goto out_end_trans;
	}

	rc = 0;

out_end_trans:
	PIOS_FLASH_end_transaction(streamfs->partition_id);

out_exit:
	return rc;
}

int32_t PIOS_STREAMFS_MinFileId(uintptr_t fs_id)
{
	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;

	if (!streamfs_validate(streamfs)) {
		return -1;
	}

	return streamfs->min_file_id;
}

int32_t PIOS_STREAMFS_MaxFileId(uintptr_t fs_id)
{
	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;

	if (!streamfs_validate(streamfs)) {
		return -1;
	}

	return streamfs->max_file_id;
}

int32_t PIOS_STREAMFS_Close(uintptr_t fs_id)
{
	int32_t rc;

	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;

	if (!streamfs_validate(streamfs)) {
		rc = -1;
		goto out_exit;
	}

	if (streamfs->file_open_reading) {
		streamfs->file_open_reading = false;
		rc = 0;
		goto out_exit;
	}

	if (!streamfs->file_open_writing) {
		rc = -2;
		goto out_exit;
	}

	if (PIOS_FLASH_start_transaction(streamfs->partition_id) != 0) {
		rc = -2;
		goto out_exit;
	}

	if (streamfs->active_file_arena_offset != 0) {
		// Close segment when something has been written. This avoids creating
		// null files with an open/close operation
		if (streamfs_close_sector(streamfs) != 0) {
			rc = -3;
			goto out_end_trans;
		}
	}


	// TODO: make sure to flush remaining data
	streamfs->file_open_writing = false;

	if (streamfs_scan_filesystem(streamfs) != 0) {
		rc = -4;
		goto out_end_trans;
	}

	rc = 0;

out_end_trans:
	PIOS_FLASH_end_transaction(streamfs->partition_id);

out_exit:
	return rc;
}

// Testing methods for unit tests

int32_t PIOS_STREAMFS_Testing_Write(uintptr_t fs_id, uint8_t *data, uint32_t len)
{
	int32_t rc;

	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;

	bool valid = streamfs_validate(streamfs);
	PIOS_Assert(valid);

	if (PIOS_FLASH_start_transaction(streamfs->partition_id) != 0) {
		rc = -1;
		goto out_exit;
	}

	rc = streamfs_append_to_file (streamfs, data, len);
	if (rc < 0) {
		rc = -2;
		goto out_end_trans;
	}

	rc = 0;

out_end_trans:
	PIOS_FLASH_end_transaction(streamfs->partition_id);

out_exit:
	return rc;
}

int32_t PIOS_STREAMFS_Testing_Read(uintptr_t fs_id, uint8_t *data, uint32_t len) {
	int32_t rc;

	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;

	bool valid = streamfs_validate(streamfs);
	PIOS_Assert(valid);

	uint16_t num_arenas = streamfs->partition_size / streamfs->cfg->arena_size;
	if (streamfs->active_file_arena >= num_arenas)
		return -1;

	if (PIOS_FLASH_start_transaction(streamfs->partition_id) != 0) {
		rc = -2;
		goto out_exit;
	}

	rc = streamfs_read_from_file (streamfs, data, len);
	if (rc < 0) {
		rc = -3;
		goto out_end_trans;
	}

out_end_trans:
	PIOS_FLASH_end_transaction(streamfs->partition_id);

out_exit:
	return rc;
}

/**********************************
 *
 * Provide a PIOS_COM driver
 *
 *********************************/

static void PIOS_STREAMFS_RxStart(uintptr_t fs_id, uint16_t rx_bytes_avail)
{
	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;
	
	bool valid = streamfs_validate(streamfs);
	PIOS_Assert(valid);

	if (!streamfs->file_open_reading)
		return;

	if (!streamfs->rx_in_cb) {
		return;
	}

	if (PIOS_FLASH_start_transaction(streamfs->partition_id) != 0) {
		return;
	}

	while (rx_bytes_avail) {

		int32_t bytes_to_read = MIN(rx_bytes_avail, streamfs->cfg->write_size);
		if (bytes_to_read == 0)
			goto out_end_trans;

		int32_t bytes_buffered = streamfs_read_from_file(streamfs, streamfs->com_buffer, bytes_to_read);
		if (bytes_buffered == 0)
			goto out_end_trans;

		int32_t bytes_written = (streamfs->rx_in_cb)(streamfs->rx_in_context, streamfs->com_buffer,
			                                         bytes_buffered, NULL, NULL);

		rx_bytes_avail -= bytes_written;
	}

out_end_trans:
	PIOS_FLASH_end_transaction(streamfs->partition_id);
}

static void PIOS_STREAMFS_TxStart(uintptr_t fs_id, uint16_t tx_bytes_avail)
{
	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;
	
	bool valid = streamfs_validate(streamfs);
	PIOS_Assert(valid);

	if (!streamfs->file_open_writing)
		return;

	if (!streamfs->tx_out_cb) {
		return;
	}

	if (PIOS_FLASH_start_transaction(streamfs->partition_id) != 0) {
		return;
	}

	// Flush available data from PIOS_COM interface to file system
	int32_t bytes_to_write;
	while(1) {
		bytes_to_write = (streamfs->tx_out_cb)(streamfs->tx_out_context,
			              streamfs->com_buffer, streamfs->cfg->write_size, NULL, NULL);

		if (bytes_to_write <= 0)
			break;

		if (streamfs_append_to_file (streamfs, streamfs->com_buffer, bytes_to_write) != 0) {
			goto out_end_trans;
		}
	}

out_end_trans:
	PIOS_FLASH_end_transaction(streamfs->partition_id);
}


static void PIOS_STREAMFS_RegisterRxCallback(uintptr_t fs_id, pios_com_callback rx_in_cb, uintptr_t context)
{
	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;

	bool valid = streamfs_validate(streamfs);
	PIOS_Assert(valid);
	
	/* 
	 * Order is important in these assignments since ISR uses _cb
	 * field to determine if it's ok to dereference _cb and _context
	 */
	streamfs->rx_in_context = context;
	streamfs->rx_in_cb = rx_in_cb;
}

static void PIOS_STREAMFS_RegisterTxCallback(uintptr_t fs_id, pios_com_callback tx_out_cb, uintptr_t context)
{
	struct streamfs_state *streamfs = (struct streamfs_state *)fs_id;

	bool valid = streamfs_validate(streamfs);
	PIOS_Assert(valid);
	
	/* 
	 * Order is important in these assignments since ISR uses _cb
	 * field to determine if it's ok to dereference _cb and _context
	 */
	streamfs->tx_out_context = context;
	streamfs->tx_out_cb = tx_out_cb;
}

/**
 * @}
 * @}
 */
