/**
 ******************************************************************************
 * @file       pios_at45_flashfs_logfs.c
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

/* Project Includes */
#include "pios.h"

#include "openpilot.h"

#include "pios_flashfs_logfs_at45_priv.h"

#include <stdbool.h>
#include <stddef.h>		/* NULL */

#define MIN(x,y) ((x) < (y) ? (x) : (y))

/*
 * Filesystem state data tracked in RAM
 */

enum pios_flashfs_logfs_dev_magic {
	PIOS_FLASHFS_LOGFS_DEV_MAGIC = 0x94938201,
};

struct logfs_state {
	enum pios_flashfs_logfs_dev_magic magic;
	const struct flashfs_logfs_cfg * cfg;
	bool mounted;
	uint8_t active_arena_id;

	/* NOTE: num_active_slots + num_free_slots will not typically add
	 *       up to the number of slots in the arena since some of the
	 *       slots will be obsolete or otherwise invalidated
	 */
	uint16_t num_free_slots;   /* slots in free state */
	uint16_t num_active_slots; /* slots in active state */

	/* Underlying flash driver glue */
	const struct pios_flash_driver * driver;
	uintptr_t flash_id;
};

/*
 * Internal Utility functions
 */

/**
 * @brief Return the offset in flash of a particular slot within an arena
 * @return address of the requested slot
 */
static uintptr_t logfs_get_addr(const struct logfs_state * logfs, uint8_t arena_id, uint16_t slot_id)
{
	PIOS_Assert(slot_id < (logfs->cfg->arena_size / logfs->cfg->slot_size));

	return (logfs->cfg->start_offset + (slot_id / (logfs->cfg->page_size / logfs->cfg->slot_size))) ;
}

/*
 * The bits within these enum values must progress ONLY
 * from 1 -> 0 so that we can write later ones on top
 * of earlier ones in NOR flash without an erase cycle.
 */
enum arena_state {
	/*
	 * The STM32F30X flash subsystem is only capable of
	 * writing words or halfwords. In this case we use halfwords.
	 * In addition to that it is only capable to write to erased
	 * cells (0xffff) or write a cell from anything to (0x0000).
	 * To cope with this, the F3 needs carefully crafted enum values.
	 * For this to work the underlying flash driver has to
	 * check each halfword if it has changed before writing.
	 */
	ARENA_STATE_ERASED   = 0xFFFFFFFF,
	ARENA_STATE_RESERVED = 0xE6E6FFFF,
	ARENA_STATE_ACTIVE   = 0xE6E66666,
	ARENA_STATE_OBSOLETE = 0x00000000,
};

struct arena_header {
	uint32_t magic;
	enum arena_state state;
} __attribute__((packed));


/****************************************
 * Arena life-cycle transition functions
 ****************************************/

/**
 * @brief Erases all sectors within the given arena and sets arena to erased state.
 * @return 0 if success, < 0 on failure
 * @note Must be called while holding the flash transaction lock
 */
static int32_t logfs_erase_arena(const struct logfs_state * logfs, uint8_t arena_id)
{
	uintptr_t arena_addr = logfs_get_addr (logfs, arena_id, 0);

	for (uint16_t sector_id = 0;
	     sector_id < (logfs->cfg->arena_size / logfs->cfg->page_size);
	     sector_id++) {
		if (logfs->driver->erase_sector(logfs->flash_id, arena_addr + sector_id)) {
			return -1;
		}
		if (sector_id % 10 == 0)
		  PIOS_LED_Toggle(PIOS_LED_ALARM);
	}

	/* Mark this arena as fully erased */
	struct arena_header arena_hdr = {
		.magic = logfs->cfg->fs_magic,
		.state = ARENA_STATE_ERASED,
	};

	if (logfs->driver->write_data(logfs->flash_id,
					arena_addr,
					(uint8_t *)&arena_hdr,
					sizeof(arena_hdr)) != 0) {
		return -2;
	}

	/* Arena is ready to be activated */
	return 0;
}

/**
 * @brief Marks the given arena as active so it can be mounted.
 * @return 0 if success, < 0 on failure
 * @note Arena must have been previously erased or reserved before calling this
 * @note Must be called while holding the flash transaction lock
 */
static int32_t logfs_activate_arena(const struct logfs_state * logfs, uint8_t arena_id)
{
	uintptr_t arena_addr = logfs_get_addr(logfs, arena_id, 0);

	/* Make sure this arena has been previously erased */
	struct arena_header arena_hdr;
	if (logfs->driver->read_data(logfs->flash_id,
					arena_addr,
					0,
					(uint8_t *)&arena_hdr,
					sizeof (arena_hdr)) != 0) {
		/* Failed to read arena header */
		return -1;
	}

	if ((arena_hdr.state != ARENA_STATE_RESERVED) &&
		(arena_hdr.state != ARENA_STATE_ERASED)) {
		/* Arena was not erased or reserved, can't activate it */
		return -2;
	}
	/* Mark this arena as active */
	arena_hdr.state = ARENA_STATE_ACTIVE;


	// fill buffer with 0xFF
	if (logfs->driver->page_to_buffer(logfs->flash_id, 4095) != 0) {
		return -3;
	}

	if (logfs->driver->write_buffer (logfs->flash_id,
						0,
						(uint8_t *)&arena_hdr,
						sizeof(arena_hdr)) != 0) {
			/* Failed to write the data to the buffer */
			return -3;
		}

    // Write buffer to page
	if (logfs->driver->buffer_to_page (logfs->flash_id,
					arena_addr) != 0){
			/* Failed to write the data to the buffer */
			return -3;
		}

	/* The arena is now activated and the log may be mounted */
	return 0;
}


/**
 * @brief Find the first active arena in flash
 * @return arena_id (>=0) of first active arena
 * @return -1 if no active arena is found
 * @return -2 if failed to read arena header
 * @note Must be called while holding the flash transaction lock
 */
static int32_t logfs_find_active_arena(const struct logfs_state * logfs)
{
	uint8_t arena_id = 0;
	uintptr_t arena_addr = logfs_get_addr (logfs, 0, 0);

	/* Load the arena header */
	struct arena_header arena_hdr;
	if (logfs->driver->read_data(logfs->flash_id,
					arena_addr,
					0,
					(uint8_t *)&arena_hdr,
					sizeof (arena_hdr)) != 0) {
		return -2;
	}

	if ((arena_hdr.state == ARENA_STATE_ACTIVE) &&
		(arena_hdr.magic == logfs->cfg->fs_magic)) {
		/* This is the first active arena */
		return arena_id;
	}
	/* Didn't find an active arena */
	return -1;
}

/*
 * The bits within these enum values must progress ONLY
 * from 1 -> 0 so that we can write later ones on top
 * of earlier ones in NOR flash without an erase cycle.
 */
enum slot_state {
	/*
	 * The STM32F30X flash subsystem is only capable of
	 * writing words or halfwords. In this case we use halfwords.
	 * In addition to that it is only capable to write to erased
	 * cells (0xffff) or write a cell from anything to (0x0000).
	 * To cope with this, the F3 needs carfully crafted enum values.
	 * For this to work the underlying flash driver has to
	 * check each halfword if it has changed before writing.
	 */
	SLOT_STATE_EMPTY    = 0xFFFFFFFF,
	SLOT_STATE_RESERVED = 0xFAFAFFFF,
	SLOT_STATE_ACTIVE   = 0xFAFAAAAA,
	SLOT_STATE_OBSOLETE = 0x00000000,
};

struct slot_header {
	enum slot_state state;
	uint32_t obj_id;
	uint16_t obj_inst_id;
	uint16_t obj_size;
} __attribute__((packed));

/*
 * Is the entire filesystem full?
 * true = all slots in the arena are in the ACTIVE state (ie. garbage collection won't free anything)
 * false = some slots in the arena are either currently free or could be free'd by garbage collection
 */
static bool logfs_fs_is_full(const struct logfs_state * logfs)
{
	return (logfs->num_active_slots == (logfs->cfg->arena_size / logfs->cfg->slot_size) - (logfs->cfg->page_size / logfs->cfg->slot_size) - 1);
}

static int32_t logfs_unmount_log(struct logfs_state * logfs)
{
	PIOS_Assert (logfs->mounted);

	logfs->num_active_slots = 0;
	logfs->num_free_slots   = 0;
	logfs->mounted          = false;

	return 0;
}

static int32_t logfs_mount_log(struct logfs_state * logfs, uint8_t arena_id)
{
	PIOS_Assert (!logfs->mounted);

	logfs->num_active_slots = 0;
	logfs->num_free_slots   = 0;
	logfs->active_arena_id  = arena_id;

	/* Scan the log to find out how full it is */
		uint16_t slots_in_page = logfs->cfg->page_size / logfs->cfg->slot_size;

	/* First slot in the arena is reserved for arena header, skip it. */
	struct slot_header slot_hdr;
	for (uint16_t slot_id = slots_in_page;  // page 0 is reserved for arena header
		 slot_id < (logfs->cfg->arena_size / logfs->cfg->slot_size);
		 slot_id++) {
		 uintptr_t page_addr = logfs_get_addr (logfs, logfs->active_arena_id, slot_id);

			if (logfs->driver->read_data(logfs->flash_id,
							page_addr,
							logfs->cfg->slot_size * (slot_id % slots_in_page),
							(uint8_t *)&slot_hdr,
							sizeof (slot_hdr)) != 0) {
				return -2;
			}

			/*
			 * Empty slots must be in a continguous block at the
			 * end of the arena.
			 */
			switch (slot_hdr.state) {
			case SLOT_STATE_EMPTY:
				logfs->num_free_slots++;
				break;
			case SLOT_STATE_ACTIVE:
				logfs->num_active_slots++;
				break;
			case SLOT_STATE_RESERVED:
			case SLOT_STATE_OBSOLETE:
				break;
			}
	}

	/* Scan is complete, mark the arena mounted */
	logfs->active_arena_id = arena_id;
	logfs->mounted = true;

	return 0;
}

static bool PIOS_FLASHFS_Logfs_validate(const struct logfs_state * logfs)
{
	return (logfs && (logfs->magic == PIOS_FLASHFS_LOGFS_DEV_MAGIC));
}

#if defined(PIOS_INCLUDE_FREERTOS)
static struct logfs_state * PIOS_FLASHFS_Logfs_alloc(void)
{
	struct logfs_state * logfs;

	logfs = (struct logfs_state *)pvPortMalloc(sizeof(*logfs));
	if (!logfs) return (NULL);

	logfs->magic = PIOS_FLASHFS_LOGFS_DEV_MAGIC;
	return(logfs);
}
static void PIOS_FLASHFS_Logfs_free(struct logfs_state * logfs)
{
	/* Invalidate the magic */
	logfs->magic = ~PIOS_FLASHFS_LOGFS_DEV_MAGIC;
	vPortFree(logfs);
}
#else
static struct logfs_state pios_flashfs_logfs_devs[PIOS_FLASHFS_LOGFS_MAX_DEVS];
static uint8_t pios_flashfs_logfs_num_devs;
static struct logfs_state * PIOS_FLASHFS_Logfs_alloc(void)
{
	struct logfs_state * logfs;

	if (pios_flashfs_logfs_num_devs >= PIOS_FLASHFS_LOGFS_MAX_DEVS) {
		return (NULL);
	}

	logfs = &pios_flashfs_logfs_devs[pios_flashfs_logfs_num_devs++];
	logfs->magic = PIOS_FLASHFS_LOGFS_DEV_MAGIC;

	return (logfs);
}
static void PIOS_FLASHFS_Logfs_free(struct logfs_state * logfs)
{
	/* Invalidate the magic */
	logfs->magic = ~PIOS_FLASHFS_LOGFS_DEV_MAGIC;

	/* Can't free the resources with this simple allocator */
}
#endif

/**
 * @brief Initialize the flash object setting FS
 * @return 0 if success, -1 if failure
 */
int32_t PIOS_FLASHFS_Logfs_Init(uintptr_t * fs_id, const struct flashfs_logfs_cfg * cfg, const struct pios_flash_driver * driver, uintptr_t flash_id)
{
	PIOS_Assert(cfg);
	PIOS_Assert(fs_id);
	PIOS_Assert(driver);

	/* Make sure the underlying flash driver provides the minimal set of required methods */
	PIOS_Assert(driver->start_transaction);
	PIOS_Assert(driver->end_transaction);
	PIOS_Assert(driver->erase_sector);
	PIOS_Assert(driver->write_data);
	PIOS_Assert(driver->read_data);

	int8_t rc;

	struct logfs_state * logfs;

	logfs = (struct logfs_state *) PIOS_FLASHFS_Logfs_alloc();
	if (!logfs) {
		rc = -1;
		goto out_exit;
	}

	/* Bind configuration parameters to this filesystem instance */
	logfs->cfg      = cfg;	/* filesystem configuration */
	logfs->driver   = driver; /* lower-level flash driver */
	logfs->flash_id = flash_id; /* lower-level flash device id */
	logfs->mounted  = false;

	if (logfs->driver->start_transaction(logfs->flash_id) != 0) {
		rc = -1;
		goto out_exit;
	}

	bool found = false;
	int32_t arena_id;
	for (uint8_t try = 0; !found && try < 2; try++) {
		/* Find the active arena */
		arena_id = logfs_find_active_arena(logfs);
		if (arena_id >= 0) {
			/* Found the active arena */
			found = true;
			break;
		} else {
			/* No active arena found, erase and activate arena 0 */
			if (logfs_erase_arena(logfs, 0) != 0)
				break;

			if (logfs_activate_arena(logfs, 0) != 0)
				break;
		}
	}

	if (!found) {
		/* Still no active arena, something is broken */
		rc = -2;
		goto out_end_trans;
	}

	/* We've found an active arena, mount it */
	if (logfs_mount_log(logfs, arena_id) != 0) {
		/* Failed to mount the log, something is broken */
		rc = -3;
		goto out_end_trans;
	}
	/* Log has been mounted */
	rc = 0;

	*fs_id = (uintptr_t) logfs;

out_end_trans:
	logfs->driver->end_transaction(logfs->flash_id);

out_exit:
	return rc;
}

int32_t PIOS_FLASHFS_Logfs_Destroy(uintptr_t fs_id)
{
	int32_t rc;

	struct logfs_state * logfs = (struct logfs_state *)fs_id;

	if (!PIOS_FLASHFS_Logfs_validate(logfs)) {
		rc = -1;
		goto out_exit;
	}

	PIOS_FLASHFS_Logfs_free(logfs);
	rc = 0;

out_exit:
	return rc;
}


/* NOTE: Must be called while holding the flash transaction lock */
static int16_t logfs_object_find_next (const struct logfs_state * logfs, struct slot_header * slot_hdr, uint16_t * curr_slot, uint32_t obj_id, uint16_t obj_inst_id)
{
	PIOS_Assert(slot_hdr);
	PIOS_Assert(curr_slot);

	uint16_t slots_in_page = logfs->cfg->page_size / logfs->cfg->slot_size;
	/* First slot in the arena is reserved for arena header, skip it. */
	if (*curr_slot < slots_in_page) *curr_slot = slots_in_page;

	for (uint16_t slot_id = *curr_slot;
	     slot_id < (logfs->cfg->arena_size / logfs->cfg->slot_size);
	     slot_id++) {
		uintptr_t page_addr = logfs_get_addr (logfs, logfs->active_arena_id, slot_id);

		if (logfs->driver->read_data(logfs->flash_id,
						page_addr,
						logfs->cfg->slot_size * (slot_id % slots_in_page),
						(uint8_t *)slot_hdr,
						sizeof (*slot_hdr)) != 0) {
			return -2;
		}

			if (slot_hdr->state == SLOT_STATE_EMPTY ) {
				/* We hit the end of the log */
				// object does not exist Return slot addres of first Empty slot
				return -3;
			}

			if (slot_hdr->state == SLOT_STATE_ACTIVE &&
				slot_hdr->obj_id      == obj_id &&
				slot_hdr->obj_inst_id == obj_inst_id) {
				/* Found what we were looking for */
				*curr_slot = slot_id;
				return 0;
			}
	}

	/* No matching entry was found */
	return -1;
}

/* NOTE: Must be called while holding the flash transaction lock */
static int16_t logfs_get_free_slot (const struct logfs_state * logfs, uint16_t * curr_slot)
{
	PIOS_Assert(curr_slot);

	uint16_t slots_in_page = logfs->cfg->page_size / logfs->cfg->slot_size;

	if (*curr_slot < slots_in_page) *curr_slot = slots_in_page;
	struct slot_header slot_hdr;
	for (uint16_t slot_id = *curr_slot; // 2
	     slot_id < (logfs->cfg->arena_size / logfs->cfg->slot_size);
	     slot_id++) {
		uintptr_t page_addr = logfs_get_addr (logfs, logfs->active_arena_id, slot_id);

			if (logfs->driver->read_data(logfs->flash_id,
							page_addr,
							logfs->cfg->slot_size * (slot_id % slots_in_page),
							(uint8_t *)&slot_hdr,
							sizeof (slot_hdr)) != 0) {
				return -2;
			}

			if (slot_hdr.state == SLOT_STATE_EMPTY || slot_hdr.state == SLOT_STATE_OBSOLETE ) {
				*curr_slot = slot_id;
				return 0;
			}
	}
	// No matching entry was found //
	return -1;
}


/* NOTE: Must be called while holding the flash transaction lock */
/* OPTIMIZE: could trust that there is at most one active version of every object and terminate the search when we find one */
static int8_t logfs_delete_object (struct logfs_state * logfs, uint32_t obj_id, uint16_t obj_inst_id)
{
	int8_t rc;

	bool more = true;
	uint16_t curr_slot_id = 0;
	do {
		struct slot_header slot_hdr;
		switch (logfs_object_find_next (logfs, &slot_hdr, &curr_slot_id, obj_id, obj_inst_id)) {

		case 0:
			/* Found a matching slot.  Obsolete it. */
			slot_hdr.state = SLOT_STATE_OBSOLETE;
			uintptr_t slot_addr = logfs_get_addr (logfs, logfs->active_arena_id, curr_slot_id);
			// Read page to buffer
			if (logfs->driver->page_to_buffer(logfs->flash_id, slot_addr) != 0) {
				/* Failed to read slot header for candidate slot */
				return -2;
			}

			/* Write the data into the reserved slot, starting after the slot header */
			if (logfs->driver->write_buffer (logfs->flash_id,
								logfs->cfg->slot_size * (curr_slot_id % (logfs->cfg->page_size / logfs->cfg->slot_size)),
								(uint8_t *)&slot_hdr,
								sizeof(slot_hdr)) != 0){
					/* Failed to write the data to the buffer */
					return -2;
				}

		    // Write buffer to slot_addr
			if (logfs->driver->buffer_to_page (logfs->flash_id,
								slot_addr) != 0){
					/* Failed to write the data to the buffer */
					return -2;
				}

			/* Object has been successfully obsoleted and is no longer active */
			logfs->num_active_slots--;
			break;
		case -1:
			/* Search completed, object not found */
			more = false;
			rc = 0;
			break;
		default:
			/* Error occurred during search */
			rc = -1;
			goto out_exit;
		}
	} while (more);

out_exit:
	return rc;
}

/* NOTE: Must be called while holding the flash transaction lock */
static int8_t logfs_append_to_log (struct logfs_state * logfs, uint32_t obj_id, uint16_t obj_inst_id, uint8_t * obj_data, uint16_t obj_size)
{
	/* Reserve a free slot for our new object */
	uint16_t append_slot_id = logfs->cfg->page_size / logfs->cfg->slot_size;
	struct slot_header slot_hdr;

	/* Find the object in the log */
	if (logfs_object_find_next (logfs, &slot_hdr, &append_slot_id, obj_id, obj_inst_id) < 0) {
		 /* Object currently not saved */
		append_slot_id = logfs->cfg->page_size / logfs->cfg->slot_size;
		 if( logfs_get_free_slot (logfs,  &append_slot_id) !=0){
			 // no free slot
			goto out_exit;
		 }

	}
	/* Compute slot address */
	uintptr_t page_addr =  logfs_get_addr (logfs, logfs->active_arena_id, append_slot_id);

	// Read this page to buffer
	if (logfs->driver->page_to_buffer(logfs->flash_id, page_addr) != 0) {
		/* Failed to read slot header for candidate slot */
		return -2;
	}
	/* Mark this slot active in one atomic step */
	slot_hdr.state = SLOT_STATE_ACTIVE;
	slot_hdr.obj_id      = obj_id;
	slot_hdr.obj_inst_id = obj_inst_id;
	slot_hdr.obj_size    = obj_size;
    // write slot header to buffer
	if (logfs->driver->write_buffer (logfs->flash_id,
					logfs->cfg->slot_size * (append_slot_id % (logfs->cfg->page_size / logfs->cfg->slot_size)),
					(uint8_t *)&slot_hdr,
					sizeof (slot_hdr)) != 0){
		/* Failed to write the data to the buffer */
		return -2;
	}



	/* Write the data into the reserved slot, starting after the slot header */
	uintptr_t slot_offset = sizeof(slot_hdr);
	if (logfs->driver->write_buffer (logfs->flash_id,
						logfs->cfg->slot_size * (append_slot_id % (logfs->cfg->page_size / logfs->cfg->slot_size)) + slot_offset,
						obj_data,
						obj_size) != 0){
			/* Failed to write the data to the buffer */
			return -2;
		}

    // Write buffer to page
	if (logfs->driver->buffer_to_page (logfs->flash_id,
						page_addr) != 0){
			/* Failed to write the data to the buffer */
			return -2;
		}

	/* Object has been successfully written to the slot */
	logfs->num_active_slots++;

	logfs->num_free_slots--;
out_exit:
	return 0;
}


/**********************************
 *
 * Provide a PIOS_FLASHFS_* driver
 *
 *********************************/
#include "pios_flashfs.h"	/* API for flash filesystem */

/**
 * @brief Saves one object instance to the filesystem
 * @param[in] fs_id The filesystem to use for this action
 * @param[in] obj UAVObject ID of the object to save
 * @param[in] obj_inst_id The instance number of the object being saved
 * @param[in] obj_data Contents of the object being saved
 * @param[in] obj_size Size of the object being saved
 * @return 0 if success or error code
 * @retval -1 if fs_id is not a valid filesystem instance
 * @retval -2 if failed to start transaction
 * @retval -3 if failure to delete any previous versions of the object
 * @retval -4 if filesystem is entirely full and garbage collection won't help
 * @retval -5 if garbage collection failed
 * @retval -6 if filesystem is full even after garbage collection should have freed space
 * @retval -7 if writing the new object to the filesystem failed
 */
int32_t PIOS_FLASHFS_ObjSave(uintptr_t fs_id, uint32_t obj_id, uint16_t obj_inst_id, uint8_t * obj_data, uint16_t obj_size)
{
	int8_t rc;

	struct logfs_state * logfs = (struct logfs_state *)fs_id;

	if (!PIOS_FLASHFS_Logfs_validate(logfs)) {
		rc = -1;
		goto out_exit;
	}

	PIOS_Assert(obj_size <= (logfs->cfg->slot_size - sizeof(struct slot_header)));

	if (logfs->driver->start_transaction(logfs->flash_id) != 0) {
		rc = -2;
		goto out_exit;
	}

	/*
	 * All old versions of this object + instance have been invalidated.
	 * Write the new object.
	 */

	/* Check if the arena is entirely full. */
	if (logfs_fs_is_full(logfs)) {
		/* Note: Filesystem Full means we're full of *active* records so gc won't help at all. */
		rc = -4;
		goto out_end_trans;
	}

	/* We have room for our new object.  Append it to the log. */
	if (logfs_append_to_log(logfs, obj_id, obj_inst_id, obj_data, obj_size) != 0) {
		/* Error during append */
		rc = -7;
		goto out_end_trans;
	}
	/* Object successfully written to the log */
	rc = 0;

out_end_trans:
	logfs->driver->end_transaction(logfs->flash_id);

out_exit:
	return rc;
}

/**
 * @brief Load one object instance from the filesystem
 * @param[in] fs_id The filesystem to use for this action
 * @param[in] obj UAVObject ID of the object to load
 * @param[in] obj_inst_id The instance of the object to load
 * @param[in] obj_data Buffer to hold the contents of the loaded object
 * @param[in] obj_size Size of the object to be loaded
 * @return 0 if success or error code
 * @retval -1 if fs_id is not a valid filesystem instance
 * @retval -2 if failed to start transaction
 * @retval -3 if object not found in filesystem
 * @retval -4 if object size in filesystem does not exactly match buffer size
 * @retval -5 if reading the object data from flash fails
 */
int32_t PIOS_FLASHFS_ObjLoad(uintptr_t fs_id, uint32_t obj_id, uint16_t obj_inst_id, uint8_t * obj_data, uint16_t obj_size)
{
	int8_t rc;

	struct logfs_state * logfs = (struct logfs_state *)fs_id;

	if (!PIOS_FLASHFS_Logfs_validate(logfs)) {
		rc = -1;
		goto out_exit;
	}

	PIOS_Assert(obj_size <= (logfs->cfg->slot_size - sizeof(struct slot_header)));
	if (logfs->driver->start_transaction(logfs->flash_id) != 0) {
		rc = -2;
		goto out_exit;
	}

	/* Find the object in the log */
	uint16_t slot_id = 0;
	struct slot_header slot_hdr;
	if (logfs_object_find_next (logfs, &slot_hdr, &slot_id, obj_id, obj_inst_id) != 0) {
		/* Object does not exist in fs */
		rc = -3;
		goto out_end_trans;
	}

	/* Sanity check what we've found */
	if (slot_hdr.obj_size != obj_size) {
		/* Object sizes don't match.  Not safe to copy contents. */
		rc = -4;
		goto out_end_trans;
	}

	/* Read the contents of the object from the log */
	if (obj_size > 0) {
		uintptr_t slot_addr = logfs_get_addr (logfs, logfs->active_arena_id, slot_id);
		// Read data from page=slot_addr
		if (logfs->driver->read_data(logfs->flash_id,
						slot_addr,
						logfs->cfg->slot_size * (slot_id % (logfs->cfg->page_size / logfs->cfg->slot_size)) + sizeof(slot_hdr),
						(uint8_t *)obj_data,
						obj_size) != 0) {
			/* Failed to read object data from the log */
			rc = -5;
			goto out_end_trans;
		}
	}

	/* Object successfully loaded */
	rc = 0;

out_end_trans:
	logfs->driver->end_transaction(logfs->flash_id);

out_exit:
	return rc;
}

/**
 * @brief Delete one instance of an object from the filesystem
 * @param[in] fs_id The filesystem to use for this action
 * @param[in] obj UAVObject ID of the object to delete
 * @param[in] obj_inst_id The instance of the object to delete
 * @return 0 if success or error code
 * @retval -1 if fs_id is not a valid filesystem instance
 * @retval -2 if failed to start transaction
 * @retval -3 if failed to delete the object from the filesystem
 */
int32_t PIOS_FLASHFS_ObjDelete(uintptr_t fs_id, uint32_t obj_id, uint16_t obj_inst_id)
{
	int8_t rc;

	struct logfs_state * logfs = (struct logfs_state *)fs_id;

	if (!PIOS_FLASHFS_Logfs_validate(logfs)) {
		rc = -1;
		goto out_exit;
	}

	if (logfs->driver->start_transaction(logfs->flash_id) != 0) {
		rc = -2;
		goto out_exit;
	}

	if (logfs_delete_object (logfs, obj_id, obj_inst_id) != 0) {
		rc = -3;
		goto out_end_trans;
	}

	/* Object successfully deleted from the log */
	rc = 0;

out_end_trans:
	logfs->driver->end_transaction(logfs->flash_id);

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
int32_t PIOS_FLASHFS_Format(uintptr_t fs_id)
{
	int32_t rc;

	struct logfs_state * logfs = (struct logfs_state *)fs_id;

	if (!PIOS_FLASHFS_Logfs_validate(logfs)) {
		rc = -1;
		goto out_exit;
	}

	if (logfs->mounted) {
		logfs_unmount_log(logfs);
	}

	if (logfs->driver->start_transaction(logfs->flash_id) != 0) {
		rc = -2;
		goto out_exit;
	}

	if (logfs_erase_arena(logfs, 0) != 0) {
		rc = -3;
		goto out_end_trans;
	}

	/* Reinitialize arena 0 */
	if (logfs_activate_arena(logfs, 0) != 0) {
		rc = -4;
		goto out_end_trans;
	}

	/* Mount arena 0 */
	if (logfs_mount_log(logfs, 0) != 0) {
		rc = -5;
		goto out_end_trans;
	}

	/* Chip erased and log remounted successfully */
	rc = 0;

out_end_trans:
	logfs->driver->end_transaction(logfs->flash_id);

out_exit:
	return rc;
}

/**
 * @}
 * @}
 */
