/**
 ******************************************************************************
 * @file       bl_xfer.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup Bootloader
 * @{
 * @addtogroup Bootloader
 * @{
 * @brief Data transfer functions for the Tau Labs unified bootloader
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

#ifndef BL_XFER_H_
#define BL_XFER_H_

#include <stdint.h>		/* uint*_t */
#include <stdbool.h>		/* bool */

#include "bl_messages.h"	/* struct msg_xfer_* */

struct xfer_state {
	bool in_progress;

	uintptr_t partition_id;
	uint32_t partition_base;
	uint32_t partition_size;
	uint32_t original_partition_offset;
	uint32_t current_partition_offset;
	uint32_t next_packet_number;
	bool     check_crc;
	uint32_t bytes_to_crc;
	uint32_t crc;

	uint32_t bytes_to_xfer;
};

extern bool bl_xfer_completed_p(const struct xfer_state * xfer);
extern bool bl_xfer_crc_ok_p(const struct xfer_state * xfer);
extern bool bl_xfer_read_start(struct xfer_state * xfer, const struct msg_xfer_start *xfer_start);
extern bool bl_xfer_send_next_read_packet(struct xfer_state * xfer);
extern bool bl_xfer_write_start(struct xfer_state * xfer, const struct msg_xfer_start *xfer_start);
extern bool bl_xfer_write_cont(struct xfer_state * xfer, const struct msg_xfer_cont *xfer_cont);
extern bool bl_xfer_wipe_partition(const struct msg_wipe_partition *wipe_partition);
extern bool bl_xfer_send_capabilities_self(void);

#endif	/* BL_XFER_H_ */

/**
 * @}
 * @}
 */
