/**
 ******************************************************************************
 * @file       bl_messages.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup Bootloader
 * @{
 * @addtogroup Bootloader
 * @{
 * @brief Message definitions for the Tau Labs unified bootloader
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

#ifndef BL_MESSAGES_H_
#define BL_MESSAGES_H_

#include <stdint.h>		/* uint*_t */

/* Note:
 *   Writes are from PC -> FC
 *   Reads  are from FC -> PC
 */
enum bl_commands {
	BL_MSG_RESERVED = 0,
	BL_MSG_CAP_REQ,
	BL_MSG_CAP_REP,
	BL_MSG_ENTER_DFU,
	BL_MSG_JUMP_FW,
	BL_MSG_RESET,
	BL_MSG_OP_ABORT,
	BL_MSG_WRITE_CONT,
	BL_MSG_OP_END,
	BL_MSG_READ_START,
	BL_MSG_READ_CONT,
	BL_MSG_STATUS_REQ,
	BL_MSG_STATUS_REP,
	BL_MSG_WIPE_PARTITION,

	BL_MSG_WRITE_START = 0x27,
};

#define BL_MSG_FLAGS_ECHO_REQ 0x80
#define BL_MSG_FLAGS_ECHO_REP 0x40
#define BL_MSG_FLAGS_MASK     0xC0
#define BL_MSG_COMMAND_MASK   0x3F


#define ntohl(v) (				\
	(((v) & 0xFF000000) >> 24) |		\
	(((v) & 0x00FF0000) >>  8) |		\
	(((v) & 0x0000FF00) <<  8) |		\
	(((v) & 0x000000FF) << 24))

#define ntohs(v) (				\
	(((v) & 0xFF00) >> 8) |			\
	(((v) & 0x00FF) << 8))

#define htonl(v) ntohl((v))

#define htons(v) ntohs((v))

/*
 * Note: These enum values MUST NOT be changed or backward
 *       compatibility will be broken
 */
enum dfu_partition_label {
	DFU_PARTITION_FW,
	DFU_PARTITION_DESC,
	DFU_PARTITION_BL,
	DFU_PARTITION_SETTINGS,
	DFU_PARTITION_WAYPOINTS,
} __attribute__((packed));

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */

struct bl_messages {
	uint8_t flags_command;

	union {
		struct msg_capabilities_req {
			uint8_t unused[4];
			uint8_t device_number;
		} cap_req;

		struct msg_capabilities_rep_all {
			uint8_t unused[4];
			uint16_t number_of_devices;
			uint16_t wrflags;
		} cap_rep_all;

		struct msg_capabilities_rep_specific {
			uint32_t fw_size;
			uint8_t device_number;
			uint8_t bl_version;
			uint8_t desc_size;
			uint8_t board_rev;
			uint32_t fw_crc;
			uint16_t device_id;
#if defined(BL_INCLUDE_CAP_EXTENSIONS)
			/* Extensions to original protocol */
#define BL_CAP_EXTENSION_MAGIC 0x3456
			uint16_t cap_extension_magic;
			uint32_t partition_sizes[10];
#endif	/* BL_INCLUDE_CAP_EXTENSIONS */
		} cap_rep_specific;

		struct msg_enter_dfu {
			uint8_t unused[4];
			uint8_t device_number;
		} enter_dfu;

		struct msg_jump_fw {
			uint8_t unused[4];
			uint8_t unused2[2];
			uint16_t safe_word;
		} jump_fw;

		struct msg_reset {
			/* No subfields */
		} reset;

		struct msg_op_abort {
			/* No subfields */
		} op_abort;

		struct msg_op_end {
			/* No subfields */
		} op_end;

		struct msg_xfer_start {
			uint32_t packets_in_transfer;
			enum dfu_partition_label label;
			uint8_t words_in_last_packet;
			uint32_t expected_crc; /* only used in writes */
		} xfer_start;

#define XFER_BYTES_PER_PACKET 56
		struct msg_xfer_cont {
			uint32_t current_packet_number;
			uint8_t data[XFER_BYTES_PER_PACKET];
		} xfer_cont;

		struct msg_status_req {
			/* No subfields */
		} status_req;

		struct msg_status_rep {
			uint32_t unused;
			uint8_t current_state;
		} status_rep;

		struct msg_wipe_partition {
			enum dfu_partition_label label;
		} wipe_partition;

		uint8_t pad[62];
	} __attribute__((aligned(1)))v;
} __attribute__((packed));

#pragma pack(pop)   /* restore original alignment from stack */

#endif	/* BL_MESSAGES_H_ */

/**
 * @}
 * @}
 */
