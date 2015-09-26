/**
 ******************************************************************************
 *
 * @file       bl_messages.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Uploader Uploader Plugin
 * @{
 * @brief Low level bootloader protocol structures
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

#ifndef PACK
#ifdef _MSC_VER
#define PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop) )
#else
#define PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
#endif /* _MSC_VER */
#endif /* PACK */

namespace tl_dfu {

#define BL_INCLUDE_CAP_EXTENSIONS

#define uint8_t quint8
#define uint16_t quint16
#define uint32_t quint32

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

#if !defined(ntohl)
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
#endif

/*
 * Note: These enum values MUST NOT be changed or backward
 *       compatibility will be broken
 */

#ifdef _MSC_VER
#pragma pack(push,1)
#endif
enum dfu_partition_label {
    DFU_PARTITION_FW,
    DFU_PARTITION_DESC,
    DFU_PARTITION_BL,
    DFU_PARTITION_SETTINGS,
    DFU_PARTITION_WAYPOINTS,
    DFU_PARTITION_LOG,
    DFU_PARTITION_OTA,
#ifdef _MSC_VER
};
#pragma pack(pop)
#else
}__attribute__((packed));
#endif

struct msg_capabilities_req {
	uint8_t unused[4];
	uint8_t device_number;
};

struct msg_capabilities_rep_all {
	uint8_t unused[4];
	uint16_t number_of_devices;
	uint16_t wrflags;
};

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
};

struct msg_enter_dfu {
	uint8_t unused[4];
	uint8_t device_number;
};

struct msg_jump_fw {
	uint8_t unused[4];
	uint8_t unused2[2];
	uint16_t safe_word;
};

struct msg_reset {
	/* No subfields */
};

struct msg_op_abort {
	/* No subfields */
};

struct msg_op_end {
	/* No subfields */
};

PACK(struct msg_xfer_start {
	uint32_t packets_in_transfer;
	uint8_t label;
	uint8_t words_in_last_packet;
	uint32_t expected_crc; /* only used in writes */
});

#define XFER_BYTES_PER_PACKET 56
struct msg_xfer_cont {
	uint32_t current_packet_number;
	uint8_t data[XFER_BYTES_PER_PACKET];
};

struct msg_status_req {
	/* No subfields */
};

struct msg_status_rep {
	uint32_t unused;
	uint8_t current_state;
};

struct msg_wipe_partition {
	uint8_t label;
};

PACK(union msg_contents {
    struct msg_capabilities_req cap_req;
    struct msg_capabilities_rep_all cap_rep_all;
    struct msg_capabilities_rep_specific cap_rep_specific;
    struct msg_enter_dfu enter_dfu;
    struct msg_jump_fw jump_fw;
    struct msg_reset reset;
    struct msg_op_abort op_abort;
    struct msg_op_end op_end;
    struct msg_xfer_start xfer_start;
    struct msg_xfer_cont xfer_cont;
    struct msg_status_req status_req;
    struct msg_status_rep status_rep;
    struct msg_wipe_partition wipe_partition;
    uint8_t pad[62];
});

PACK(struct bl_messages {
    uint8_t flags_command;

    union msg_contents v;
});

} /* namespace tl_dfu */

#endif // BL_MESSAGES_H
