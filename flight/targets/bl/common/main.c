/**
 ******************************************************************************
 * @file       main.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup Bootloader
 * @{
 * @addtogroup Bootloader
 * @{
 * @brief Tau Labs unified bootloader main loop and FSM
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

#include <pios.h>
#include <pios_board_info.h>	/* struct pios_board_info */
#include <stdbool.h>		/* bool */
#include "pios_iap.h"		/* PIOS_IAP_* */
#include "pios_com_msg.h"	/* PIOS_COM_MSG_* */
#include "pios_usbhook.h"	/* PIOS_USBHOOK_* */

#include "pios_bl_helper.h"	/* PIOS_BL_HELPER_* */

#include "led_pwm.h"		/* led_pwm_* */
#include "bl_messages.h"	/* struct bl_messages */
#include "bl_xfer.h"		/* bl_xfer_* */

extern void PIOS_Board_Init(void);

#define MSEC_TO_USEC(ms) ((ms) * 1000)
#define SEC_TO_MSEC(s) ((s) * 1000)
#define SEC_TO_USEC(s) ((s) * 1000 * 1000)

#define BL_DETECT_BREAK_TO_BL_TIMER MSEC_TO_USEC(500)
#define BL_WAIT_FOR_DFU_TIMER SEC_TO_USEC(6)
#define BL_RECOVER_FROM_FAULT_TIMER SEC_TO_USEC(10)

enum bl_states {
	BL_STATE_FSM_FAULT = 0,	/* Must be zero so undefined transitions land here */
	BL_STATE_INIT,
	BL_STATE_DETECT_BREAK_TO_BL,
	BL_STATE_WAIT_FOR_DFU,
	BL_STATE_JUMPING_TO_APP,
	BL_STATE_JUMP_FAILED,
	BL_STATE_DFU_OPERATION_FAILED,
	BL_STATE_DFU_OPERATION_OK,
	BL_STATE_DFU_IDLE,
	BL_STATE_DFU_READ_IN_PROGRESS,
	BL_STATE_DFU_WRITE_IN_PROGRESS,

	BL_STATE_NUM_STATES	/* Must be last */
};

enum legacy_dfu_states {
	DFU_IDLE = 0,
	DFU_WRITING,		/* Uploading */
	DFU_WRONG_PACKET_RX,
	DFU_TOO_MANY_PACKETS,
	DFU_TOO_FEW_PACKETS,
	DFU_LAST_OP_SUCCESS,
	DFU_READING,		/* Downloading */
	DFU_BL_IDLE,
	DFU_LAST_OP_FAILED,
	DFU_WRITE_START,	/* Upload Starting */
	DFU_OUTSIDE_DEV_CAP,
	DFU_CRC_FAIL,
	DFU_FAILED_JUMP,
};

/* Map from internal bootloader states to the legacy states that the old bootloaders advertised */
static const uint8_t fsm_to_dfu_state_map[] = {
	[BL_STATE_FSM_FAULT]             = DFU_OUTSIDE_DEV_CAP,
	[BL_STATE_INIT]                  = DFU_BL_IDLE,
	[BL_STATE_DETECT_BREAK_TO_BL]    = DFU_BL_IDLE,
	[BL_STATE_WAIT_FOR_DFU]          = DFU_BL_IDLE,
	[BL_STATE_JUMPING_TO_APP]        = DFU_IDLE,
	[BL_STATE_JUMP_FAILED]           = DFU_FAILED_JUMP,
	[BL_STATE_DFU_OPERATION_FAILED]  = DFU_LAST_OP_FAILED,
	[BL_STATE_DFU_OPERATION_OK]      = DFU_LAST_OP_SUCCESS,
	[BL_STATE_DFU_IDLE]              = DFU_IDLE,
	[BL_STATE_DFU_READ_IN_PROGRESS]  = DFU_READING,
	[BL_STATE_DFU_WRITE_IN_PROGRESS] = DFU_WRITING,
};

enum bl_events {
	BL_EVENT_ENTER_DFU,
	BL_EVENT_USB_CONNECTED,
	BL_EVENT_USB_DISCONNECTED,
	BL_EVENT_JUMP_TO_APP,
	BL_EVENT_TIMER_EXPIRY,
	BL_EVENT_READ_START,
	BL_EVENT_WRITE_START,
	BL_EVENT_ABORT_OPERATION,
	BL_EVENT_TRANSFER_DONE,
	BL_EVENT_TRANSFER_ERROR,
	BL_EVENT_AUTO,

	BL_EVENT_NUM_EVENTS	/* Must be last */
};

struct bl_fsm_context {
	enum bl_states curr_state;

	/* Received packet */
	struct bl_messages msg;

	/* FSM timer */
	bool fsm_timer_enabled;
	uint32_t fsm_timer_remaining_us;

	/* LED state */
	struct led_pwm_state leds;

	/* Transfer state */
	uint8_t dfu_device_number;
	struct xfer_state xfer;
};

struct bl_transition {
	void (*entry_fn) (struct bl_fsm_context * context);
	enum bl_states next_state[BL_EVENT_NUM_EVENTS];
};

static void go_fsm_fault(struct bl_fsm_context * context);
static void go_detect_break_to_bl(struct bl_fsm_context * context);
static void go_wait_for_dfu(struct bl_fsm_context * context);
static void go_jumping_to_app(struct bl_fsm_context * context);
static void go_dfu_idle(struct bl_fsm_context * context);
static void go_read_in_progress(struct bl_fsm_context * context);
static void go_write_in_progress(struct bl_fsm_context * context);
static void go_dfu_operation_ok(struct bl_fsm_context * context);
static void go_dfu_operation_failed(struct bl_fsm_context * context);

const static struct bl_transition bl_transitions[BL_STATE_NUM_STATES] = {
	[BL_STATE_FSM_FAULT] = {
		.entry_fn = go_fsm_fault,
		.next_state = {
			[BL_EVENT_TIMER_EXPIRY]     = BL_STATE_INIT,
		}
	},
	[BL_STATE_INIT] = {
		.entry_fn = NULL,
		.next_state = {
			[BL_EVENT_AUTO]             = BL_STATE_DETECT_BREAK_TO_BL,
		},
	},
	[BL_STATE_DETECT_BREAK_TO_BL] = {
		.entry_fn = go_detect_break_to_bl,
		.next_state = {
			[BL_EVENT_ENTER_DFU]        = BL_STATE_DFU_IDLE,
			[BL_EVENT_ABORT_OPERATION]  = BL_STATE_WAIT_FOR_DFU,
			[BL_EVENT_USB_CONNECTED]    = BL_STATE_WAIT_FOR_DFU,
			[BL_EVENT_TIMER_EXPIRY]     = BL_STATE_JUMPING_TO_APP,
		},
	},
	[BL_STATE_WAIT_FOR_DFU] = {
		.entry_fn = go_wait_for_dfu,
		.next_state = {
			[BL_EVENT_ENTER_DFU]        = BL_STATE_DFU_IDLE,
			[BL_EVENT_TIMER_EXPIRY]     = BL_STATE_JUMPING_TO_APP,
			[BL_EVENT_ABORT_OPERATION]  = BL_STATE_WAIT_FOR_DFU,
			[BL_EVENT_USB_CONNECTED]    = BL_STATE_WAIT_FOR_DFU,
			[BL_EVENT_USB_DISCONNECTED] = BL_STATE_JUMPING_TO_APP,
		},
	},
	[BL_STATE_JUMPING_TO_APP] = {
		.entry_fn = go_jumping_to_app,
		.next_state = {
			[BL_EVENT_AUTO]             = BL_STATE_JUMP_FAILED,
		},
	},
	[BL_STATE_JUMP_FAILED] = {
		.entry_fn = NULL,
		.next_state = {
			[BL_EVENT_AUTO]             = BL_STATE_INIT,
		},
	},
	[BL_STATE_DFU_IDLE] = {
		.entry_fn = go_dfu_idle,
		.next_state = {
			[BL_EVENT_ENTER_DFU]        = BL_STATE_DFU_IDLE,
			[BL_EVENT_READ_START]       = BL_STATE_DFU_READ_IN_PROGRESS,
			[BL_EVENT_WRITE_START]      = BL_STATE_DFU_WRITE_IN_PROGRESS,
			[BL_EVENT_ABORT_OPERATION]  = BL_STATE_DFU_IDLE,
			[BL_EVENT_JUMP_TO_APP]      = BL_STATE_JUMPING_TO_APP,
			[BL_EVENT_USB_CONNECTED]    = BL_STATE_DFU_IDLE,
			[BL_EVENT_USB_DISCONNECTED] = BL_STATE_JUMPING_TO_APP,
		},
	},
	[BL_STATE_DFU_READ_IN_PROGRESS] = {
		.entry_fn = go_read_in_progress,
		.next_state = {
			[BL_EVENT_ABORT_OPERATION]  = BL_STATE_DFU_IDLE,
			[BL_EVENT_TRANSFER_DONE]    = BL_STATE_DFU_IDLE,
			[BL_EVENT_TRANSFER_ERROR]   = BL_STATE_DFU_OPERATION_FAILED,
			[BL_EVENT_USB_DISCONNECTED] = BL_STATE_JUMPING_TO_APP,
			[BL_EVENT_JUMP_TO_APP]      = BL_STATE_JUMPING_TO_APP,
		},
	},

	[BL_STATE_DFU_WRITE_IN_PROGRESS] = {
		.entry_fn = go_write_in_progress,
		.next_state = {
			[BL_EVENT_ABORT_OPERATION]  = BL_STATE_DFU_IDLE,
			[BL_EVENT_TRANSFER_DONE]    = BL_STATE_DFU_OPERATION_OK,
			[BL_EVENT_TRANSFER_ERROR]   = BL_STATE_DFU_OPERATION_FAILED,
			[BL_EVENT_USB_DISCONNECTED] = BL_STATE_JUMPING_TO_APP,
		},
	},

	[BL_STATE_DFU_OPERATION_OK] = {
		.entry_fn = go_dfu_operation_ok,
		.next_state = {
			[BL_EVENT_ENTER_DFU]        = BL_STATE_DFU_IDLE,
			[BL_EVENT_READ_START]       = BL_STATE_DFU_READ_IN_PROGRESS,
			[BL_EVENT_WRITE_START]      = BL_STATE_DFU_WRITE_IN_PROGRESS,
			[BL_EVENT_ABORT_OPERATION]  = BL_STATE_DFU_IDLE,
			[BL_EVENT_JUMP_TO_APP]      = BL_STATE_JUMPING_TO_APP,
			[BL_EVENT_USB_CONNECTED]    = BL_STATE_DFU_IDLE,
			[BL_EVENT_USB_DISCONNECTED] = BL_STATE_JUMPING_TO_APP,
		},
	},

	[BL_STATE_DFU_OPERATION_FAILED] = {
		.entry_fn = go_dfu_operation_failed,
		.next_state = {
			[BL_EVENT_ABORT_OPERATION]  = BL_STATE_DFU_IDLE,
			[BL_EVENT_USB_DISCONNECTED] = BL_STATE_JUMPING_TO_APP,
		},
	},
};

static enum bl_states bl_fsm_get_state(struct bl_fsm_context * context)
{
	return context->curr_state;
}

static void bl_fsm_timer_start(struct bl_fsm_context * context, uint32_t timer_duration_us)
{
	context->fsm_timer_remaining_us = timer_duration_us;
	context->fsm_timer_enabled = true;
}

static void bl_fsm_timer_cancel(struct bl_fsm_context * context)
{
	context->fsm_timer_enabled = false;
}

static void bl_fsm_timer_add_ticks(struct bl_fsm_context * context, uint32_t elapsed_us)
{
	if (context->fsm_timer_enabled) {
		if (elapsed_us >= context->fsm_timer_remaining_us) {
			/* Timer has expired */
			context->fsm_timer_remaining_us = 0;
		} else {
			/* Timer is still running, account for the elapsed time */
			context->fsm_timer_remaining_us -= elapsed_us;
		}
	}
}

static bool bl_fsm_timer_expired_p(struct bl_fsm_context * context)
{
	if ((context->fsm_timer_enabled) && (context->fsm_timer_remaining_us == 0))
		return true;

	return false;
}

static void go_fsm_fault(struct bl_fsm_context * context)
{
	bl_fsm_timer_start(context, BL_RECOVER_FROM_FAULT_TIMER);
	led_pwm_config(&context->leds, 2500, 100, 2500, 100);
}

static void go_detect_break_to_bl(struct bl_fsm_context * context)
{
	/* Start a timer for how long to wait for a USB host to appear */
	bl_fsm_timer_start(context, BL_DETECT_BREAK_TO_BL_TIMER);
	led_pwm_config(&context->leds, 0, 0, 0, 0);
	PIOS_LED_On(PIOS_LED_HEARTBEAT);
}

static void go_wait_for_dfu(struct bl_fsm_context * context)
{
	/* Start a timer for how long to wait for DFU mode to be activated */
	bl_fsm_timer_start(context, BL_WAIT_FOR_DFU_TIMER);
	led_pwm_config(&context->leds, 0, 0, 0, 0);
	PIOS_LED_On(PIOS_LED_HEARTBEAT);
}

static void go_jumping_to_app(struct bl_fsm_context * context)
{
	bl_fsm_timer_cancel(context);

	PIOS_LED_On(PIOS_LED_HEARTBEAT);

	/* Recover a pointer to the bootloader board info blob */
	const struct pios_board_info * bdinfo = &pios_board_info_blob;

	/* Load the application's PC and SP from the beginning of the firmware flash bank */
	uintptr_t initial_sp = *((__IO uint32_t*) bdinfo->fw_base);
	uintptr_t initial_pc = *((__IO uint32_t*) (bdinfo->fw_base + 4));

	/* Simple sanity check to make sure that the SP actually points to RAM */
	if (((initial_sp & 0x2FFE0000) != 0x20000000) && ((initial_sp & 0x1FFE0000) != 0x10000000)) {
		/* sp is not sane, don't attempt the jump */
		return;
	}

	/* Simple sanity check to make sure that the PC actually points to FLASH */
	if ((initial_pc & 0x08000000) != 0x08000000) {
		/* pc is not sane, don't attempt the jump */
		return;
	}

	/*
	 * pc and sp are sane, try the jump to the application
	 */

	/* Disable USB */
	PIOS_USBHOOK_Deactivate();

	/* Re-lock the internal flash */
	FLASH_Lock();

	/* Reset all peripherals */
	RCC_APB2PeriphResetCmd(0xffffffff, ENABLE);
	RCC_APB1PeriphResetCmd(0xffffffff, ENABLE);
	RCC_APB2PeriphResetCmd(0xffffffff, DISABLE);
	RCC_APB1PeriphResetCmd(0xffffffff, DISABLE);

	/* Initialize user application's stack pointer */
	__set_MSP(initial_sp);

	/* Jump to the application entry point */
	((void (*)(void))initial_pc)();
}

static void go_dfu_idle(struct bl_fsm_context * context)
{
	bl_fsm_timer_cancel(context);
	led_pwm_config(&context->leds, 5000, 100, 0, 0);
}

static void go_read_in_progress(struct bl_fsm_context * context)
{
	bl_fsm_timer_cancel(context);
	led_pwm_config(&context->leds, 5000, 100, 2500, 50);
}

static void go_write_in_progress(struct bl_fsm_context * context)
{
	bl_fsm_timer_cancel(context);
	led_pwm_config(&context->leds, 2500, 50, 0, 0);
}

static void go_dfu_operation_ok(struct bl_fsm_context * context)
{
	bl_fsm_timer_cancel(context);
	led_pwm_config(&context->leds, 5000, 100, 0, 0);
}

static void go_dfu_operation_failed(struct bl_fsm_context * context)
{
	bl_fsm_timer_cancel(context);
	led_pwm_config(&context->leds, 5000, 100, 0, 0);
}

static void bl_fsm_process_auto(struct bl_fsm_context *context)
{
	while (bl_transitions[context->curr_state].next_state[BL_EVENT_AUTO]) {
		context->curr_state = bl_transitions[context->curr_state].next_state[BL_EVENT_AUTO];

		/* Call the entry function (if any) for the next state. */
		if (bl_transitions[context->curr_state].entry_fn) {
			bl_transitions[context->curr_state].entry_fn(context);
		}
	}
}

static void bl_fsm_inject_event(struct bl_fsm_context * context, enum bl_events event)
{
	/*
	 * Move to the next state
	 *
	 * This is done prior to calling the new state's entry function to
	 * guarantee that the entry function never depends on the previous
	 * state.  This way, it cannot ever know what the previous state was.
	 */
	context->curr_state = bl_transitions[context->curr_state].next_state[event];

	/* Call the entry function (if any) for the next state. */
	if (bl_transitions[context->curr_state].entry_fn) {
		bl_transitions[context->curr_state].entry_fn(context);
	}

	/* Process any AUTO transitions in the FSM */
	bl_fsm_process_auto(context);
}

static void bl_fsm_init(struct bl_fsm_context *context)
{
	memset(context, 0, sizeof(*context));

	context->curr_state = BL_STATE_INIT;
	bl_fsm_process_auto(context);
}

static void process_packet_rx(struct bl_fsm_context * context, const struct bl_messages * msg);

int main(void)
{
	/* Configure and enable system clocks */
	PIOS_SYS_Init();

	/* Initialize the board-specific HW configuration */
	PIOS_Board_Init();

	/* Initialize the bootloader FSM */
	struct bl_fsm_context bl_fsm_context;
	bl_fsm_init(&bl_fsm_context);

	/* Check if the user has requested that we boot into DFU mode */
	PIOS_IAP_Init();
	if (PIOS_IAP_CheckRequest() == true) {
		/* User has requested that we boot into DFU mode */
		PIOS_IAP_ClearRequest();
		bl_fsm_inject_event(&bl_fsm_context, BL_EVENT_ENTER_DFU);
	}

	/* Assume no USB connected */
	bool usb_connected = false;

	uint32_t prev_ticks = PIOS_DELAY_GetuS();
	while (1) {
		uint32_t elapsed_ticks = PIOS_DELAY_GetuSSince(prev_ticks);

		/* Run the fsm timer */
		if (elapsed_ticks) {
			bl_fsm_timer_add_ticks(&bl_fsm_context, elapsed_ticks);
			if (bl_fsm_timer_expired_p(&bl_fsm_context) == true) {
				/* Timer has expired, inject an expiry event */
				bl_fsm_inject_event(&bl_fsm_context, BL_EVENT_TIMER_EXPIRY);
			}

			/* pulse the LEDs */
			led_pwm_add_ticks(&bl_fsm_context.leds, elapsed_ticks);
			led_pwm_update_leds(&bl_fsm_context.leds);

			prev_ticks += elapsed_ticks;
		}

		/* check for changes in USB connection state */
		if (!usb_connected) {
			if (PIOS_USB_CableConnected(0)) {
				bl_fsm_inject_event(&bl_fsm_context, BL_EVENT_USB_CONNECTED);
				usb_connected = true;
			}
		} else {
			if (!PIOS_USB_CableConnected(0)) {
				bl_fsm_inject_event(&bl_fsm_context, BL_EVENT_USB_DISCONNECTED);
				usb_connected = false;
			}
		}

		/* Manage any active read transfers */
		if (bl_fsm_get_state(&bl_fsm_context) == BL_STATE_DFU_READ_IN_PROGRESS) {
			if (bl_xfer_send_next_read_packet(&bl_fsm_context.xfer)) {
				/* Sent a packet.  Are we finished? */
				if (bl_xfer_completed_p(&bl_fsm_context.xfer)) {
					/* Transfer is finished */
					bl_fsm_inject_event(&bl_fsm_context, BL_EVENT_TRANSFER_DONE);
				}
			} else {
				/* Failed to send packet, fail the transfer */
				bl_fsm_inject_event(&bl_fsm_context, BL_EVENT_TRANSFER_ERROR);
			}
		}

		/* check for an incoming packet */
		bool packet_available = PIOS_COM_MSG_Receive(PIOS_COM_TELEM_USB,
							(uint8_t *)&bl_fsm_context.msg,
							sizeof(bl_fsm_context.msg));
		if (packet_available) {
			process_packet_rx(&bl_fsm_context, &bl_fsm_context.msg);
		}
	}
}

static bool bl_select_dfu_device(struct bl_fsm_context * context, uint8_t device_number)
{
	if (device_number != 0) {
		/* BL currently only deals with device number 0 (ie. self) so refuse any other device */
		return false;
	}

	/* TODO: Should check if we're in the right state for this */

	context->dfu_device_number = device_number;

	return true;
}

static bool bl_send_status(struct bl_fsm_context * context)
{
	struct bl_messages msg = {
		.flags_command = BL_MSG_STATUS_REP,
		.v.status_rep = {
			.current_state = fsm_to_dfu_state_map[context->curr_state],
		},
	};

	PIOS_COM_MSG_Send(PIOS_COM_TELEM_USB, (uint8_t *)&msg, sizeof(msg));

	return true;
}

static bool bl_send_capabilities(struct bl_fsm_context * context, uint8_t device_number)
{
	if (device_number == 0) {
		/* Return capabilities of all devices */
		struct bl_messages msg = {
			.flags_command = BL_MSG_CAP_REP,
			.v.cap_rep_all = {
				.number_of_devices = htons(1),
				.wrflags = htons(0x3),
			},
		};
		PIOS_COM_MSG_Send(PIOS_COM_TELEM_USB, (uint8_t *)&msg, sizeof(msg));
	} else if (device_number == 1) {
		bl_xfer_send_capabilities_self();
	} else {
		return false;
	}

	return true;
}

static void process_packet_rx(struct bl_fsm_context * context, const struct bl_messages * msg)
{
	/* Sanity checks */
	if (!msg)
		return;

	/* Split the flags_command field into flags and command */
	uint8_t flags   = msg->flags_command & BL_MSG_FLAGS_MASK;
	uint8_t command = msg->flags_command & BL_MSG_COMMAND_MASK;

	struct bl_messages echo;
	if (flags & BL_MSG_FLAGS_ECHO_REQ) {
		/* Request should be echoed back to the sender.  Make a copy of the request */
		memcpy (&echo, msg, sizeof(echo));
	}

	switch (command) {
	case BL_MSG_CAP_REQ:
		bl_send_capabilities(context, msg->v.cap_req.device_number);
		break;
	case BL_MSG_ENTER_DFU:
		if (bl_select_dfu_device(context, msg->v.enter_dfu.device_number)) {
			bl_fsm_inject_event(context, BL_EVENT_ENTER_DFU);
		} else {
			/* Failed to select the requested device */
		}
		break;
	case BL_MSG_JUMP_FW:
		if (ntohs(msg->v.jump_fw.safe_word) == 0x5afe) {
			/* Force board into safe mode */
			PIOS_IAP_WriteBootCount(0xFFFF);
		}
		bl_fsm_inject_event(context, BL_EVENT_JUMP_TO_APP);
		break;
	case BL_MSG_RESET:
		PIOS_SYS_Reset();
		break;
	case BL_MSG_OP_ABORT:
		bl_fsm_inject_event(context, BL_EVENT_ABORT_OPERATION);
		break;

	case BL_MSG_WRITE_START:
		if (bl_xfer_write_start(&context->xfer, &(msg->v.xfer_start))) {
			bl_fsm_inject_event(context, BL_EVENT_WRITE_START);
		} else {
			/* Failed to start the write */
		}
		break;
	case BL_MSG_WRITE_CONT:
		if (bl_fsm_get_state(context) == BL_STATE_DFU_WRITE_IN_PROGRESS) {
			if (!bl_xfer_write_cont(&context->xfer, &(msg->v.xfer_cont))) {
				/* Invalid packet, fail the transfer */
				bl_fsm_inject_event(context, BL_EVENT_TRANSFER_ERROR);
			}
		}
		break;

	case BL_MSG_OP_END:
		if (bl_fsm_get_state(context) == BL_STATE_DFU_WRITE_IN_PROGRESS) {
			if (bl_xfer_completed_p(&context->xfer)) {
				/* Transfer is finished, check the CRC */
				if (bl_xfer_crc_ok_p(&context->xfer)) {
					bl_fsm_inject_event(context, BL_EVENT_TRANSFER_DONE);
				} else {
					/* Mismatched CRC */
					bl_fsm_inject_event(context, BL_EVENT_TRANSFER_ERROR);
				}
			}
		}
		break;

	case BL_MSG_READ_START:
		if (bl_xfer_read_start(&context->xfer, &(msg->v.xfer_start))) {
			bl_fsm_inject_event(context, BL_EVENT_READ_START);
		} else {
			/* Failed to start the read */
		}
		break;

	case BL_MSG_STATUS_REQ:
		bl_send_status(context);
		break;

	case BL_MSG_WIPE_PARTITION:
		bl_xfer_wipe_partition(&(msg->v.wipe_partition));
		break;

	case BL_MSG_CAP_REP:
	case BL_MSG_STATUS_REP:
	case BL_MSG_READ_CONT:
		/* We've received a *reply* packet when we expected a request. */
		break;
	case BL_MSG_RESERVED:
		/* We've received an reserved command.  Ignore this packet. */
		break;

	default:
		/* We've received an unknown command.  Ignore this packet. */
		break;
	}

	if (flags & BL_MSG_FLAGS_ECHO_REQ) {
		/* Reflect the original message back to the originator as an echo reply */
		echo.flags_command &= ~(BL_MSG_FLAGS_MASK);
		echo.flags_command |= BL_MSG_FLAGS_ECHO_REP;

		PIOS_COM_MSG_Send(PIOS_COM_TELEM_USB, (uint8_t *)&echo, sizeof(echo));
	}

}
