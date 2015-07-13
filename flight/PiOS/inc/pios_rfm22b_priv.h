/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_RFM22B Radio Functions
 * @brief PIOS interface for RFM22B Radio
 * @{
 *
 * @file       pios_rfm22b_priv.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      RFM22B private definitions.
 * @see        The GNU Public License (GPL) Version 3
 *
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

#ifndef PIOS_RFM22B_PRIV_H
#define PIOS_RFM22B_PRIV_H

#include <pios.h>
#include <fifo_buffer.h>
#include <uavobjectmanager.h>
#include <rfm22bstatus.h>
#include "pios_rfm22b.h"
#include "pios_rfm22b_regs.h"
#include "pios_semaphore.h"
#include "pios_thread.h"

// External type definitions

typedef int16_t(*t_rfm22_TxDataByteCallback) (void);
typedef bool(*t_rfm22_RxDataCallback) (void *data, uint8_t len);
enum pios_rfm22b_dev_magic {
	PIOS_RFM22B_DEV_MAGIC = 0x68e971b6,
};

enum pios_radio_state {
	RADIO_STATE_UNINITIALIZED,
	RADIO_STATE_INITIALIZING,
	RADIO_STATE_RX_MODE,
	RADIO_STATE_RX_DATA,
	RADIO_STATE_RX_FAILURE,
	RADIO_STATE_TX_START,
	RADIO_STATE_TX_DATA,
	RADIO_STATE_TIMEOUT,
	RADIO_STATE_ERROR,
	RADIO_STATE_FATAL_ERROR,

	RADIO_STATE_NUM_STATES	// Must be last
};

enum pios_radio_event {
	RADIO_EVENT_DEFAULT,
	RADIO_EVENT_INT_RECEIVED,
	RADIO_EVENT_INITIALIZE,
	RADIO_EVENT_INITIALIZED,
	RADIO_EVENT_RX_MODE,
	RADIO_EVENT_RX_COMPLETE,
	RADIO_EVENT_TX_START,
	RADIO_EVENT_TIMEOUT,
	RADIO_EVENT_ERROR,
	RADIO_EVENT_FATAL_ERROR,

	RADIO_EVENT_NUM_EVENTS	// Must be last
};

enum pios_rfm22b_state {
	RFM22B_STATE_INITIALIZING,
	RFM22B_STATE_TRANSITION,
	RFM22B_STATE_RX_WAIT,
	RFM22B_STATE_RX_WAIT_SYNC,
	RFM22B_STATE_RX_MODE,
	RFM22B_STATE_TX_MODE,
	RFM22B_STATE_TRANSMITTING,

	RFM22B_STATE_NUM_STATES	// Must be last
};

#define RFM22B_RX_PACKET_STATS_LEN 32
enum pios_rfm22b_rx_packet_status {
	RADIO_STATS_IGNORE = 0x00,
	RADIO_GOOD_RX_PACKET = 0x01,
	RADIO_CORRECTED_RX_PACKET = 0x02,
	RADIO_ERROR_RX_PACKET = 0x03,
	RADIO_ERROR_RX_SYNC_MISSED = 0x04,
	RADIO_ERROR_TX_MISSED = 0x05,
};

enum pios_rfm22b_chip_power_state {
	RFM22B_IDLE_STATE = 0x00,
	RFM22B_RX_STATE = 0x01,
	RFM22B_TX_STATE = 0x10,
	RFM22B_INVALID_STATE = 0x11
};

// Device Status
typedef union {
	struct {
		uint8_t state:2;
		bool frequency_error:1;
		bool header_error:1;
		bool rx_fifo_empty:1;
		bool fifo_underflow:1;
		bool fifo_overflow:1;
	};
	uint8_t raw;
} rfm22b_device_status_reg;

// EzMAC Status
typedef union {
	struct {
		bool packet_sent:1;
		bool packet_transmitting:1;
		bool crc_error:1;
		bool valid_packet_received:1;
		bool packet_receiving:1;
		bool packet_searching:1;
		bool crc_is_all_ones:1;
		bool reserved;
	};
	uint8_t raw;
} rfm22b_ezmac_status_reg;

// Interrrupt Status Register 1
typedef union {
	struct {
		bool crc_error:1;
		bool valid_packet_received:1;
		bool packet_sent_interrupt:1;
		bool external_interrupt:1;
		bool rx_fifo_almost_full:1;
		bool tx_fifo_almost_empty:1;
		bool tx_fifo_almost_full:1;
		bool fifo_underoverflow_error:1;
	};
	uint8_t raw;
} rfm22b_int_status_1;

// Interrupt Status Register 2
typedef union {
	struct {
		bool poweron_reset:1;
		bool chip_ready:1;
		bool low_battery_detect:1;
		bool wakeup_timer:1;
		bool rssi_above_threshold:1;
		bool invalid_preamble_detected:1;
		bool valid_preamble_detected:1;
		bool sync_word_detected:1;
	};
	uint8_t raw;
} rfm22b_int_status_2;

typedef struct {
	rfm22b_device_status_reg device_status;
	rfm22b_device_status_reg ezmac_status;
	rfm22b_int_status_1 int_status_1;
	rfm22b_int_status_2 int_status_2;
} rfm22b_device_status;

struct pios_rfm22b_dev {
	enum pios_rfm22b_dev_magic magic;
	struct pios_rfm22b_cfg cfg;

	// The SPI bus information
	uint32_t spi_id;
	uint32_t slave_num;

	// Should this modem ack as a coordinator.
	bool coordinator;

	// The device ID
	uint32_t deviceID;

	// The coodinator ID (0 if this modem is a coordinator).
	uint32_t coordinatorID;

	// The task handle
	struct pios_thread *taskHandle;

	// The COM callback functions.
	pios_com_callback rx_in_cb;
	uint32_t rx_in_context;
	pios_com_callback tx_out_cb;
	uint32_t tx_out_context;

	// the transmit power to use for data transmissions
	uint8_t tx_power;

	// The RF datarate lookup index.
	uint8_t datarate;

	// The radio state machine state
	enum pios_radio_state state;

	// The event queue handle
	struct pios_semaphore *sema_isr;

	// The device status registers.
	rfm22b_device_status status_regs;

	// The error statistics counters
	uint32_t rx_packet_stats[RFM22B_RX_PACKET_STATS_LEN];

	// The RFM22B state machine state
	enum pios_rfm22b_state rfm22b_state;

	// The packet statistics
	struct rfm22b_stats stats;

	// Stats
	uint16_t errors;

	// RSSI in dBm
	int8_t rssi_dBm;

	// The tx data packet
	uint8_t tx_packet[RFM22B_MAX_PACKET_LEN];
	// The current tx packet
	uint8_t *tx_packet_handle;
	// The tx data read index
	uint16_t tx_data_rd;
	// The tx data write index
	uint16_t tx_data_wr;

	// The rx data packet
	uint8_t rx_packet[RFM22B_MAX_PACKET_LEN];
	// The rx data packet
	uint8_t *rx_packet_handle;
	// The receive buffer write index
	uint16_t rx_buffer_wr;
	// The receive buffer write index
	uint16_t rx_packet_len;

	// The PPM buffer
	int16_t ppm[RFM22B_PPM_NUM_CHANNELS];

	// RFM22B RCVR interface
	uintptr_t rfm22b_rcvr_id;

	// The id that the packet was received from
	uint32_t rx_destination_id;
	// The maximum packet length (including header, etc.)
	uint8_t max_packet_len;
	// The packet transmit time in ms.
	uint8_t packet_time;
	// Do all packets originate from the coordinator modem?
	bool one_way_link;
	// Should this modem send PPM data?
	bool ppm_send_mode;
	// Should this modem receive PPM data?
	bool ppm_recv_mode;
	// Are we sending / receiving only PPM data?
	bool ppm_only_mode;

	// The channel list
	uint8_t channels[RFM22B_NUM_CHANNELS];
	// The number of frequency hopping channels.
	uint8_t num_channels;
	// The frequency hopping step size
	float frequency_step_size;
	// current frequency hop channel
	uint8_t channel;
	// current frequency hop channel index
	uint8_t channel_index;
	// afc correction reading (in Hz)
	int8_t afc_correction_Hz;

	// The packet timers.
	uint32_t packet_start_ticks;
	uint32_t tx_complete_ticks;
	uint32_t time_delta;

	// Track when a packet is received in this slice
	bool packet_received_slice;
	// Track consecutive sync packets that were missed
	uint8_t sync_pulses_missed;
};

// External function definitions

bool PIOS_RFM22_EXT_Int(void);
bool PIOS_RFM22B_Validate(struct pios_rfm22b_dev *rfm22b_dev);

// Global variable definitions

extern const struct pios_com_driver pios_rfm22b_com_driver;

#endif /* PIOS_RFM22B_PRIV_H */

/**
 * @}
 * @}
 */
