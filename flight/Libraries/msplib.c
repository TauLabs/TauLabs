/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 *
 * @file       msplib.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015-2016
 * @brief      Library for handling MSP protocol communications
 *
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

#include "msplib.h"

//! Initialize an MSP parser
struct msp_bridge * msp_init(uintptr_t com)
{
	struct msp_bridge * msp = PIOS_malloc(sizeof(*msp));
	if (msp != NULL) {
		memset(msp, 0x00, sizeof(*msp));
		msp->com = com;
	}

	return msp;
}

 //! Send response to a query packet
void msp_send_response(struct msp_bridge *m, uint8_t cmd, const uint8_t *data, size_t len)
{
	uint8_t buf[5];
	uint8_t cs = (uint8_t)(len) ^ cmd;

	buf[0] = '$';
	buf[1] = 'M';
	buf[2] = '>';
	buf[3] = (uint8_t)(len);
	buf[4] = cmd;

	PIOS_COM_SendBuffer(m->com, buf, sizeof(buf));
	PIOS_COM_SendBuffer(m->com, data, len);

	for (int i = 0; i < len; i++) {
		cs ^= data[i];
	}
	cs ^= 0;

	buf[0] = cs;
	PIOS_COM_SendBuffer(m->com, buf, 1);
}

//! Query an update
void msp_send_request(struct msp_bridge *m, uint8_t type)
{
	const uint32_t REQUEST_PACKET_SIZE = 6;

	uint8_t buf[REQUEST_PACKET_SIZE];
	buf[0] = '$';
	buf[1] = 'M';
	buf[2] = '<';
	buf[3] = 0;
	buf[4] = type;
	buf[5] = 0 ^ type; // calculate checksum

	PIOS_COM_SendBuffer(m->com, buf, REQUEST_PACKET_SIZE);
}

void msp_set_response_cb(struct msp_bridge *m, msp_cb response_cb)
{
	m->response_cb = (msp_cb_store) response_cb;
}

void msp_set_request_cb(struct msp_bridge *m, msp_cb request_cb)
{
	m->request_cb = (msp_cb_store) request_cb;
}

//! Receive the size of the next packet
static msp_state msp_state_size(struct msp_bridge *m, uint8_t b)
{
	m->cmd_size = b;
	m->checksum = b;

	// Advance to next state but track if we are in a command or response
	return (m->state == MSP_HEADER_C_SIZE) ? MSP_HEADER_C_CMD : MSP_HEADER_R_CMD;
}

//! Receive the command of the next packet
static msp_state msp_state_cmd(struct msp_bridge *m, uint8_t b)
{
	m->cmd_i = 0;
	m->cmd_id = b;
	m->checksum ^= m->cmd_id;

	if (m->cmd_size > sizeof(m->cmd_data)) {
		// Too large a body.  Let's ignore it.
		return MSP_DISCARD;
	}

	// Advance to next state but track if we are in a command or response
	return m->cmd_size == 0 ? (m->state == MSP_HEADER_C_CMD ? MSP_C_CHECKSUM : MSP_R_CHECKSUM) : 
	                          (m->state == MSP_HEADER_C_CMD ? MSP_C_FILLBUF : MSP_R_FILLBUF);
}

//! Receive the data packet for commands or responses
static msp_state msp_state_fill_buf(struct msp_bridge *m, uint8_t b)
{
	m->cmd_data.data[m->cmd_i++] = b;
	m->checksum ^= b;

	// Advance to next state but track if we are in a command or response
	return m->cmd_i == m->cmd_size ? (m->state == MSP_HEADER_C_CMD ? MSP_C_CHECKSUM : MSP_R_CHECKSUM) : m->state;
}

//! Receive the checksum of the current packet
static msp_state msp_state_checksum(struct msp_bridge *m, uint8_t b)
{
	if ((m->checksum ^ b) != 0) {
		return MSP_IDLE;
	}

	if (m->state == MSP_C_CHECKSUM) {
		// Respond to requests/commands we support
		if(m->request_cb) {
			m->request_cb((void *)m, m->cmd_id, m->cmd_data.data, m->cmd_size);
		}
	} else if (m->state == MSP_R_CHECKSUM) {
		// Respond to requests/commands we support
		if(m->response_cb) {
			m->response_cb((void *)m, m->cmd_id, m->cmd_data.data, m->cmd_size);
		}
	}

	return MSP_IDLE;
}

//! Throw out bytes when we can't receive a large packet
static msp_state msp_state_discard(struct msp_bridge *m, uint8_t b)
{
	return m->cmd_i++ == m->cmd_size ? MSP_IDLE : MSP_DISCARD;
}

/**
 * Process incoming bytes from an MSP packet
 * for the OSD this should mostly be responses to queries
 * we made
 * @param[in] b received byte
 * @return true if we should continue processing bytes
 */
bool msp_receive_byte(struct msp_bridge *m, uint8_t b)
{
	switch (m->state) {
	case MSP_IDLE:
		m->state = b == '$' ? MSP_HEADER_START : MSP_IDLE;
		break;
	case MSP_HEADER_START:
		m->state = b == 'M' ? MSP_HEADER_M : MSP_IDLE;
		break;
	case MSP_HEADER_M:
		m->state = b == '<' ? MSP_HEADER_C_SIZE : // command/request packet
		           b == '>' ? MSP_HEADER_R_SIZE : // response packet
		           MSP_IDLE;
		break;
	case MSP_HEADER_C_SIZE:
	case MSP_HEADER_R_SIZE:
		m->state = msp_state_size(m, b);
		break;
	case MSP_HEADER_C_CMD:
	case MSP_HEADER_R_CMD:
		m->state = msp_state_cmd(m, b);
		break;
	case MSP_C_FILLBUF:
	case MSP_R_FILLBUF:
		m->state = msp_state_fill_buf(m, b);
		break;
	case MSP_C_CHECKSUM:
	case MSP_R_CHECKSUM:
		m->state = msp_state_checksum(m, b);
		break;
	case MSP_DISCARD:
		m->state = msp_state_discard(m, b);
		break;
	}

	return true;
}

/**
 * @}
 */
