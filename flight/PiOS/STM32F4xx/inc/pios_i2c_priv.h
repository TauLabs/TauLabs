/**
 ******************************************************************************
 *
 * @file       pios_i2c_priv.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      I2C private definitions.
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

#ifndef PIOS_I2C_PRIV_H
#define PIOS_I2C_PRIV_H

#include <pios.h>
#include <pios_stm32.h>
#include <stdbool.h>
#include "pios_semaphore.h"
#include <inttypes.h>

struct pios_i2c_adapter_cfg {
	I2C_TypeDef *regs;
	uint32_t remap;
	I2C_InitTypeDef init;

	uint32_t transfer_timeout_ms;
	struct stm32_gpio scl;
	struct stm32_gpio sda;
	struct stm32_irq event;
	struct stm32_irq error;
};

enum pios_i2c_adapter_magic {
	PIOS_I2C_DEV_MAGIC = 0xa9a9b8b8,
};

enum i2c_adapter_state {
	I2C_STATE_FSM_FAULT = 0,	/* Must be zero so undefined transitions land here */

	I2C_STATE_BUS_ERROR,

	I2C_STATE_STOPPED,
	I2C_STATE_STOPPING,
	I2C_STATE_STARTING,
	I2C_STATE_RESTARTING,

	I2C_STATE_R_MORE_TXN_ADDR,
	I2C_STATE_R_MORE_TXN_PRE_ONE,
	I2C_STATE_R_MORE_TXN_PRE_FIRST,
	I2C_STATE_R_MORE_TXN_PRE_MIDDLE,
	I2C_STATE_R_MORE_TXN_PRE_LAST,
	I2C_STATE_R_MORE_TXN_POST_LAST,

	I2C_STATE_R_LAST_TXN_ADDR,
	I2C_STATE_R_LAST_TXN_PRE_ONE,
	I2C_STATE_R_LAST_TXN_PRE_FIRST,
	I2C_STATE_R_LAST_TXN_PRE_MIDDLE,
	I2C_STATE_R_LAST_TXN_PRE_LAST,
	I2C_STATE_R_LAST_TXN_POST_LAST,

	I2C_STATE_W_MORE_TXN_ADDR,
	I2C_STATE_W_MORE_TXN_PRE_MIDDLE,
	I2C_STATE_W_MORE_TXN_PRE_LAST,
	I2C_STATE_W_MORE_TXN_POST_LAST,

	I2C_STATE_W_LAST_TXN_ADDR,
	I2C_STATE_W_LAST_TXN_PRE_MIDDLE,
	I2C_STATE_W_LAST_TXN_PRE_LAST,
	I2C_STATE_W_LAST_TXN_POST_LAST,

	I2C_STATE_NACK,

	I2C_STATE_NUM_STATES	/* Must be last */
};

enum i2c_adapter_event {
	I2C_EVENT_BUS_ERROR,
	I2C_EVENT_START,
	I2C_EVENT_R_MORE_TXN_STARTED,
	I2C_EVENT_W_MORE_TXN_STARTED,
	I2C_EVENT_R_LAST_TXN_STARTED,
	I2C_EVENT_W_LAST_TXN_STARTED,
	I2C_EVENT_ADDR_SENT_LEN_EQ_0,
	I2C_EVENT_ADDR_SENT_LEN_EQ_1,
	I2C_EVENT_ADDR_SENT_LEN_EQ_2,
	I2C_EVENT_ADDR_SENT_LEN_GT_2,
	I2C_EVENT_TRANSFER_DONE_LEN_EQ_0,
	I2C_EVENT_TRANSFER_DONE_LEN_EQ_1,
	I2C_EVENT_TRANSFER_DONE_LEN_EQ_2,
	I2C_EVENT_TRANSFER_DONE_LEN_GT_2,
	I2C_EVENT_NACK,
	I2C_EVENT_STOPPED,
	I2C_EVENT_AUTO,
	I2C_EVENT_NUM_EVENTS	/* Must be last */
};

#if defined(PIOS_I2C_DIAGNOSTICS)
#define I2C_LOG_DEPTH 20

enum pios_i2c_error_type {
	PIOS_I2C_ERROR_EVENT,
	PIOS_I2C_ERROR_FSM,
	PIOS_I2C_ERROR_INTERRUPT
};

struct pios_i2c_fault_history {
	enum pios_i2c_error_type type;
	uint32_t evirq[I2C_LOG_DEPTH];
	uint32_t erirq[I2C_LOG_DEPTH];
	enum i2c_adapter_event event[I2C_LOG_DEPTH];
	enum i2c_adapter_state state[I2C_LOG_DEPTH];
};
#endif

struct pios_i2c_adapter {
	enum pios_i2c_adapter_magic         magic;
	const struct pios_i2c_adapter_cfg * cfg;

	struct pios_semaphore *sem_busy;
	struct pios_semaphore *sem_ready;

	bool bus_error;
	bool nack;

	volatile enum i2c_adapter_state state;
	const struct pios_i2c_txn *active_txn;
	const struct pios_i2c_txn *last_txn;
	
	uint8_t *active_byte;
	uint8_t *last_byte;

#if defined(PIOS_I2C_DIAGNOSTICS)
	volatile struct pios_i2c_fault_history i2c_adapter_fault_history;

	volatile uint32_t i2c_evirq_history[I2C_LOG_DEPTH];
	volatile uint8_t i2c_evirq_history_pointer;

	volatile uint32_t i2c_erirq_history[I2C_LOG_DEPTH];
	volatile uint8_t i2c_erirq_history_pointer;

	volatile enum i2c_adapter_state i2c_state_history[I2C_LOG_DEPTH];
	volatile uint8_t i2c_state_history_pointer;

	volatile enum i2c_adapter_event i2c_state_event_history[I2C_LOG_DEPTH];
	volatile uint8_t i2c_state_event_history_pointer;

	volatile uint32_t i2c_fsm_fault_count;
	volatile uint32_t i2c_bad_event_counter;
	volatile uint32_t i2c_error_interrupt_counter;
	volatile uint32_t i2c_nack_counter;
	volatile uint32_t i2c_timeout_counter;
#endif
};

int32_t PIOS_I2C_Init(uint32_t * i2c_id, const struct pios_i2c_adapter_cfg * cfg);

#endif /* PIOS_I2C_PRIV_H */
