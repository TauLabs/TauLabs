/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_I2C I2C Functions
 * @brief STM32F4xx Hardware dependent I2C functionality
 * @{
 *
 * @file       pios_i2c.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      I2C Enable/Disable routines
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

/* Project Includes */
#include "pios.h"

#if defined(PIOS_INCLUDE_I2C)

#if defined(PIOS_INCLUDE_FREERTOS)
#define USE_FREERTOS
#endif

#include <pios_i2c_priv.h>

/* dirty hack to keep compatible with enum in pios_i2c_priv.h */
#define I2C_STATE_WRITE_BYTE				I2C_STATE_W_MORE_TXN_ADDR
#define I2C_STATE_READ_BYTE					I2C_STATE_W_MORE_TXN_MIDDLE
#define I2C_STATE_TRANSFER_COMPLETE			I2C_STATE_W_MORE_TXN_LAST

//#define I2C_HALT_ON_ERRORS

static void go_fsm_fault(struct pios_i2c_adapter *i2c_adapter, bool *woken);
static void go_bus_error(struct pios_i2c_adapter *i2c_adapter, bool *woken);
static void go_stopped(struct pios_i2c_adapter *i2c_adapter, bool *woken);
static void go_starting(struct pios_i2c_adapter *i2c_adapter, bool *woken);
static void go_write_byte(struct pios_i2c_adapter *i2c_adapter, bool *woken);
static void go_read_byte(struct pios_i2c_adapter *i2c_adapter, bool *woken);
static void go_transfer_complete(struct pios_i2c_adapter *i2c_adapter, bool *woken);
static void go_nack(struct pios_i2c_adapter *i2c_adapter, bool *woken);

struct i2c_adapter_transition {
	void (*entry_fn)(struct pios_i2c_adapter *i2c_adapter, bool *woken);
	enum i2c_adapter_state next_state[I2C_EVENT_NUM_EVENTS];
};

static void i2c_adapter_process_auto(struct pios_i2c_adapter *i2c_adapter, bool *woken);
static void i2c_adapter_inject_event(struct pios_i2c_adapter *i2c_adapter, enum i2c_adapter_event event, bool *woken);
static void i2c_adapter_fsm_init(struct pios_i2c_adapter *i2c_adapter);
static void i2c_adapter_reset_bus(struct pios_i2c_adapter *i2c_adapter);
#ifndef USE_FREERTOS
static bool i2c_adapter_fsm_terminated(struct pios_i2c_adapter *i2c_adapter);
#endif

#if defined(PIOS_I2C_DIAGNOSTICS)
static void i2c_adapter_log_fault(struct pios_i2c_adapter *i2c_adapter, enum pios_i2c_error_type type);
#endif

const static struct i2c_adapter_transition i2c_adapter_transitions[I2C_STATE_NUM_STATES] = {
	[I2C_STATE_FSM_FAULT] = {
		.entry_fn = go_fsm_fault,
		.next_state = {
			[I2C_EVENT_AUTO] = I2C_STATE_STOPPED,
		},
	},
	[I2C_STATE_BUS_ERROR] = {
		.entry_fn = go_bus_error,
		.next_state = {
			[I2C_EVENT_AUTO] = I2C_STATE_STOPPED,
		},
	},

	[I2C_STATE_STOPPED] = {
		.entry_fn = go_stopped,
		.next_state = {
			[I2C_EVENT_START] = I2C_STATE_STARTING,
			[I2C_EVENT_BUS_ERROR] = I2C_STATE_BUS_ERROR,
		},
	},

	[I2C_STATE_STARTING] = {
		.entry_fn = go_starting,
		.next_state = {
			[I2C_EVENT_TRANSMIT_BUFFER_EMPTY] = I2C_STATE_WRITE_BYTE,
			[I2C_EVENT_RECEIVER_BUFFER_NOT_EMPTY] = I2C_STATE_READ_BYTE,
			[I2C_EVENT_NACK] = I2C_STATE_NACK,
			[I2C_EVENT_BUS_ERROR] = I2C_STATE_BUS_ERROR,
		},
	},

	[I2C_STATE_WRITE_BYTE] = {
		.entry_fn = go_write_byte,
		.next_state = {
			[I2C_EVENT_TRANSMIT_BUFFER_EMPTY] = I2C_STATE_WRITE_BYTE,
			[I2C_EVENT_RECEIVER_BUFFER_NOT_EMPTY] = I2C_STATE_READ_BYTE,
			[I2C_EVENT_TRANSFER_COMPLETE] = I2C_STATE_TRANSFER_COMPLETE,
			[I2C_EVENT_STOP] = I2C_STATE_STOPPED,
			[I2C_EVENT_BUS_ERROR] = I2C_STATE_BUS_ERROR,
		},
	},

	[I2C_STATE_READ_BYTE] = {
		.entry_fn = go_read_byte,
		.next_state = {
			[I2C_EVENT_TRANSMIT_BUFFER_EMPTY] = I2C_STATE_WRITE_BYTE,
			[I2C_EVENT_RECEIVER_BUFFER_NOT_EMPTY] = I2C_STATE_READ_BYTE,
			[I2C_EVENT_TRANSFER_COMPLETE] = I2C_STATE_TRANSFER_COMPLETE,
			[I2C_EVENT_STOP] = I2C_STATE_STOPPED,
			[I2C_EVENT_BUS_ERROR] = I2C_STATE_BUS_ERROR,
		},
	},

	[I2C_STATE_TRANSFER_COMPLETE] = {
		.entry_fn = go_transfer_complete,
		.next_state = {
			[I2C_EVENT_AUTO] = I2C_STATE_STARTING,
			[I2C_EVENT_BUS_ERROR] = I2C_STATE_BUS_ERROR,
		},
	},

	[I2C_STATE_NACK] = {
		.entry_fn = go_nack,
		.next_state = {
			[I2C_EVENT_AUTO] = I2C_STATE_STOPPED,
		},
	},
};

static void go_fsm_fault(struct pios_i2c_adapter *i2c_adapter, bool *woken)
{
#if defined(I2C_HALT_ON_ERRORS)
	PIOS_DEBUG_Assert(0);
#endif
	/* Note that this transfer has hit a bus error */
	i2c_adapter->bus_error = true;

	i2c_adapter_reset_bus(i2c_adapter);

}

static void go_bus_error(struct pios_i2c_adapter *i2c_adapter, bool *woken)
{
	/* Note that this transfer has hit a bus error */
	i2c_adapter->bus_error = true;

	i2c_adapter_reset_bus(i2c_adapter);
}

static void go_stopped(struct pios_i2c_adapter *i2c_adapter, bool *woken)
{
	I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_ERRI | I2C_IT_TCI | I2C_IT_NACKI | I2C_IT_RXI | I2C_IT_STOPI | I2C_IT_TXI, DISABLE);

	/* wake up blocked PIOS_I2C_Transfer() */
	if (PIOS_Semaphore_Give_FromISR(i2c_adapter->sem_ready, woken) == false) {
#if defined(I2C_HALT_ON_ERRORS)
		PIOS_DEBUG_Assert(0);
#endif
	}
}

static void go_starting(struct pios_i2c_adapter *i2c_adapter, bool *woken)
{
	PIOS_DEBUG_Assert(i2c_adapter->active_txn);
	PIOS_DEBUG_Assert(i2c_adapter->active_txn <= i2c_adapter->last_txn);
	PIOS_DEBUG_Assert(i2c_adapter->active_txn->len <= 255);	//FIXME: implement this using TCR

	i2c_adapter->active_byte = &(i2c_adapter->active_txn->buf[0]);
	i2c_adapter->last_byte = &(i2c_adapter->active_txn->buf[i2c_adapter->active_txn->len - 1]);

	I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_ERRI | I2C_IT_TCI | I2C_IT_NACKI | I2C_IT_RXI | I2C_IT_STOPI | I2C_IT_TXI, ENABLE);

	I2C_TransferHandling(
		i2c_adapter->cfg->regs,
		(i2c_adapter->active_txn->addr << 1),
		i2c_adapter->active_txn->len,
		i2c_adapter->active_txn == i2c_adapter->last_txn ? I2C_AutoEnd_Mode : I2C_SoftEnd_Mode,
		i2c_adapter->active_txn->rw == PIOS_I2C_TXN_WRITE ? I2C_Generate_Start_Write : I2C_Generate_Start_Read);
}

static void go_write_byte(struct pios_i2c_adapter *i2c_adapter, bool *woken)
{
	PIOS_DEBUG_Assert(i2c_adapter->active_txn);
	PIOS_DEBUG_Assert(i2c_adapter->active_txn <= i2c_adapter->last_txn);

	I2C_SendData(i2c_adapter->cfg->regs, *(i2c_adapter->active_byte));

	/* Move to the next byte */
	i2c_adapter->active_byte++;
	PIOS_DEBUG_Assert(i2c_adapter->active_byte <= i2c_adapter->last_byte);
}

static void go_read_byte(struct pios_i2c_adapter *i2c_adapter, bool *woken)
{
	PIOS_DEBUG_Assert(i2c_adapter->active_txn);
	PIOS_DEBUG_Assert(i2c_adapter->active_txn <= i2c_adapter->last_txn);

	*(i2c_adapter->active_byte) = I2C_ReceiveData(i2c_adapter->cfg->regs);

	/* Move to the next byte */
	i2c_adapter->active_byte++;
	PIOS_DEBUG_Assert(i2c_adapter->active_byte <= i2c_adapter->last_byte);
}

static void go_transfer_complete(struct pios_i2c_adapter *i2c_adapter, bool *woken)
{
	/* Move to the next transaction */
	i2c_adapter->active_txn++;
	PIOS_DEBUG_Assert(i2c_adapter->active_txn <= i2c_adapter->last_txn);
}

static void go_nack(struct pios_i2c_adapter *i2c_adapter, bool *woken)
{
	i2c_adapter->nack = true;
	I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_ERRI | I2C_IT_TCI | I2C_IT_NACKI | I2C_IT_RXI | I2C_IT_STOPI | I2C_IT_TXI, DISABLE);
	I2C_AcknowledgeConfig(i2c_adapter->cfg->regs, DISABLE);
	I2C_GenerateSTOP(i2c_adapter->cfg->regs, ENABLE);
}

static void i2c_adapter_inject_event(struct pios_i2c_adapter *i2c_adapter, enum i2c_adapter_event event, bool *woken)
{
#if defined(PIOS_I2C_DIAGNOSTICS)
	i2c_adapter->i2c_state_event_history[i2c_adapter->i2c_state_event_history_pointer] = event;
	i2c_adapter->i2c_state_event_history_pointer = (i2c_adapter->i2c_state_event_history_pointer + 1) % I2C_LOG_DEPTH;

	i2c_adapter->i2c_state_history[i2c_adapter->i2c_state_history_pointer] = i2c_adapter->curr_state;
	i2c_adapter->i2c_state_history_pointer = (i2c_adapter->i2c_state_history_pointer + 1) % I2C_LOG_DEPTH;

	if (i2c_adapter_transitions[i2c_adapter->curr_state].next_state[event] == I2C_STATE_FSM_FAULT)
		i2c_adapter_log_fault(i2c_adapter, PIOS_I2C_ERROR_FSM);
#endif
	/*
	 * Move to the next state
	 *
	 * This is done prior to calling the new state's entry function to
	 * guarantee that the entry function never depends on the previous
	 * state.  This way, it cannot ever know what the previous state was.
	 */
	enum i2c_adapter_state prev_state = i2c_adapter->curr_state;
	if (prev_state);

	i2c_adapter->curr_state = i2c_adapter_transitions[i2c_adapter->curr_state].next_state[event];

	/* Call the entry function (if any) for the next state. */
	if (i2c_adapter_transitions[i2c_adapter->curr_state].entry_fn) {
		i2c_adapter_transitions[i2c_adapter->curr_state].entry_fn(i2c_adapter, woken);
	}

	/* Process any AUTO transitions in the FSM */
	i2c_adapter_process_auto(i2c_adapter, woken);
}

static void i2c_adapter_process_auto(struct pios_i2c_adapter *i2c_adapter, bool *woken)
{
	enum i2c_adapter_state prev_state = i2c_adapter->curr_state;
	if (prev_state);

	while (i2c_adapter_transitions[i2c_adapter->curr_state].next_state[I2C_EVENT_AUTO]) {
		i2c_adapter->curr_state = i2c_adapter_transitions[i2c_adapter->curr_state].next_state[I2C_EVENT_AUTO];

		/* Call the entry function (if any) for the next state. */
		if (i2c_adapter_transitions[i2c_adapter->curr_state].entry_fn) {
			i2c_adapter_transitions[i2c_adapter->curr_state].entry_fn(i2c_adapter, woken);
		}
	}
}

static void i2c_adapter_fsm_init(struct pios_i2c_adapter *i2c_adapter)
{
	i2c_adapter_reset_bus(i2c_adapter);
	i2c_adapter->curr_state = I2C_STATE_STOPPED;
}

static void i2c_adapter_reset_bus(struct pios_i2c_adapter *i2c_adapter)
{
	/* Reset the I2C block */
	I2C_DeInit(i2c_adapter->cfg->regs);

	/* Make sure the bus is free by clocking it until any slaves release the bus. */
	GPIO_InitTypeDef scl_gpio_init;
	scl_gpio_init = i2c_adapter->cfg->scl.init;
	scl_gpio_init.GPIO_Mode  = GPIO_Mode_OUT;
	GPIO_SetBits(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin);
	GPIO_Init(i2c_adapter->cfg->scl.gpio, &scl_gpio_init);

	GPIO_InitTypeDef sda_gpio_init;
	sda_gpio_init = i2c_adapter->cfg->sda.init;
	sda_gpio_init.GPIO_Mode  = GPIO_Mode_OUT;
	GPIO_SetBits(i2c_adapter->cfg->sda.gpio, i2c_adapter->cfg->sda.init.GPIO_Pin);
	GPIO_Init(i2c_adapter->cfg->sda.gpio, &sda_gpio_init);

	/* Check SDA line to determine if slave is asserting bus and clock out if so, this may  */
	/* have to be repeated (due to further bus errors) but better than clocking 0xFF into an */
	/* ESC */
	//bool sda_hung = GPIO_ReadInputDataBit(i2c_adapter->cfg->sda.gpio, i2c_adapter->cfg->sda.init.GPIO_Pin) == Bit_RESET;
	while (GPIO_ReadInputDataBit(i2c_adapter->cfg->sda.gpio, i2c_adapter->cfg->sda.init.GPIO_Pin) == Bit_RESET) {

		/* Set clock high and wait for any clock stretching to finish. */
		GPIO_SetBits(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin);
		while (GPIO_ReadInputDataBit(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin) == Bit_RESET);
		PIOS_DELAY_WaituS(2);

		/* Set clock low */
		GPIO_ResetBits(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin);
		PIOS_DELAY_WaituS(2);

		/* Clock high again */
		GPIO_SetBits(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin);
		PIOS_DELAY_WaituS(2);
	}

	/* Generate a start then stop condition */
	GPIO_SetBits(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin);
	PIOS_DELAY_WaituS(2);
	GPIO_ResetBits(i2c_adapter->cfg->sda.gpio, i2c_adapter->cfg->sda.init.GPIO_Pin);
	PIOS_DELAY_WaituS(2);
	GPIO_ResetBits(i2c_adapter->cfg->sda.gpio, i2c_adapter->cfg->sda.init.GPIO_Pin);
	PIOS_DELAY_WaituS(2);

	/* Set data and clock high and wait for any clock stretching to finish. */
	GPIO_SetBits(i2c_adapter->cfg->sda.gpio, i2c_adapter->cfg->sda.init.GPIO_Pin);
	GPIO_SetBits(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin);
	while (GPIO_ReadInputDataBit(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin) == Bit_RESET);
	/* Wait for data to be high */
	while (GPIO_ReadInputDataBit(i2c_adapter->cfg->sda.gpio, i2c_adapter->cfg->sda.init.GPIO_Pin) != Bit_SET);


	/* Bus signals are guaranteed to be high (ie. free) after this point */
	/* Initialize the GPIO pins to the peripheral function */
	if (i2c_adapter->cfg->remap) {
		GPIO_PinAFConfig(i2c_adapter->cfg->scl.gpio,
				__builtin_ctz(i2c_adapter->cfg->scl.init.GPIO_Pin),
				i2c_adapter->cfg->remap);
		GPIO_PinAFConfig(i2c_adapter->cfg->sda.gpio,
				__builtin_ctz(i2c_adapter->cfg->sda.init.GPIO_Pin),
				i2c_adapter->cfg->remap);
	}
	GPIO_Init(i2c_adapter->cfg->scl.gpio, (GPIO_InitTypeDef *) & (i2c_adapter->cfg->scl.init)); // Struct is const, function signature not
	GPIO_Init(i2c_adapter->cfg->sda.gpio, (GPIO_InitTypeDef *) & (i2c_adapter->cfg->sda.init));

	/* Reset the I2C block */
	I2C_DeInit(i2c_adapter->cfg->regs);

	/* Initialize the I2C block */
	I2C_Init(i2c_adapter->cfg->regs, (I2C_InitTypeDef *) & (i2c_adapter->cfg->init));

	/* Enable the I2C block */
	I2C_Cmd(i2c_adapter->cfg->regs, ENABLE);

	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_BUSY))
		I2C_SoftwareResetCmd(i2c_adapter->cfg->regs);
}

#ifndef USE_FREERTOS
/* Return true if the FSM is in a terminal state */
static bool i2c_adapter_fsm_terminated(struct pios_i2c_adapter *i2c_adapter)
{
	switch (i2c_adapter->curr_state) {
	case I2C_STATE_STOPPING:
	case I2C_STATE_STOPPED:
		return (true);
	default:
		return (false);
	}
}
#endif

/**
 * Logs the last N state transitions and N IRQ events due to
 * an error condition
 * \param[in] i2c the adapter number to log an event for
 */
#if defined(PIOS_I2C_DIAGNOSTICS)
void i2c_adapter_log_fault(struct pios_i2c_adapter *i2c_adapter, enum pios_i2c_error_type type)
{
	i2c_adapter->i2c_adapter_fault_history.type = type;
	for (uint8_t i = 0; i < I2C_LOG_DEPTH; i++) {
		i2c_adapter->i2c_adapter_fault_history.evirq[i] =
				i2c_adapter->i2c_evirq_history[(I2C_LOG_DEPTH + i2c_adapter->i2c_evirq_history_pointer - 1 - i) % I2C_LOG_DEPTH];
		i2c_adapter->i2c_adapter_fault_history.erirq[i] =
				i2c_adapter->i2c_erirq_history[(I2C_LOG_DEPTH + i2c_adapter->i2c_erirq_history_pointer - 1 - i) % I2C_LOG_DEPTH];
		i2c_adapter->i2c_adapter_fault_history.event[i] =
				i2c_adapter->i2c_state_event_history[(I2C_LOG_DEPTH + i2c_adapter->i2c_state_event_history_pointer - 1 - i) % I2C_LOG_DEPTH];
		i2c_adapter->i2c_adapter_fault_history.state[i] =
				i2c_adapter->i2c_state_history[(I2C_LOG_DEPTH + i2c_adapter->i2c_state_history_pointer - 1 - i) % I2C_LOG_DEPTH];
	}
	switch (type) {
	case PIOS_I2C_ERROR_EVENT:
		i2c_adapter->i2c_bad_event_counter++;
		break;
	case PIOS_I2C_ERROR_FSM:
		i2c_adapter->i2c_fsm_fault_count++;
		break;
	case PIOS_I2C_ERROR_INTERRUPT:
		i2c_adapter->i2c_error_interrupt_counter++;
		break;
	}
}
#endif

static bool PIOS_I2C_validate(struct pios_i2c_adapter *i2c_adapter)
{
	return (i2c_adapter->magic == PIOS_I2C_DEV_MAGIC);
}

#if defined(PIOS_INCLUDE_FREERTOS) && 0
static struct pios_i2c_dev *PIOS_I2C_alloc(void)
{
	struct pios_i2c_dev *i2c_adapter;

	i2c_adapter = (struct pios_i2c_adapter *)PIOS_malloc(sizeof(*i2c_adapter));
	if (!i2c_adapter) return (NULL);

	i2c_adapter->magic = PIOS_I2C_DEV_MAGIC;
	return (i2c_adapter);
}
#else
static struct pios_i2c_adapter pios_i2c_adapters[PIOS_I2C_MAX_DEVS];
static uint8_t pios_i2c_num_adapters;
static struct pios_i2c_adapter *PIOS_I2C_alloc(void)
{
	struct pios_i2c_adapter *i2c_adapter;

	if (pios_i2c_num_adapters >= PIOS_I2C_MAX_DEVS) {
		return (NULL);
	}

	i2c_adapter = &pios_i2c_adapters[pios_i2c_num_adapters++];
	i2c_adapter->magic = PIOS_I2C_DEV_MAGIC;

	return (i2c_adapter);
}
#endif


/**
* Initializes IIC driver
* \param[in] mode currently only mode 0 supported
* \return < 0 if initialisation failed
*/
int32_t PIOS_I2C_Init(uint32_t *i2c_id, const struct pios_i2c_adapter_cfg *cfg)
{
	PIOS_DEBUG_Assert(i2c_id);
	PIOS_DEBUG_Assert(cfg);

	struct pios_i2c_adapter *i2c_adapter;

	i2c_adapter = (struct pios_i2c_adapter *) PIOS_I2C_alloc();
	if (!i2c_adapter)
		goto out_fail;

	/* Bind the configuration to the device instance */
	i2c_adapter->cfg = cfg;

	i2c_adapter->sem_ready = PIOS_Semaphore_Create();
	i2c_adapter->lock = PIOS_Mutex_Create();

	/* Initialize the state machine */
	i2c_adapter_fsm_init(i2c_adapter);

	*i2c_id = (uint32_t)i2c_adapter;

	/* Configure and enable I2C interrupts */
	NVIC_Init((NVIC_InitTypeDef *) & (i2c_adapter->cfg->event.init));
	NVIC_Init((NVIC_InitTypeDef *) & (i2c_adapter->cfg->error.init));

	/* No error */
	return 0;

out_fail:
	return -1;
}

/**
 * @brief Check the I2C bus is clear and in a properly reset state
 * @returns  0 Bus is clear
 * @returns -1 Bus is in use
 * @returns -2 Bus not clear
 */
int32_t PIOS_I2C_CheckClear(uint32_t i2c_id)
{
	struct pios_i2c_adapter *i2c_adapter = (struct pios_i2c_adapter *)i2c_id;

	bool valid = PIOS_I2C_validate(i2c_adapter);
	PIOS_Assert(valid)

	if (PIOS_Mutex_Lock(i2c_adapter->lock, i2c_adapter->cfg->transfer_timeout_ms) == false)
		return -1;

	if (i2c_adapter->curr_state != I2C_STATE_STOPPED) {
		PIOS_Mutex_Unlock(i2c_adapter->lock);
		return -2;
	}

	if (GPIO_ReadInputDataBit(i2c_adapter->cfg->sda.gpio, i2c_adapter->cfg->sda.init.GPIO_Pin) == Bit_RESET ||
		GPIO_ReadInputDataBit(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin) == Bit_RESET) {
		PIOS_Mutex_Unlock(i2c_adapter->lock);
		return -3;
	}

	PIOS_Mutex_Unlock(i2c_adapter->lock);

	return 0;
}

int32_t PIOS_I2C_Transfer(uint32_t i2c_id, const struct pios_i2c_txn txn_list[], uint32_t num_txns)
{
	//FIXME: only supports transfer sizes up to 255 bytes
	struct pios_i2c_adapter *i2c_adapter = (struct pios_i2c_adapter *)i2c_id;

	bool valid = PIOS_I2C_validate(i2c_adapter);
	PIOS_Assert(valid)

	PIOS_DEBUG_Assert(txn_list);
	PIOS_DEBUG_Assert(num_txns);

	if (PIOS_Mutex_Lock(i2c_adapter->lock, i2c_adapter->cfg->transfer_timeout_ms) == false)
		return -2;

	PIOS_DEBUG_Assert(i2c_adapter->curr_state == I2C_STATE_STOPPED);

	i2c_adapter->last_txn = &txn_list[num_txns - 1];
	i2c_adapter->active_txn = &txn_list[0];
	i2c_adapter->bus_error = false;
	i2c_adapter->nack = false;

	/* Make sure the done/ready semaphore is consumed before we start */
	PIOS_Semaphore_Take(i2c_adapter->sem_ready, 0);

	bool dummy = false;
	i2c_adapter_inject_event(i2c_adapter, I2C_EVENT_START, &dummy);

	/* Wait for the transfer to complete */
	bool semaphore_success = (PIOS_Semaphore_Take(i2c_adapter->sem_ready, i2c_adapter->cfg->transfer_timeout_ms) == true);

#if defined(PIOS_I2C_DIAGNOSTICS)
	if (!semaphore_success)
		i2c_adapter->i2c_timeout_counter++;
#endif

	int32_t result = !semaphore_success ? -2 :
			i2c_adapter->bus_error ? -1 :
			i2c_adapter->nack ? -3 :
			0;

	PIOS_Mutex_Unlock(i2c_adapter->lock);

	return result;
}

void PIOS_I2C_EV_IRQ_Handler(uint32_t i2c_id)
{
	struct pios_i2c_adapter *i2c_adapter = (struct pios_i2c_adapter *)i2c_id;

	bool valid = PIOS_I2C_validate(i2c_adapter);
	PIOS_Assert(valid)

	bool woken = false;

	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_RXNE)) {
		//flag will be cleared by event
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_evirq_history[i2c_adapter->i2c_evirq_history_pointer] = I2C_FLAG_RXNE;
		i2c_adapter->i2c_evirq_history_pointer = (i2c_adapter->i2c_evirq_history_pointer + 1) % I2C_LOG_DEPTH;
#endif
		i2c_adapter_inject_event(i2c_adapter, I2C_EVENT_RECEIVER_BUFFER_NOT_EMPTY, &woken);
	}

	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_TXIS)) {
		//flag will be cleared by event
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_evirq_history[i2c_adapter->i2c_evirq_history_pointer] = I2C_FLAG_TXIS;
		i2c_adapter->i2c_evirq_history_pointer = (i2c_adapter->i2c_evirq_history_pointer + 1) % I2C_LOG_DEPTH;
#endif
		i2c_adapter_inject_event(i2c_adapter, I2C_EVENT_TRANSMIT_BUFFER_EMPTY, &woken);
	}

	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_NACKF)) {
		I2C_ClearFlag(i2c_adapter->cfg->regs, I2C_FLAG_NACKF);
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_evirq_history[i2c_adapter->i2c_evirq_history_pointer] = I2C_FLAG_NACKF;
		i2c_adapter->i2c_evirq_history_pointer = (i2c_adapter->i2c_evirq_history_pointer + 1) % I2C_LOG_DEPTH;
		++i2c_adapter->i2c_nack_counter;
#endif
		i2c_adapter_inject_event(i2c_adapter, I2C_EVENT_NACK, &woken);
	}

	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_TC)) {
		I2C_ClearFlag(i2c_adapter->cfg->regs, I2C_FLAG_TC);
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_evirq_history[i2c_adapter->i2c_evirq_history_pointer] = I2C_FLAG_TC;
		i2c_adapter->i2c_evirq_history_pointer = (i2c_adapter->i2c_evirq_history_pointer + 1) % I2C_LOG_DEPTH;
#endif
		i2c_adapter_inject_event(i2c_adapter, I2C_EVENT_TRANSFER_COMPLETE, &woken);
	}

	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_STOPF)) {
		I2C_ClearFlag(i2c_adapter->cfg->regs, I2C_FLAG_STOPF);
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_evirq_history[i2c_adapter->i2c_evirq_history_pointer] = I2C_FLAG_STOPF;
		i2c_adapter->i2c_evirq_history_pointer = (i2c_adapter->i2c_evirq_history_pointer + 1) % I2C_LOG_DEPTH;
#endif
		i2c_adapter_inject_event(i2c_adapter, I2C_EVENT_STOP, &woken);
	}

#ifdef USE_FREERTOS
	portEND_SWITCHING_ISR(woken == true ? pdTRUE : pdFALSE);
#endif
}


void PIOS_I2C_ER_IRQ_Handler(uint32_t i2c_id)
{
	struct pios_i2c_adapter *i2c_adapter = (struct pios_i2c_adapter *)i2c_id;

	bool valid = PIOS_I2C_validate(i2c_adapter);
	PIOS_Assert(valid)

	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_BERR)) {
		I2C_ClearFlag(i2c_adapter->cfg->regs, I2C_FLAG_BERR);
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_erirq_history[i2c_adapter->i2c_erirq_history_pointer] = I2C_FLAG_BERR;
		i2c_adapter->i2c_erirq_history_pointer = (i2c_adapter->i2c_erirq_history_pointer + 1) % I2C_LOG_DEPTH;
#endif
	}
	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_ARLO)) {
		I2C_ClearFlag(i2c_adapter->cfg->regs, I2C_FLAG_ARLO);
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_erirq_history[i2c_adapter->i2c_erirq_history_pointer] = I2C_FLAG_ARLO;
		i2c_adapter->i2c_erirq_history_pointer = (i2c_adapter->i2c_erirq_history_pointer + 1) % I2C_LOG_DEPTH;
#endif
	}
	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_OVR)) {
		I2C_ClearFlag(i2c_adapter->cfg->regs, I2C_FLAG_OVR);
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_erirq_history[i2c_adapter->i2c_erirq_history_pointer] = I2C_FLAG_OVR;
		i2c_adapter->i2c_erirq_history_pointer = (i2c_adapter->i2c_erirq_history_pointer + 1) % I2C_LOG_DEPTH;
#endif
	}
	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_PECERR)) {
		I2C_ClearFlag(i2c_adapter->cfg->regs, I2C_FLAG_PECERR);
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_erirq_history[i2c_adapter->i2c_erirq_history_pointer] = I2C_FLAG_PECERR;
		i2c_adapter->i2c_erirq_history_pointer = (i2c_adapter->i2c_erirq_history_pointer + 1) % I2C_LOG_DEPTH;
#endif
	}
	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_TIMEOUT)) {
		I2C_ClearFlag(i2c_adapter->cfg->regs, I2C_FLAG_TIMEOUT);
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_erirq_history[i2c_adapter->i2c_erirq_history_pointer] = I2C_FLAG_TIMEOUT;
		i2c_adapter->i2c_erirq_history_pointer = (i2c_adapter->i2c_erirq_history_pointer + 1) % I2C_LOG_DEPTH;
#endif
	}
	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_ALERT)) {
		I2C_ClearFlag(i2c_adapter->cfg->regs, I2C_FLAG_ALERT);
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_erirq_history[i2c_adapter->i2c_erirq_history_pointer] = I2C_FLAG_ALERT;
		i2c_adapter->i2c_erirq_history_pointer = (i2c_adapter->i2c_erirq_history_pointer + 1) % I2C_LOG_DEPTH;
#endif
	}
	if (I2C_GetFlagStatus(i2c_adapter->cfg->regs, I2C_FLAG_BUSY)) {
		I2C_ClearFlag(i2c_adapter->cfg->regs, I2C_FLAG_BUSY);
#if defined(PIOS_I2C_DIAGNOSTICS)
		i2c_adapter->i2c_erirq_history[i2c_adapter->i2c_erirq_history_pointer] = I2C_FLAG_BUSY;
		i2c_adapter->i2c_erirq_history_pointer = (i2c_adapter->i2c_erirq_history_pointer + 1) % I2C_LOG_DEPTH;
#endif
	}

#if defined(PIOS_I2C_DIAGNOSTICS)
	i2c_adapter_log_fault(i2c_adapter, PIOS_I2C_ERROR_INTERRUPT);
#endif

	/* Fail hard on any errors for now */
	bool woken = false;
	i2c_adapter_inject_event(i2c_adapter, I2C_EVENT_BUS_ERROR, &woken);

#ifdef USE_FREERTOS
	portEND_SWITCHING_ISR(woken == true ? pdTRUE : pdFALSE);
#endif
}

#endif

/**
  * @}
  * @}
  */
