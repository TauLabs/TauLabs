/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_FSKDAC FSK DAC Functions
 * @brief PIOS interface for FSK DAC implementation
 * @{
 *
 * @file       pios_fskdac.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 * @brief      Generates Bel202 encoded serial data on the DAC channel
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

#if defined(PIOS_INCLUDE_FSK)

#if defined(PIOS_INCLUDE_FREERTOS)
#include "FreeRTOS.h"
#endif /* defined(PIOS_INCLUDE_FREERTOS) */

#include <pios_usart_priv.h>

/* Provide a COM driver */
static void PIOS_FSKDAC_RegisterRxCallback(uintptr_t fskdac_id, pios_com_callback rx_in_cb, uintptr_t context);
static void PIOS_FSKDAC_RegisterTxCallback(uintptr_t fskdac_id, pios_com_callback tx_out_cb, uintptr_t context);
static void PIOS_FSKDAC_TxStart(uintptr_t fskdac_id, uint16_t tx_bytes_avail);
static void PIOS_FSKDAC_RxStart(uintptr_t fskdac_id, uint16_t rx_bytes_avail);

const struct pios_com_driver pios_usart_com_driver = {
	.tx_start   = PIOS_FSKDAC_TxStart,
	.bind_tx_cb = PIOS_FSKDAC_RegisterTxCallback,
};

enum pios_fskdac_dev_magic {
	PIOS_FSKDAC_DEV_MAGIC = 0x1453834A,
};

enum BYTE_TX_STATE = {
	IDLE, START, BIT0, BIT1, BIT2,
	BIT3, BIT4, BIT5, BIT6, BIT7
	STOP
};

struct pios_fskdac_dev {
	enum pios_fskdac_dev_magic     magic;
	const struct pios_fskdac_cfg * cfg;

	//! Track the state of sending an individual bit
	enum BYTE_TX_STATE tx_state;

	pios_com_callback rx_in_cb;
	uintptr_t rx_in_context;
	pios_com_callback tx_out_cb;
	uintptr_t tx_out_context;
};

const uint32_t SAMPLES_PER_BIT = 64;
const uint8_t MARK[SAMPLES_PER_BIT] = {};
const uint8_t SPACE[SAMPLES_PER_BIT] = {};

static bool PIOS_FSKDAC_validate(struct pios_fskdac_dev * fskdac_dev)
{
	return (fskdac_dev->magic == PIOS_FSKDAC_DEV_MAGIC);
}

static struct pios_fskdac_dev * PIOS_FSKDAC_alloc(void)
{
	struct pios_fskdac_dev * fskdac_dev;

	fskdac_dev = (struct pios_fskdac_dev *)PIOS_malloc(sizeof(*fskdac_dev));
	if (!fskdac_dev) return(NULL);

	memset(fskdac_dev, 0, sizeof(*fskdac_dev));
	fskdac_dev->magic = PIOS_FSKDAC_DEV_MAGIC;
	return(fskdac_dev);
}

/**
* Initialise a single USART device
*/
int32_t PIOS_FSKDAC_Init(uintptr_t * fskdac_id, const struct pios_fskdac_cfg * cfg)
{
	PIOS_DEBUG_Assert(usart_id);
	PIOS_DEBUG_Assert(cfg);

	struct pios_fskdac_dev * fskdac_dev;

	fskdac_dev = (struct pios_fskdac_dev *) PIOS_FSKDAC_alloc();
	if (!fskdac_dev) goto out_fail;

	/* Bind the configuration to the device instance */
	fskdac_dev->cfg = cfg;

	// TODO: initialize the DAC hardware

	return(0);

out_fail:
	return(-1);
}


static void PIOS_FSKDAC_TxStart(uintptr_t fskdac_id, uint16_t tx_bytes_avail)
{
	struct pios_fskdac_dev * fskdac_dev = (struct pios_fskdac_dev *)fskdac_id;
	
	bool valid = PIOS_FSKDAC_validate(fskdac_dev);
	PIOS_Assert(valid);
	
	// TODO: equivalent. USART_ITConfig(usart_dev->cfg->regs, USART_IT_TXE, ENABLE);
}

static void PIOS_FSKDAC_RegisterTxCallback(uintptr_t fskdac_id, pios_com_callback tx_out_cb, uintptr_t context)
{
	struct pios_fskdac_dev * fskdac_dev = (struct pios_fskdac_dev *)fskdac_id;

	bool valid = PIOS_FSKDAC_validate(usart_dev);
	PIOS_Assert(valid);
	
	/* 
	 * Order is important in these assignments since ISR uses _cb
	 * field to determine if it's ok to dereference _cb and _context
	 */
	fskdac_dev->tx_out_context = context;
	fskdac_dev->tx_out_cb = tx_out_cb;
}

static void PIOS_FSKDAC_generic_irq_handler(uintptr_t fskdac_id)
{
	struct pios_fskdac_dev * fskdac_dev = (struct pios_fskdac_dev *)fskdac_id;

	bool valid = PIOS_FSKDAC_validate(usart_dev);
	PIOS_Assert(valid);
	
	/* Check if TXE flag is set */
	bool tx_need_yield = false;
	if (sr & USART_SR_TXE) {
		if (usart_dev->tx_out_cb) {
			uint8_t b;
			uint16_t bytes_to_send;
			
			bytes_to_send = (usart_dev->tx_out_cb)(usart_dev->tx_out_context, &b, 1, NULL, &tx_need_yield);
			
			if (bytes_to_send > 0) {
				/* Send the byte we've been given */
				usart_dev->cfg->regs->DR = b;
			} else {
				/* No bytes to send, disable TXE interrupt */
				USART_ITConfig(usart_dev->cfg->regs, USART_IT_TXE, DISABLE);
			}
		} else {
			/* No bytes to send, disable TXE interrupt */
			USART_ITConfig(usart_dev->cfg->regs, USART_IT_TXE, DISABLE);
		}
	}
	
#if defined(PIOS_INCLUDE_FREERTOS)
	portEND_SWITCHING_ISR((rx_need_yield || tx_need_yield) ? pdTRUE : pdFALSE);
#endif	/* defined(PIOS_INCLUDE_FREERTOS) */
}

#endif /* PIOS_INCLUDE_FSK */

/**
  * @}
  * @}
  */
