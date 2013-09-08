/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_CAN PiOS CAN interface layer
 * @brief CAN interface for PiOS
 * @{
 *
 * @file       pios_can.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      PiOS CAN interface header
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


#include "pios.h"

#if defined(PIOS_INCLUDE_CAN)

#include "pios_can_priv.h"

/* Provide a COM driver */
static void PIOS_CAN_RegisterRxCallback(uintptr_t can_id, pios_com_callback rx_in_cb, uintptr_t context);
static void PIOS_CAN_RegisterTxCallback(uintptr_t can_id, pios_com_callback tx_out_cb, uintptr_t context);
static void PIOS_CAN_TxStart(uintptr_t can_id, uint16_t tx_bytes_avail);
static void PIOS_CAN_RxStart(uintptr_t can_id, uint16_t rx_bytes_avail);

const struct pios_com_driver pios_can_com_driver = {
	.tx_start   = PIOS_CAN_TxStart,
	.rx_start   = PIOS_CAN_RxStart,
	.bind_tx_cb = PIOS_CAN_RegisterTxCallback,
	.bind_rx_cb = PIOS_CAN_RegisterRxCallback,
};

enum pios_can_dev_magic {
	PIOS_CAN_DEV_MAGIC = 0x41fa834A,
};

//! Structure for an initialized CAN handle
struct pios_can_dev {
	enum pios_can_dev_magic     magic;
	const struct pios_can_cfg  *cfg;
	pios_com_callback rx_in_cb;
	uintptr_t rx_in_context;
	pios_com_callback tx_out_cb;
	uintptr_t tx_out_context;
};

// Local constants
#define CAN_COM_ID      0x11
#define MAX_SEND_LEN    8

void USB_HP_CAN1_TX_IRQHandler(void);

static bool PIOS_CAN_validate(struct pios_can_dev *can_dev)
{
	return (can_dev->magic == PIOS_CAN_DEV_MAGIC);
}

#if !defined(PIOS_INCLUDE_FREERTOS)
#error PIOS_CAN REQUIRES FREERTOS
#endif

static struct pios_can_dev *PIOS_CAN_alloc(void)
{
	struct pios_can_dev *can_dev;

	can_dev = (struct pios_can_dev *)PIOS_malloc(sizeof(*can_dev));
	if (!can_dev) return(NULL);

	memset(can_dev, 0, sizeof(*can_dev));
	can_dev->magic = PIOS_CAN_DEV_MAGIC;

	return(can_dev);
}

//! The local handle for the CAN device
static struct pios_can_dev *can_dev;

/**
 * Initialize the CAN driver and return an opaque id
 * @param[out]   id the CAN interface handle
 * @param[in]    cfg the configuration structure
 * @return 0 if successful, negative otherwise
 */
int32_t PIOS_CAN_Init(uintptr_t *can_id, const struct pios_can_cfg *cfg)
{
	PIOS_DEBUG_Assert(can_id);
	PIOS_DEBUG_Assert(cfg);

	can_dev = (struct pios_can_dev *) PIOS_CAN_alloc();
	if (!can_dev) goto out_fail;

	/* Bind the configuration to the device instance */
	can_dev->cfg = cfg;

	/* Configure the CAN device */
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_CAN1, ENABLE);

	/* Map pins to CAN function */
	if (can_dev->cfg->remap) {
		if (can_dev->cfg->rx.gpio != 0)
			GPIO_PinAFConfig(can_dev->cfg->rx.gpio,
				can_dev->cfg->rx.pin_source,
				can_dev->cfg->remap);
		if (can_dev->cfg->tx.gpio != 0)
			GPIO_PinAFConfig(can_dev->cfg->tx.gpio,
				can_dev->cfg->tx.pin_source,
				can_dev->cfg->remap);
	}

	/* Initialize the CAN Rx and Tx pins */
	if (can_dev->cfg->rx.gpio != 0)
		GPIO_Init(can_dev->cfg->rx.gpio, (GPIO_InitTypeDef *)&can_dev->cfg->rx.init);
	if (can_dev->cfg->tx.gpio != 0)
		GPIO_Init(can_dev->cfg->tx.gpio, (GPIO_InitTypeDef *)&can_dev->cfg->tx.init);

	*can_id = (uintptr_t)can_dev;

	CAN_DeInit(can_dev->cfg->regs);
	CAN_Init(can_dev->cfg->regs, (CAN_InitTypeDef *)&can_dev->cfg->init);

	/* CAN filter init */
	CAN_FilterInitTypeDef CAN_FilterInitStructure;
	CAN_FilterInitStructure.CAN_FilterNumber = 0;
	CAN_FilterInitStructure.CAN_FilterMode = CAN_FilterMode_IdMask;
	CAN_FilterInitStructure.CAN_FilterScale = CAN_FilterScale_32bit;
	CAN_FilterInitStructure.CAN_FilterIdHigh = 0x0000;
	CAN_FilterInitStructure.CAN_FilterIdLow = 0x0000;
	CAN_FilterInitStructure.CAN_FilterMaskIdHigh = 0x0000;
	CAN_FilterInitStructure.CAN_FilterMaskIdLow = 0x0000;  
	CAN_FilterInitStructure.CAN_FilterFIFOAssignment = 1;

	CAN_FilterInitStructure.CAN_FilterActivation = ENABLE;
	CAN_FilterInit(&CAN_FilterInitStructure);

	// Enable the receiver IRQ
 	NVIC_Init((NVIC_InitTypeDef*) &can_dev->cfg->rx_irq.init);
 	NVIC_Init((NVIC_InitTypeDef*) &can_dev->cfg->tx_irq.init);

	return(0);

out_fail:
	return(-1);
}

static void PIOS_CAN_RxStart(uintptr_t can_id, uint16_t rx_bytes_avail)
{
	struct pios_can_dev *can_dev = (struct pios_can_dev *)can_id;
	
	bool valid = PIOS_CAN_validate(can_dev);
	PIOS_Assert(valid);
	
	CAN_ITConfig(can_dev->cfg->regs, CAN_IT_FMP1, ENABLE);
}

static void PIOS_CAN_TxStart(uintptr_t can_id, uint16_t tx_bytes_avail)
{
	struct pios_can_dev *can_dev = (struct pios_can_dev *)can_id;
	
	bool valid = PIOS_CAN_validate(can_dev);
	PIOS_Assert(valid);

 	CAN_ITConfig(can_dev->cfg->regs, CAN_IT_TME, ENABLE);
	
	USB_HP_CAN1_TX_IRQHandler();
}

static void PIOS_CAN_RegisterRxCallback(uintptr_t can_id, pios_com_callback rx_in_cb, uintptr_t context)
{
	struct pios_can_dev *can_dev = (struct pios_can_dev *)can_id;

	bool valid = PIOS_CAN_validate(can_dev);
	PIOS_Assert(valid);
	
	/* 
	 * Order is important in these assignments since ISR uses _cb
	 * field to determine if it's ok to dereference _cb and _context
	 */
	can_dev->rx_in_context = context;
	can_dev->rx_in_cb = rx_in_cb;
}

static void PIOS_CAN_RegisterTxCallback(uintptr_t can_id, pios_com_callback tx_out_cb, uintptr_t context)
{
	struct pios_can_dev *can_dev = (struct pios_can_dev *)can_id;

	bool valid = PIOS_CAN_validate(can_dev);
	PIOS_Assert(valid);
	
	/* 
	 * Order is important in these assignments since ISR uses _cb
	 * field to determine if it's ok to dereference _cb and _context
	 */
	can_dev->tx_out_context = context;
	can_dev->tx_out_cb = tx_out_cb;
}

/**
 * @brief  This function handles CAN1 RX1 request.
 * @note   We are using RX1 instead of RX0 to avoid conflicts with the
 *         USB IRQ handler.
 */
void CAN1_RX1_IRQHandler(void)
{
	CAN_ClearITPendingBit(can_dev->cfg->regs, CAN_IT_FMP1);

	bool valid = PIOS_CAN_validate(can_dev);
	PIOS_Assert(valid);

	CanRxMsg RxMessage;
	CAN_Receive(CAN1, CAN_FIFO1, &RxMessage);

	if (RxMessage.StdId != CAN_COM_ID)
		return;

	bool rx_need_yield = false;
	if (can_dev->rx_in_cb) {
		(void) (can_dev->rx_in_cb)(can_dev->rx_in_context, RxMessage.Data, RxMessage.DLC, NULL, &rx_need_yield);
	}

	portEND_SWITCHING_ISR(rx_need_yield);
}

/**
 * @brief  This function handles CAN1 TX irq and sends more data if available
 */
void USB_HP_CAN1_TX_IRQHandler(void)
{
	CAN_ClearITPendingBit(can_dev->cfg->regs, CAN_IT_TME);

	bool valid = PIOS_CAN_validate(can_dev);
	PIOS_Assert(valid);

	bool tx_need_yield = false;
	
	if (can_dev->tx_out_cb) {

		// Prepare CAN message structure
		CanTxMsg msg;
		msg.StdId = CAN_COM_ID;
		msg.ExtId = 0;
		msg.IDE = CAN_ID_STD;
		msg.RTR = CAN_RTR_DATA;			
		msg.DLC = (can_dev->tx_out_cb)(can_dev->tx_out_context, msg.Data, MAX_SEND_LEN, NULL, &tx_need_yield);

		// Send message and get mailbox number
		if (msg.DLC > 0) {
			CAN_Transmit(can_dev->cfg->regs, &msg);
		} else {
			CAN_ITConfig(can_dev->cfg->regs, CAN_IT_TME, DISABLE);
		}

		// TODO: deal with failure to send and keep the message to retransmit
	}
	
	portEND_SWITCHING_ISR(tx_need_yield);
}

#endif /* PIOS_INCLUDE_CAN */
/**
 * @}
 * @}
 */
