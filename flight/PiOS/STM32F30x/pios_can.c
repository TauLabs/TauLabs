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
static void PIOS_CAN_RegisterRxCallback(uint32_t can_id, pios_com_callback rx_in_cb, uint32_t context);
static void PIOS_CAN_RegisterTxCallback(uint32_t can_id, pios_com_callback tx_out_cb, uint32_t context);
static void PIOS_CAN_TxStart(uint32_t can_id, uint16_t tx_bytes_avail);
static void PIOS_CAN_RxStart(uint32_t can_id, uint16_t rx_bytes_avail);

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
	const struct pios_can_cfg * cfg;

	pios_com_callback rx_in_cb;
	uint32_t rx_in_context;
	pios_com_callback tx_out_cb;
	uint32_t tx_out_context;
};

/* Local helper methods */
//static int32_t PIOS_CAN_SendData(uint32_t can_id, uint8_t *data, uint32_t data_len);


static bool PIOS_CAN_validate(struct pios_can_dev * can_dev)
{
	return (can_dev->magic == PIOS_CAN_DEV_MAGIC);
}

#if !defined(PIOS_INCLUDE_FREERTOS)
#error PIOS_CAN REQUIRES FREERTOS
#endif

static struct pios_can_dev * PIOS_CAN_alloc(void)
{
	struct pios_can_dev * can_dev;

	can_dev = (struct pios_can_dev *)pvPortMalloc(sizeof(*can_dev));
	if (!can_dev) return(NULL);

	memset(can_dev, 0, sizeof(*can_dev));
	can_dev->magic = PIOS_CAN_DEV_MAGIC;

	return(can_dev);
}

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

	struct pios_can_dev * can_dev;

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



	*can_id = (uint32_t)can_dev;

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
	CAN_FilterInitStructure.CAN_FilterFIFOAssignment = 0;

	CAN_FilterInitStructure.CAN_FilterActivation = ENABLE;
	CAN_FilterInit(&CAN_FilterInitStructure);

	return(0);

out_fail:
	return(-1);
}

static void PIOS_CAN_RxStart(uint32_t can_id, uint16_t rx_bytes_avail)
{
	struct pios_can_dev * can_dev = (struct pios_can_dev *)can_id;
	
	bool valid = PIOS_CAN_validate(can_dev);
	PIOS_Assert(valid);
	
	//USART_ITConfig(usart_dev->cfg->regs, USART_IT_RXNE, ENABLE);
}
static void PIOS_CAN_TxStart(uint32_t can_id, uint16_t tx_bytes_avail)
{
	struct pios_can_dev * can_dev = (struct pios_can_dev *)can_id;
	
	bool valid = PIOS_CAN_validate(can_dev);
	PIOS_Assert(valid);
	
	//USART_ITConfig(usart_dev->cfg->regs, USART_IT_TXE, ENABLE);
}

static void PIOS_CAN_RegisterRxCallback(uint32_t can_id, pios_com_callback rx_in_cb, uint32_t context)
{
	struct pios_can_dev * can_dev = (struct pios_can_dev *)can_id;

	bool valid = PIOS_CAN_validate(can_dev);
	PIOS_Assert(valid);
	
	/* 
	 * Order is important in these assignments since ISR uses _cb
	 * field to determine if it's ok to dereference _cb and _context
	 */
	can_dev->rx_in_context = context;
	can_dev->rx_in_cb = rx_in_cb;
}

static void PIOS_CAN_RegisterTxCallback(uint32_t can_id, pios_com_callback tx_out_cb, uint32_t context)
{
	struct pios_can_dev * can_dev = (struct pios_can_dev *)can_id;

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
 * Send a block of data via the CAN device
 * @param[in]  can_id     the device handle
 * @param[in]  data       the array of data to send
 * @param[in]  data_len   the number of valid bytes in the buffer
 * @returns number of bytes sent or < 0 for failure
 */
int32_t PIOS_CAN_SendData(uint32_t can_id, uint8_t *data, uint32_t data_len)
{
/*
    (#) Transmit the desired CAN frame using CAN_Transmit() function.
    (#) Check the transmission of a CAN frame using CAN_TransmitStatus() function.
    (#) Cancel the transmission of a CAN frame using CAN_CancelTransmit() function.  
    (#) Receive a CAN frame using CAN_Recieve() function.
    (#) Release the receive FIFOs using CAN_FIFORelease() function.
    (#) Return the number of pending received frames using CAN_MessagePending() function.            
    (#) To control CAN events you can use one of the following two methods:
        (++) Check on CAN flags using the CAN_GetFlagStatus() function.  
        (++) Use CAN interrupts through the function CAN_ITConfig() at initialization 
             phase and CAN_GetITStatus() function into interrupt routines to check 
             if the event has occurred or not.
             After checking on a flag you should clear it using CAN_ClearFlag()
             function. And after checking on an interrupt event you should clear it 
             using CAN_ClearITPendingBit() function.            
*/
	struct pios_can_dev * can_dev = (struct pios_can_dev *)can_id;

	bool valid = PIOS_CAN_validate(can_dev);
	PIOS_Assert(valid);

	const uint32_t MAX_SEND_LEN =  8;
	const uint32_t MAX_SEND_TIME = 10; // mss

	if (data_len > MAX_SEND_LEN)
		data_len = MAX_SEND_LEN;

	// Prepare CAN message structure
	CanTxMsg msg;
	msg.StdId = 0x11;
	msg.ExtId = 0;
	msg.IDE = CAN_ID_STD;
	msg.RTR = CAN_RTR_DATA;
	msg.DLC = data_len;
	memcpy(msg.Data, data, data_len);

	// Send message and get mailbox number
	uint8_t mbx = CAN_Transmit(can_dev->cfg->regs, &msg);
	if (mbx == CAN_TxStatus_NoMailBox)
		return -1;

	// Wait for acknowledgment of send or timeout
	uint32_t raw_time = PIOS_DELAY_GetRaw();
	uint8_t status = CAN_TxStatus_Pending;
	while (status != CAN_TxStatus_Ok && (PIOS_DELAY_DiffuS(raw_time) < (MAX_SEND_TIME * 1000)) ) {
		status = CAN_TransmitStatus(can_dev->cfg->regs, mbx);
	}

	// Succeede, return sent number of bytes
	if (status != CAN_TxStatus_Ok)
		return -3;

	// If timeout, cancel message
	if (PIOS_DELAY_DiffuS(raw_time) >= (MAX_SEND_TIME * 1000))
		CAN_CancelTransmit(can_dev->cfg->regs, mbx);

	/***** Loopback testing *******/
	// verify we received the same message we sent
	uint32_t i = 0;
	while((CAN_MessagePending(CAN1, CAN_FIFO0) < 1) && (i  !=  0xFFFF))
	{
		i++;
	}

	/* receive */
	CanRxMsg RxMessage;
	RxMessage.StdId = 0x00;
	RxMessage.IDE = CAN_ID_STD;
	RxMessage.DLC = 0;
	RxMessage.Data[0] = 0x00;
	RxMessage.Data[1] = 0x00;
	CAN_Receive(CAN1, CAN_FIFO0, &RxMessage);

	if (RxMessage.StdId != 0x11)
	{
		return -3;
	}

	if (RxMessage.IDE != CAN_ID_STD)
	{
		return -4;
	}

	if (RxMessage.DLC != data_len)
	{
		return -5;  
	}

	if (RxMessage.Data[0] != data[0])
	{
		return -6;
	}	

	return data_len;
}

// static void PIOS_CAN_generic_irq_handler(uint32_t usart_id)
// {
// 	struct pios_usart_dev * usart_dev = (struct pios_usart_dev *)usart_id;

// 	bool rx_need_yield = false;
// 	bool tx_need_yield = false;

// 	bool valid = PIOS_USART_validate(usart_dev);
// 	PIOS_Assert(valid);
	
// 	/* Check if RXNE flag is set */
// 	if (USART_GetITStatus(usart_dev->cfg->regs, USART_IT_RXNE)) {
// 		uint8_t byte = (uint8_t)USART_ReceiveData(usart_dev->cfg->regs);
// 		if (usart_dev->rx_in_cb) {
// 			(void) (usart_dev->rx_in_cb)(usart_dev->rx_in_context, &byte, 1, NULL, &rx_need_yield);
// 		}
// 	}
// 	/* Check if TXE flag is set */
// 	if (USART_GetITStatus(usart_dev->cfg->regs, USART_IT_TXE)) {
// 		if (usart_dev->tx_out_cb) {
// 			uint8_t b;
// 			uint16_t bytes_to_send;
			
// 			bytes_to_send = (usart_dev->tx_out_cb)(usart_dev->tx_out_context, &b, 1, NULL, &tx_need_yield);
			
// 			if (bytes_to_send > 0) {
// 				/* Send the byte we've been given */
// 				USART_SendData(usart_dev->cfg->regs, b);
// 			} else {
// 				/* No bytes to send, disable TXE interrupt */
// 				USART_ITConfig(usart_dev->cfg->regs, USART_IT_TXE, DISABLE);
// 			}
// 		} else {
// 			/* No bytes to send, disable TXE interrupt */
// 			USART_ITConfig(usart_dev->cfg->regs, USART_IT_TXE, DISABLE);
// 		}
// 	}
// 	/* Check for overrun condition
// 	 * Note i really wanted to use USART_GetITStatus but it fails on getting the
// 	 * ORE flag although RXNE interrupt is enabled.
// 	 * Probably a bug in the ST library...
// 	 */
// 	if (USART_GetFlagStatus(usart_dev->cfg->regs, USART_FLAG_ORE)) {
// 		USART_ClearITPendingBit(usart_dev->cfg->regs, USART_IT_ORE);
// 		++usart_dev->error_overruns;
// 	}
	
// #if defined(PIOS_INCLUDE_FREERTOS)
// 	portEND_SWITCHING_ISR(rx_need_yield || tx_need_yield);
// #endif	/* PIOS_INCLUDE_FREERTOS */
// }


#endif /* PIOS_INCLUDE_CAN */
/**
 * @}
 * @}
 */
