/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_FSKDAC FSK DAC Functions
 * @brief PIOS interface for FSK DAC implementation
 * @{
 *
 * @file       pios_fskdac.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015-2016
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

#include "pios_dac_common.h"

/* Private methods */
static void PIOS_FSKDAC_DMA_irq_cb();
static void PIOS_FSKDAC_Start(uint8_t b);

/* Provide a COM driver */
static void PIOS_FSKDAC_RegisterTxCallback(uintptr_t fskdac_id, pios_com_callback tx_out_cb, uintptr_t context);
static void PIOS_FSKDAC_TxStart(uintptr_t fskdac_id, uint16_t tx_bytes_avail);

const struct pios_com_driver pios_fskdac_com_driver = {
	.tx_start   = PIOS_FSKDAC_TxStart,
	.bind_tx_cb = PIOS_FSKDAC_RegisterTxCallback,
};

enum pios_fskdac_dev_magic {
	PIOS_FSKDAC_DEV_MAGIC = 0x1453834A,
};

enum BYTE_TX_STATE {
	IDLE, START, BIT0, BIT1, BIT2,
	BIT3, BIT4, BIT5, BIT6, BIT7,
	STOP
};

struct pios_fskdac_dev {
	enum pios_fskdac_dev_magic     magic;

	//! Track the state of sending an individual bit
	enum BYTE_TX_STATE tx_state;
	uint8_t cur_byte;

	pios_com_callback tx_out_cb;
	uintptr_t tx_out_context;
};

// Matlab code to generate symbols
/* // Note fudging F2 a touch to make them the same legth
DAC_SBS = 50000; F1 = 1200; p1 = 4; F2 = 2105; p2 = 7;
mark = round((-sin(linspace(0,2*pi*p1,DAC_SBS/F1*p1))*(2^12/3.3*0.5)+(2^12/3.3*0.5)));
space = round((-sin(linspace(0,2*pi*p2,DAC_SBS/F2*p2))*(2^12/3.3*0.5)+(2^12/3.3*0.5)));
['const uint16_t MARK_SAMPLES[] = {', sprintf('%d,', mark), '};']
['const uint16_t SPACE_SAMPLES[] = {', sprintf('%d,', space), '};']
*/

const uint16_t MARK_SAMPLES[] = {621,526,434,347,265,192,129,77,38,12,1,3,20,51,95,152,219,295,379,469,562,656,750,840,926,1004,1074,1133,1180,1214,1235,1241,1233,1211,1175,1126,1066,995,915,829,738,644,550,457,368,285,210,144,89,47,17,2,1,15,42,83,136,201,275,357,446,538,632,726,818,905,985,1057,1119,1169,1207,1231,1241,1236,1218,1185,1139,1082,1013,936,851,761,668,573,480,390,305,228,159,102,56,24,5,0,10,34,72,122,184,256,336,423,515,609,703,795,884,966,1040,1105,1158,1199,1226,1240,1239,1224,1195,1152,1097,1031,956,873,784,691,597,503,412,326,246,176,115,66,30,8,0,6,27,61,108,167,237,315,401,492,585,680,773,862,946,1023,1090,1146,1190,1221,1238,1241,1229,1203,1164,1112,1049,976,894,807,715,621,};
const uint16_t SPACE_SAMPLES[] = {621,457,305,176,77,17,0,27,95,201,336,492,656,818,966,1090,1180,1231,1239,1203,1126,1013,873,715,550,390,246,129,47,5,6,51,136,256,401,562,726,884,1023,1133,1207,1240,1229,1175,1082,956,807,644,480,326,192,89,24,0,20,83,184,315,469,632,795,946,1074,1169,1226,1241,1211,1139,1031,894,738,573,412,265,144,56,8,3,42,122,237,379,538,703,862,1004,1119,1199,1238,1233,1185,1097,976,829,668,503,347,210,102,30,1,15,72,167,295,446,609,773,926,1057,1158,1221,1241,1218,1152,1049,915,761,597,434,285,159,66,12,1,34,108,219,357,515,680,840,985,1105,1190,1235,1236,1195,1112,995,851,691,526,368,228,115,38,2,10,61,152,275,423,585,750,905,1040,1146,1214,1241,1224,1164,1066,936,784,621,};
const uint16_t BLANK_SAMPLES[] = {621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,};

static const uint32_t SAMPLES_PER_BIT = NELEMENTS(MARK_SAMPLES);
static const uint32_t DAC_SBS = 50000;

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

struct pios_fskdac_dev * g_fskdac_dev;

/**
* Initialise a single USART device
*/
int32_t PIOS_FSKDAC_Init(uintptr_t * fskdac_id)
{
	PIOS_DEBUG_Assert(fskdac_id);
	PIOS_DEBUG_Assert(cfg);

	struct pios_fskdac_dev * fskdac_dev;

	fskdac_dev = (struct pios_fskdac_dev *) PIOS_FSKDAC_alloc();
	if (!fskdac_dev) return -1;

	// Handle for the IRQ
	g_fskdac_dev = fskdac_dev;


	PIOS_DAC_COMMON_Init(PIOS_FSKDAC_DMA_irq_cb);

	TIM_SetAutoreload(TIM6, PIOS_PERIPHERAL_APB1_CLOCK / DAC_SBS);

	DMA_DeInit(DMA1_Stream5);
	DMA_InitTypeDef DMA_InitStructure;	
	DMA_InitStructure.DMA_Channel = DMA_Channel_7;  
	DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)&DAC->DHR12R1;
	DMA_InitStructure.DMA_Memory0BaseAddr = (uint32_t)&MARK_SAMPLES[0];
	DMA_InitStructure.DMA_BufferSize = SAMPLES_PER_BIT;
	DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_HalfWord;
	DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_HalfWord;
	DMA_InitStructure.DMA_DIR = DMA_DIR_MemoryToPeripheral;
	DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
	DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
	DMA_InitStructure.DMA_Mode = DMA_Mode_Circular;
	DMA_InitStructure.DMA_Priority = DMA_Priority_High;
	DMA_InitStructure.DMA_FIFOMode = DMA_FIFOMode_Disable;
	DMA_InitStructure.DMA_FIFOThreshold = DMA_FIFOThreshold_HalfFull;
	DMA_InitStructure.DMA_MemoryBurst = DMA_MemoryBurst_Single;
	DMA_InitStructure.DMA_PeripheralBurst = DMA_PeripheralBurst_Single;
	DMA_Init(DMA1_Stream5, &DMA_InitStructure);

	/* Configure double buffering */
	DMA_DoubleBufferModeConfig(DMA1_Stream5, (uint32_t)&SPACE_SAMPLES[0], DMA_Memory_0);
	DMA_DoubleBufferModeCmd(DMA1_Stream5, ENABLE);

	fskdac_dev->tx_state = IDLE;

	*fskdac_id = (uintptr_t) fskdac_dev;

	return 0;
}


static void PIOS_FSKDAC_TxStart(uintptr_t fskdac_id, uint16_t tx_bytes_avail)
{
	struct pios_fskdac_dev * fskdac_dev = (struct pios_fskdac_dev *)fskdac_id;

	bool valid = PIOS_FSKDAC_validate(fskdac_dev);
	PIOS_Assert(valid);
	
	uint8_t b;
	uint16_t bytes_to_send;
	bool tx_need_yield = false;

	bytes_to_send = (fskdac_dev->tx_out_cb)(fskdac_dev->tx_out_context, &b, 1, NULL, &tx_need_yield);

	if (bytes_to_send > 0) {
		/* Send the byte we've been given */
		PIOS_FSKDAC_Start(b);
	}
}

static void PIOS_FSKDAC_RegisterTxCallback(uintptr_t fskdac_id, pios_com_callback tx_out_cb, uintptr_t context)
{
	struct pios_fskdac_dev * fskdac_dev = (struct pios_fskdac_dev *)fskdac_id;

	bool valid = PIOS_FSKDAC_validate(fskdac_dev);
	PIOS_Assert(valid);

	/* 
	 * Order is important in these assignments since ISR uses _cb
	 * field to determine if it's ok to dereference _cb and _context
	 */
	fskdac_dev->tx_out_context = context;
	fskdac_dev->tx_out_cb = tx_out_cb;
}

/**
 * Schedule the next bit to be written by setting the address of the
 * double buffer
 */
static void pios_fskdac_set_symbol(struct pios_fskdac_dev * fskdac_dev, uint8_t sym)
{
	const uint16_t * next_symbol = BLANK_SAMPLES;
	switch(sym) {
	case 0:
		next_symbol = MARK_SAMPLES;
		break;
	case 1:
		next_symbol = SPACE_SAMPLES;
		break;
	case 2:
		next_symbol = BLANK_SAMPLES;
		break;
	}

	if (DMA_GetCurrentMemoryTarget(DMA1_Stream5)) {
		// If currently using memory 1 then schedule memory 0
		DMA1_Stream5->M0AR = (uint32_t) next_symbol;
	} else {
		DMA1_Stream5->M1AR = (uint32_t) next_symbol;
	}
}

static void PIOS_FSKDAC_Start(uint8_t b)
{
	struct pios_fskdac_dev * fskdac_dev = g_fskdac_dev;

	bool valid = PIOS_FSKDAC_validate(fskdac_dev);
	PIOS_Assert(valid);

	fskdac_dev->tx_state = START;
	fskdac_dev->cur_byte = b;
	pios_fskdac_set_symbol(fskdac_dev, 0);

	/* Enable transfer complete interrupt for this stream */
	DMA_ITConfig(DMA1_Stream5, DMA_IT_TC, ENABLE);

	/* Enable DMA1_Stream5 */
	DMA_Cmd(DMA1_Stream5, ENABLE);

	/* Enable DAC Channel1 */
	DAC_Cmd(DAC_Channel_1, ENABLE);

	/* Enable DMA for DAC Channel1 */
	DAC_DMACmd(DAC_Channel_1, ENABLE);
}

static void PIOS_FSKDAC_DMA_irq_cb()
{
	static uint8_t byte = 0;

	struct pios_fskdac_dev * fskdac_dev = g_fskdac_dev;

	bool valid = PIOS_FSKDAC_validate(fskdac_dev);
	PIOS_Assert(valid);

	bool tx_need_yield = false;
	
	switch(fskdac_dev->tx_state) {
	case IDLE:
		if (fskdac_dev->tx_out_cb) {
			uint8_t b;
			uint16_t bytes_to_send = 0;
			
			if (fskdac_dev->tx_out_cb)
				bytes_to_send = (fskdac_dev->tx_out_cb)(fskdac_dev->tx_out_context, &b, 1, NULL, &tx_need_yield);
			
			if (bytes_to_send > 0) {
				/* Send the byte we've been given */
				PIOS_FSKDAC_Start(b);
			} else {
				/* No bytes to send, stay here */
				DMA_ITConfig(DMA1_Stream5, DMA_IT_TC, DISABLE);
				DAC_DMACmd(DAC_Channel_1, DISABLE);
			}
		} else {
			PIOS_FSKDAC_Start(byte++);
		}
		break;
	case START:
		pios_fskdac_set_symbol(fskdac_dev, (fskdac_dev->cur_byte & 0x01) >> 0);
		fskdac_dev->tx_state = BIT0;
		break;
	case BIT0:
		pios_fskdac_set_symbol(fskdac_dev, (fskdac_dev->cur_byte & 0x02) >> 1);
		fskdac_dev->tx_state = BIT1;
		break;
	case BIT1:
		pios_fskdac_set_symbol(fskdac_dev, (fskdac_dev->cur_byte & 0x04) >> 2);
		fskdac_dev->tx_state = BIT2;
		break;
	case BIT2:
		pios_fskdac_set_symbol(fskdac_dev, (fskdac_dev->cur_byte & 0x08) >> 3);
		fskdac_dev->tx_state = BIT3;
		break;
	case BIT3:
		pios_fskdac_set_symbol(fskdac_dev, (fskdac_dev->cur_byte & 0x10) >> 4);
		fskdac_dev->tx_state = BIT4;
		break;
	case BIT4:
		pios_fskdac_set_symbol(fskdac_dev, (fskdac_dev->cur_byte & 0x20) >> 5);
		fskdac_dev->tx_state = BIT5;
		break;
	case BIT5:
		pios_fskdac_set_symbol(fskdac_dev, (fskdac_dev->cur_byte & 0x40) >> 6);
		fskdac_dev->tx_state = BIT6;
		break;
	case BIT6:
		pios_fskdac_set_symbol(fskdac_dev, (fskdac_dev->cur_byte & 0x80) >> 7);
		fskdac_dev->tx_state = BIT7;
		break;
	case BIT7:
		// Set stop bit
		pios_fskdac_set_symbol(fskdac_dev, 0);
		fskdac_dev->tx_state = STOP;
		break;
	case STOP:
		fskdac_dev->tx_state = IDLE;
		pios_fskdac_set_symbol(fskdac_dev, 2);
		break;
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
