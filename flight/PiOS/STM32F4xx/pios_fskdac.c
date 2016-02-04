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
#include "physical_constants.h"

/* Private methods */
static void PIOS_FSKDAC_DMA_irq_cb();

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

	//! The phase when starting and ending this symbol
	float starting_phase;
	float ending_phase;

	pios_com_callback tx_out_cb;
	uintptr_t tx_out_context;
};

// Matlab code to generate symbols
/*
DAC_SBS = 50000; F1 = 1200; p1 = 2; F2 = 2200; p2 = 3; % space period of each to allow allignment
mark = round((-sin(linspace(0,2*pi*p1,DAC_SBS/F1*p1))*(2^12/3.3*0.5)+(2^12/3.3*0.5)));
space = round((-sin(linspace(0,2*pi*p2,DAC_SBS/F2*p2))*(2^12/3.3*0.5)+(2^12/3.3*0.5)));
blank = zeros(1,round(DAC_SBS/1200)) + round((2^12/3.3*0.5));
['const uint16_t MARK_SAMPLES[] = {', sprintf('%d,', mark), '};']
['const uint16_t SPACE_SAMPLES[] = {', sprintf('%d,', space), '};']
['const uint16_t BLANK_SAMPLES[] = {', sprintf('%d,', blank), '};']
sprintf('SAMPLES_PER_BIT = %d ',round(DAC_SBS/1200)) % duration of symbol at 1200bps
sprintf('SAMPLES_PER_BIT = %d ',DAC_SBS) 
*/

const uint16_t MARK_SAMPLES[] = {621,526,433,345,264,190,127,75,37,11,0,4,22,54,100,157,226,303,389,479,573,668,762,853,938,1015,1084,1142,1187,1219,1237,1241,1230,1205,1166,1114,1051,978,896,808,715,621,526,433,345,264,190,127,75,37,11,0,4,22,54,100,157,226,303,389,479,573,668,762,853,938,1015,1084,1142,1187,1219,1237,1241,1230,1205,1166,1114,1051,978,896,808,715,621,};
const uint16_t SPACE_SAMPLES[] = {621,448,290,157,61,8,4,49,138,265,421,592,765,927,1065,1168,1227,1240,1203,1121,999,848,679,505,340,197,88,21,0,29,104,219,366,534,708,875,1022,1138,1213,1241,1221,1153,1044,901,736,562,393,242,120,38,2,14,74,177,315,476,650,821,976,1103,1193,1237,1233,1181,1084,952,793,621,};
const uint16_t BLANK_SAMPLES[] = {621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,621,};
static const uint32_t SAMPLES_PER_BIT = 42;
static const uint32_t DAC_SBS = 50000;
const float MARK_SAMPLES_PER_CYCLE = 41.666667f;
const float SPACE_SAMPLES_PER_CYCLE = 22.727273f;

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

	/* Enable DMA1_Stream5 */
	DMA_Cmd(DMA1_Stream5, ENABLE);

	/* Enable DAC Channel1 */
	DAC_Cmd(DAC_Channel_1, ENABLE);

	/* Enable DMA for DAC Channel1 */
	DAC_DMACmd(DAC_Channel_1, ENABLE);

	return 0;
}


static void PIOS_FSKDAC_TxStart(uintptr_t fskdac_id, uint16_t tx_bytes_avail)
{
	struct pios_fskdac_dev * fskdac_dev = (struct pios_fskdac_dev *)fskdac_id;

	bool valid = PIOS_FSKDAC_validate(fskdac_dev);
	PIOS_Assert(valid);
	
	/* Enable transfer complete interrupt for this stream */
	/* Once it is called then the next byte will be loaded */
	DMA_ITConfig(DMA1_Stream5, DMA_IT_TC, ENABLE);
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
	// TODO: based on previous symbol, we know where the phase
	// will end up, and we should ensure the next symbol starts
	// at this phase for clean generation

	// Starting phase for this symbol is previous ending phase
	fskdac_dev->starting_phase = fmodf(fskdac_dev->ending_phase, 2*PI);
	uint8_t start_phase_offset = 0;

	const uint16_t * next_symbol = BLANK_SAMPLES;
	switch(sym) {
	case 0:
		start_phase_offset = floorf(fskdac_dev->starting_phase / (2 * PI) * MARK_SAMPLES_PER_CYCLE);
		next_symbol = &MARK_SAMPLES[start_phase_offset];
		fskdac_dev->ending_phase = fskdac_dev->starting_phase + (2 * PI * SAMPLES_PER_BIT / MARK_SAMPLES_PER_CYCLE);
		break;
	case 1:
		start_phase_offset = floorf(fskdac_dev->starting_phase / (2 * PI) * SPACE_SAMPLES_PER_CYCLE);
		next_symbol = &SPACE_SAMPLES[start_phase_offset];
		fskdac_dev->ending_phase = fskdac_dev->starting_phase + (2 * PI * SAMPLES_PER_BIT / SPACE_SAMPLES_PER_CYCLE);
		break;
	case 2:
		next_symbol = BLANK_SAMPLES;
		fskdac_dev->ending_phase = 0;
		break;
	}

	uint32_t current = DMA_GetCurrentMemoryTarget(DMA1_Stream5);
	DMA_MemoryTargetConfig(DMA1_Stream5, (uint32_t) next_symbol, current == 1 ? DMA_Memory_0 : DMA_Memory_1);
}

static void PIOS_FSKDAC_DMA_irq_cb()
{
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
				fskdac_dev->tx_state = START;
				fskdac_dev->cur_byte = b;
				// Start outputing a SPACE for start symbol
				pios_fskdac_set_symbol(fskdac_dev, 1);

				DMA_ITConfig(DMA1_Stream5, DMA_IT_TC, ENABLE);
			} else {
				/* No bytes to send, stay here but do not call interrupt until more data */
				DMA_ITConfig(DMA1_Stream5, DMA_IT_TC, DISABLE);
				// STOP state already set us back to holding at MARK
			}
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
		pios_fskdac_set_symbol(fskdac_dev, 0); // Output mark between symbols
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
