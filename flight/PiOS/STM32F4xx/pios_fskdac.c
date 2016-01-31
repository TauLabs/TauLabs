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

#include "pios_fskdac_priv.h"

#if defined(PIOS_INCLUDE_FREERTOS)
#include "FreeRTOS.h"
#endif /* defined(PIOS_INCLUDE_FREERTOS) */

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
	const struct pios_fskdac_config * cfg;

	//! Track the state of sending an individual bit
	enum BYTE_TX_STATE tx_state;
	uint8_t cur_byte;

	pios_com_callback tx_out_cb;
	uintptr_t tx_out_context;
};

const uint32_t ARRAY[5] = {1,2,3,4,5};

// Lookup tables for symbols
// from ML designed to generate 1V p-p centered at 0.5V
//   sprintf('%d,', round((sin(linspace(0,2*pi,130))*(2^12/3.3*0.5)+(2^12/3.3*0.5))))
const uint16_t MARK_SAMPLES[] = {
	621,651,681,711,741,770,799,828,856,884,
	911,937,963,988,1012,1035,1057,1078,1098,1116,
	1134,1150,1165,1179,1192,1203,1213,1221,1228,1233,
	1237,1240,1241,1241,1239,1236,1231,1225,1217,1208,
	1198,1186,1173,1158,1142,1125,1107,1088,1067,1046,
	1023,1000,976,950,924,898,870,842,814,785,
	756,726,696,666,636,605,575,545,515,486,
	456,427,399,371,344,317,291,266,241,218,
	195,174,153,134,116,99,83,69,56,44,
	33,24,17,10,6,2,0,0,1,4,
	8,13,20,29,38,49,62,76,91,107,
	125,144,163,184,206,229,253,278,304,330,
	357,385,413,442,471,500,530,560,590,621,
}; // 10x13 samples


// sprintf('%d,', round((sin(linspace(0,4*pi,130))*(2^12/3.3*0.5)+(2^12/3.3*0.5))))
const uint16_t SPACE_SAMPLES[] = { // sin wave 2x freq
	621,681,741,799,856,911,963,1012,1057,1098,
	1134,1165,1192,1213,1228,1237,1241,1239,1231,1217,
	1198,1173,1142,1107,1067,1023,976,924,870,814,
	756,696,636,575,515,456,399,344,291,241,
	195,153,116,83,56,33,17,6,0,1,
	8,20,38,62,91,125,163,206,253,304,
	357,413,471,530,590,651,711,770,828,884,
	937,988,1035,1078,1116,1150,1179,1203,1221,1233,
	1240,1241,1236,1225,1208,1186,1158,1125,1088,1046,
	1000,950,898,842,785,726,666,605,545,486,
	427,371,317,266,218,174,134,99,69,44,
	24,10,2,0,4,13,29,49,76,107,
	144,184,229,278,330,385,442,500,560,621,
}; // 10x13 samples

const uint32_t SAMPLES_PER_BIT = NELEMENTS(MARK_SAMPLES);

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
int32_t PIOS_FSKDAC_Init(uintptr_t * fskdac_id, const struct pios_fskdac_config * cfg)
{
	PIOS_DEBUG_Assert(fskdac_id);
	PIOS_DEBUG_Assert(cfg);

	struct pios_fskdac_dev * fskdac_dev;

	fskdac_dev = (struct pios_fskdac_dev *) PIOS_FSKDAC_alloc();
	if (!fskdac_dev) return -1;

	// Handle for the IRQ
	g_fskdac_dev = fskdac_dev;

	/* Bind the configuration to the device instance */
	fskdac_dev->cfg = cfg; // TODO: use this

	GPIO_InitTypeDef gpio_init;
	gpio_init.GPIO_Pin  = GPIO_Pin_4;
	gpio_init.GPIO_Mode = GPIO_Mode_AN;
	gpio_init.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOA, &gpio_init);

	TIM_TimeBaseInitTypeDef TIM6_TimeBase;
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM6, ENABLE);
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_DAC, ENABLE);
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_DMA1, ENABLE);

	// TODO: move into board_hw_defs and cfg structure
	TIM_TimeBaseStructInit(&TIM6_TimeBase); 
	TIM6_TimeBase.TIM_Period        = (PIOS_PERIPHERAL_APB1_CLOCK / (5000 * SAMPLES_PER_BIT));
	TIM6_TimeBase.TIM_Prescaler     = 0;
	TIM6_TimeBase.TIM_ClockDivision = 0;
	TIM6_TimeBase.TIM_CounterMode   = TIM_CounterMode_Up;
	TIM_TimeBaseInit(TIM6, &TIM6_TimeBase);
	TIM_SelectOutputTrigger(TIM6, TIM_TRGOSource_Update);
	TIM_Cmd(TIM6, ENABLE);

	DAC_InitTypeDef DAC_INIT;
	DAC_StructInit(&DAC_INIT);
	DAC_DeInit();
	DAC_INIT.DAC_Trigger        = DAC_Trigger_T6_TRGO;
	DAC_INIT.DAC_WaveGeneration = DAC_WaveGeneration_None;
	DAC_INIT.DAC_OutputBuffer   = DAC_OutputBuffer_Enable;
	DAC_Init(DAC_Channel_1, &DAC_INIT);

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

	DMA_DoubleBufferModeConfig(fskdac_dev->cfg->dma.tx.channel, (uint32_t)&SPACE_SAMPLES[0], DMA_Memory_0);
	DMA_DoubleBufferModeCmd(fskdac_dev->cfg->dma.tx.channel, ENABLE);
	//DMA_ITConfig(fskdac_dev->cfg->dma.tx.channel, DMA_IT_TC, ENABLE);

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
	
	// TODO: equivalent. USART_ITConfig(usart_dev->cfg->regs, USART_IT_TXE, ENABLE);
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

static void pios_fskdac_set_symbol(struct pios_fskdac_dev * fskdac_dev, uint8_t sym)
{
	// TODO
	/* Set Memory 0 as current memory address */
	// if ( DMAy_Streamx->CR & (uint32_t)(DMA_SxCR_CT))

	/* Write to DMAy Streamx M1AR */
	//DMAy_Streamx->M1AR = MARK;
	//DMAy_Streamx->M0AR = SPACE;
}

// Should be aliased from DMA1_Stream7_IRQHandler
void PIOS_FSKDAC_DMA_irq_handler()
{
#if defined(PIOS_INCLUDE_CHIBIOS)
	CH_IRQ_PROLOGUE();
#endif /* defined(PIOS_INCLUDE_CHIBIOS) */

	struct pios_fskdac_dev * fskdac_dev = g_fskdac_dev;

	bool valid = PIOS_FSKDAC_validate(fskdac_dev);
	PIOS_Assert(valid);

	bool tx_need_yield = false;
	
	switch(fskdac_dev->tx_state) {
	case IDLE:
		if (fskdac_dev->tx_out_cb) {
			uint8_t b;
			uint16_t bytes_to_send;
			
			bytes_to_send = (fskdac_dev->tx_out_cb)(fskdac_dev->tx_out_context, &b, 1, NULL, &tx_need_yield);
			
			if (bytes_to_send > 0) {
				/* Send the byte we've been given */
				fskdac_dev->cur_byte = b;
				pios_fskdac_set_symbol(fskdac_dev, 0);
				fskdac_dev->tx_state = START;
			} else {
				/* No bytes to send, stay here */
			}
		} 
		break;
	case START:
		pios_fskdac_set_symbol(fskdac_dev, fskdac_dev->cur_byte & 0x01);
		fskdac_dev->tx_state = BIT0;
		break;
	case BIT0:
		pios_fskdac_set_symbol(fskdac_dev, fskdac_dev->cur_byte & 0x02);
		fskdac_dev->tx_state = BIT1;
		break;
	case BIT1:
		pios_fskdac_set_symbol(fskdac_dev, fskdac_dev->cur_byte & 0x04);
		fskdac_dev->tx_state = BIT2;
		break;
	case BIT2:
		pios_fskdac_set_symbol(fskdac_dev, fskdac_dev->cur_byte & 0x08);
		fskdac_dev->tx_state = BIT3;
		break;
	case BIT3:
		pios_fskdac_set_symbol(fskdac_dev, fskdac_dev->cur_byte & 0x10);
		fskdac_dev->tx_state = BIT4;
		break;
	case BIT4:
		pios_fskdac_set_symbol(fskdac_dev, fskdac_dev->cur_byte & 0x20);
		fskdac_dev->tx_state = BIT5;
		break;
	case BIT5:
		pios_fskdac_set_symbol(fskdac_dev, fskdac_dev->cur_byte & 0x40);
		fskdac_dev->tx_state = BIT6;
		break;
	case BIT6:
		pios_fskdac_set_symbol(fskdac_dev, fskdac_dev->cur_byte & 0x80);
		fskdac_dev->tx_state = BIT7;
		break;
	case BIT7:
		// Set stop bit
		pios_fskdac_set_symbol(fskdac_dev, fskdac_dev->cur_byte & 0x80);
		fskdac_dev->tx_state = STOP;
		break;
	case STOP:
		fskdac_dev->tx_state = IDLE;
		break;
	}
	
#if defined(PIOS_INCLUDE_FREERTOS)
	portEND_SWITCHING_ISR((rx_need_yield || tx_need_yield) ? pdTRUE : pdFALSE);
#endif	/* defined(PIOS_INCLUDE_FREERTOS) */

#if defined(PIOS_INCLUDE_CHIBIOS)
	CH_IRQ_EPILOGUE();
#endif /* defined(PIOS_INCLUDE_CHIBIOS) */

}

#endif /* PIOS_INCLUDE_FSK */

/**
  * @}
  * @}
  */
