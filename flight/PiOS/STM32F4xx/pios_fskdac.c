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

	pios_com_callback rx_in_cb;
	uintptr_t rx_in_context;
	pios_com_callback tx_out_cb;
	uintptr_t tx_out_context;
};

const uint32_t ARRAY[5] = {1,2,3,4,5};

// Lookup tables for symbols
const uint16_t MARK_SAMPLES[] = {
2048, 2145, 2242, 2339, 2435, 2530, 2624, 2717, 2808, 2897, 
2984, 3069, 3151, 3230, 3307, 3381, 3451, 3518, 3581, 3640, 
3696, 3748, 3795, 3838, 3877, 3911, 3941, 3966, 3986, 4002, 
4013, 4019, 4020, 4016, 4008, 3995, 3977, 3954, 3926, 3894, 
3858, 3817, 3772, 3722, 3669, 3611, 3550, 3485, 3416, 3344, 
3269, 3191, 3110, 3027, 2941, 2853, 2763, 2671, 2578, 2483, 
2387, 2291, 2194, 2096, 1999, 1901, 1804, 1708, 1612, 1517, 
1424, 1332, 1242, 1154, 1068, 985, 904, 826, 751, 679, 
610, 545, 484, 426, 373, 323, 278, 237, 201, 169, 
141, 118, 100, 87, 79, 75, 76, 82, 93, 109, 
129, 154, 184, 218, 257, 300, 347, 399, 455, 514, 
577, 644, 714, 788, 865, 944, 1026, 1111, 1198, 1287, 
1378, 1471, 1565, 1660, 1756, 1853, 1950, 2047,
};
const uint16_t SPACE_SAMPLES[] = {
2048, 2242, 2435, 2624, 2808, 2984, 3151, 3307, 3451, 3581, 
3696, 3795, 3877, 3941, 3986, 4013, 4020, 4008, 3977, 3926, 
3858, 3772, 3669, 3550, 3416, 3269, 3110, 2941, 2763, 2578, 
2387, 2194, 1999, 1804, 1612, 1424, 1242, 1068, 904, 751, 
610, 484, 373, 278, 201, 141, 100, 79, 76, 93, 
129, 184, 257, 347, 455, 577, 714, 865, 1026, 1198, 
1378, 1565, 1756, 1950, 2145, 2339, 2530, 2717, 2897, 3069, 
3230, 3381, 3518, 3640, 3748, 3838, 3911, 3966, 4002, 4019, 
4016, 3995, 3954, 3894, 3817, 3722, 3611, 3485, 3344, 3191, 
3027, 2853, 2671, 2483, 2291, 2096, 1901, 1708, 1517, 1332, 
1154, 985, 826, 679, 545, 426, 323, 237, 169, 118, 
87, 75, 82, 109, 154, 218, 300, 399, 514, 644, 
788, 944, 1111, 1287, 1471, 1660, 1853, 2047,
};
const uint32_t SAMPLES_PER_BIT = NELEMENTS(MARK_SAMPLES);


#define   DAC_DHR12R1_ADDR  0x40007408

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

	// TODO: move into board_hw_defs and cfg structure
	TIM_TimeBaseInitTypeDef TIM6_TimeBase;
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM6, ENABLE);
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_DAC, ENABLE);
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_DMA1, ENABLE);

	TIM_TimeBaseStructInit(&TIM6_TimeBase); 
	TIM6_TimeBase.TIM_Period        = (uint16_t)PIOS_PERIPHERAL_APB1_CLOCK / (5000 * 128);
	TIM6_TimeBase.TIM_Prescaler     = 0;
	TIM6_TimeBase.TIM_ClockDivision = 0;
	TIM6_TimeBase.TIM_CounterMode   = TIM_CounterMode_Up;
	TIM_TimeBaseInit(TIM6, &TIM6_TimeBase);
	TIM_SelectOutputTrigger(TIM6, TIM_TRGOSource_Update);
	TIM_Cmd(TIM6, ENABLE);

	DAC_InitTypeDef DAC_INIT;
	DAC_INIT.DAC_Trigger        = DAC_Trigger_T6_TRGO;
	DAC_INIT.DAC_WaveGeneration = DAC_WaveGeneration_None;
	DAC_INIT.DAC_OutputBuffer   = DAC_OutputBuffer_Enable;
	DAC_Init(DAC_Channel_1, &DAC_INIT);

	DMA_DeInit(DMA1_Stream5);
	DMA_InitTypeDef DMA_INIT;
	DMA_INIT.DMA_Channel            = DMA_Channel_7;
	DMA_INIT.DMA_PeripheralBaseAddr = (uint32_t)DAC_DHR12R1_ADDR;
	DMA_INIT.DMA_Memory0BaseAddr    = (uint32_t)&SPACE_SAMPLES[0];
	DMA_INIT.DMA_DIR                = DMA_DIR_MemoryToPeripheral;
	DMA_INIT.DMA_BufferSize         = SAMPLES_PER_BIT;
	DMA_INIT.DMA_PeripheralInc      = DMA_PeripheralInc_Disable;
	DMA_INIT.DMA_MemoryInc          = DMA_MemoryInc_Enable;
	DMA_INIT.DMA_PeripheralDataSize = DMA_PeripheralDataSize_HalfWord;
	DMA_INIT.DMA_MemoryDataSize     = DMA_MemoryDataSize_HalfWord;
	DMA_INIT.DMA_Mode               = DMA_Mode_Circular;
	DMA_INIT.DMA_Priority           = DMA_Priority_High;
	DMA_INIT.DMA_FIFOMode           = DMA_FIFOMode_Disable;
	DMA_INIT.DMA_FIFOThreshold      = DMA_FIFOThreshold_HalfFull;
	DMA_INIT.DMA_MemoryBurst        = DMA_MemoryBurst_Single;
	DMA_INIT.DMA_PeripheralBurst    = DMA_PeripheralBurst_Single;
	DMA_Init(DMA1_Stream5, &DMA_INIT);

	DMA_DoubleBufferModeConfig(fskdac_dev->cfg->dma.tx.channel, (uint32_t)&MARK_SAMPLES[0], DMA_Memory_0);
	DMA_DoubleBufferModeCmd(fskdac_dev->cfg->dma.tx.channel, ENABLE);
	DMA_ITConfig(fskdac_dev->cfg->dma.tx.channel, DMA_IT_TC, ENABLE);

	DMA_Cmd(DMA1_Stream5, ENABLE);
	DAC_Cmd(DAC_Channel_1, ENABLE);
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
}

#endif /* PIOS_INCLUDE_FSK */

/**
  * @}
  * @}
  */
