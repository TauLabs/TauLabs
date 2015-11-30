/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
  * @addtogroup   PIOS_DACBEEP DAC Beep Code Functions
 * @brief PIOS interface for DAC Beep implementation
 * @{
 *
 * @file       pios_dacbeep.c
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
#if defined(PIOS_INCLUDE_DAC_BEEPS)

#include "pios_fskdac_priv.h"

#if defined(PIOS_INCLUDE_FREERTOS)
#include "FreeRTOS.h"
#endif /* defined(PIOS_INCLUDE_FREERTOS) */

enum pios_dacbeep_dev_magic {
	PIOS_DACBEEP_DEV_MAGIC = 0x3A53834A,
};

struct pios_dacbeep_dev {
	enum pios_dacbeep_dev_magic     magic;
	const struct pios_fskdac_config * cfg;
	int32_t cycles_remaining;
};

// Lookup tables for symbols
const uint16_t SINE_SAMPLES[] = {
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

const uint32_t SAMPLES_PER_BIT = NELEMENTS(SINE_SAMPLES);


#define   DAC_DHR12R1_ADDR  0x40007408

static bool PIOS_DACBEEP_validate(struct pios_dacbeep_dev * dacbeep_dev)
{
	if (dacbeep_dev == NULL)
		return false;

	return (dacbeep_dev->magic == PIOS_DACBEEP_DEV_MAGIC);
}

static struct pios_dacbeep_dev * PIOS_DACBEEP_alloc(void)
{
	struct pios_dacbeep_dev * dacbeep_dev;

	dacbeep_dev = (struct pios_dacbeep_dev *)PIOS_malloc(sizeof(*dacbeep_dev));
	if (!dacbeep_dev) return(NULL);

	memset(dacbeep_dev, 0, sizeof(*dacbeep_dev));
	dacbeep_dev->magic = PIOS_DACBEEP_DEV_MAGIC;
	return(dacbeep_dev);
}

struct pios_dacbeep_dev * g_dacbeep_dev;

/**
* Initialise a single USART device
*/
int32_t PIOS_DACBEEP_Init(uintptr_t * dacbeep_id, const struct pios_fskdac_config * cfg)
{
	PIOS_DEBUG_Assert(dacbeep_id);
	PIOS_DEBUG_Assert(cfg);

	struct pios_dacbeep_dev * dacbeep_dev;

	dacbeep_dev = (struct pios_dacbeep_dev *) PIOS_DACBEEP_alloc();
	if (!dacbeep_dev) return -1;

	// Handle for the IRQ
	g_dacbeep_dev = dacbeep_dev;

	/* Bind the configuration to the device instance */
	dacbeep_dev->cfg = cfg; // TODO: use this

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
	DMA_InitStructure.DMA_PeripheralBaseAddr = 0x40007408; // DAC1 12R register
	DMA_InitStructure.DMA_Memory0BaseAddr = (uint32_t)&SINE_SAMPLES[0];
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

	NVIC_InitTypeDef NVIC_InitStructure;
	NVIC_InitStructure.NVIC_IRQChannel = DMA1_Stream5_IRQn;
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGH;
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
	NVIC_Init(&NVIC_InitStructure);

	dacbeep_dev->cycles_remaining = 50;

	//DMA_ITConfig(dacbeep_dev->cfg->dma.tx.channel, DMA_IT_TC, ENABLE);
	DMA_ITConfig(DMA1_Stream5, DMA_IT_TC, ENABLE);

	/* Enable DMA1_Stream5 */
	DMA_Cmd(DMA1_Stream5, ENABLE);

	/* Enable DAC Channel1 */
	DAC_Cmd(DAC_Channel_1, ENABLE);

	/* Enable DMA for DAC Channel1 */
	DAC_DMACmd(DAC_Channel_1, ENABLE);

	dacbeep_dev->cycles_remaining = 0;

	*dacbeep_id = (uintptr_t) dacbeep_dev;

	return 0;
}

/**
 * Generate a beep (sine wave) of defined frequency and duration
 * @param[in] dacbeep_id opaque handle to the dacbeep device
 * @param[in] freq frequency in Hz for sine wave
 * @param[in] duration_ms duration in milliseconds
 * @return 0 if success, -1 if failure
 */
int32_t PIOS_DACBEEP_Beep(uintptr_t dacbeep_id, uint32_t freq, uint32_t duration_ms)
{
	struct pios_dacbeep_dev * dacbeep_dev = (struct pios_dacbeep_dev *)dacbeep_id;
	
	if (!PIOS_DACBEEP_validate(dacbeep_dev))
		return -1;

	// Do not start a beep while previous is running
	if (dacbeep_dev->cycles_remaining)
		return -1;

	TIM_Cmd(TIM6, DISABLE);
	TIM_TimeBaseInitTypeDef TIM6_TimeBase;
	TIM_TimeBaseStructInit(&TIM6_TimeBase); 
	TIM6_TimeBase.TIM_Period        = (PIOS_PERIPHERAL_APB1_CLOCK / (freq * SAMPLES_PER_BIT));
	TIM6_TimeBase.TIM_Prescaler     = 0;
	TIM6_TimeBase.TIM_ClockDivision = 0;
	TIM6_TimeBase.TIM_CounterMode   = TIM_CounterMode_Up;
	TIM_TimeBaseInit(TIM6, &TIM6_TimeBase);
	TIM_SelectOutputTrigger(TIM6, TIM_TRGOSource_Update);

	int32_t cycles = (freq * duration_ms) / 1000;
	dacbeep_dev->cycles_remaining = cycles;

	DMA_ITConfig(dacbeep_dev->cfg->dma.tx.channel, DMA_IT_TC, ENABLE);
	TIM_Cmd(TIM6, ENABLE);

	return 0;
}

// Should be aliased from DMA1_Stream5_IRQHandler
void PIOS_DACBEEP_DMA_irq_handler()
{
	if (DMA_GetITStatus(DMA1_Stream5, DMA_IT_TCIF5)) {
		DMA_ClearITPendingBit(DMA1_Stream5, DMA_IT_TCIF5);
	
		struct pios_dacbeep_dev * dacbeep_dev = g_dacbeep_dev;

		bool valid = PIOS_DACBEEP_validate(dacbeep_dev);
		PIOS_Assert(valid);

		if (dacbeep_dev->cycles_remaining == 0) {
			TIM_Cmd(TIM6, DISABLE);
			DMA_ITConfig(dacbeep_dev->cfg->dma.tx.channel, DMA_IT_TC, DISABLE);
		} else {
			dacbeep_dev->cycles_remaining--;
		}
	}

#if defined(PIOS_INCLUDE_FREERTOS)
	portEND_SWITCHING_ISR((rx_need_yield || tx_need_yield) ? pdTRUE : pdFALSE);
#endif	/* defined(PIOS_INCLUDE_FREERTOS) */
}
#endif /* PIOS_INCLUDE_DAC_BEEPS */

/**
  * @}
  * @}
  */
