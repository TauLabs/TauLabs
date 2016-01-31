/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
  * @addtogroup   PIOS_DACBEEP DAC Beep Code Functions
 * @brief PIOS interface for DAC Beep implementation
 * @{
 *
 * @file       pios_dacbeep.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015-2016
 * @brief      Generates audio beeps of desired duration and frequency
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

#if defined(PIOS_INCLUDE_FREERTOS)
#include "FreeRTOS.h"
#endif /* defined(PIOS_INCLUDE_FREERTOS) */

#include "pios_dac_common.h"

enum pios_dacbeep_dev_magic {
	PIOS_DACBEEP_DEV_MAGIC = 0x3A53834A,
};

struct pios_dacbeep_dev {
	enum pios_dacbeep_dev_magic     magic;
	int32_t cycles_remaining;
};

// Lookup tables for symbols
// from ML designed to generate 0.4V p-p centered at 0.5V
//   sprintf('%d,', round((sin(linspace(0,2*pi,130))*(2^12/3.3*0.2)+(2^12/3.3*0.5))))
const uint16_t SINE_SAMPLES[] = {
	621,633,645,657,669,680,692,704,715,726,
	737,747,758,767,777,786,795,803,811,819,
	826,833,839,844,849,854,857,861,864,866,
	867,868,869,869,868,867,865,862,859,856,
	851,847,841,836,829,822,815,807,799,791,
	782,772,763,752,742,731,720,709,698,686,
	675,663,651,639,627,615,602,590,578,567,
	555,543,532,521,510,499,489,479,469,460,
	450,442,434,426,419,412,406,400,395,390,
	386,382,379,376,375,373,373,372,373,374,
	375,378,380,384,388,392,397,403,409,415,
	422,430,438,446,455,464,474,484,494,504,
	515,526,538,549,561,573,584,596,609,621,}; // 10x13 samples

const uint32_t SAMPLES_PER_BIT = NELEMENTS(SINE_SAMPLES);

// Local method definitions
static void PIOS_DACBEEP_DMA_irq_cb();

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
int32_t PIOS_DACBEEP_Init(uintptr_t * dacbeep_id)
{
	PIOS_DEBUG_Assert(dacbeep_id);
	PIOS_DEBUG_Assert(cfg);

	struct pios_dacbeep_dev * dacbeep_dev;

	dacbeep_dev = (struct pios_dacbeep_dev *) PIOS_DACBEEP_alloc();
	if (!dacbeep_dev) return -1;

	// Handle for the IRQ
	g_dacbeep_dev = dacbeep_dev;

	/* Initialize DAC hardware */
	PIOS_DAC_COMMON_Init(PIOS_DACBEEP_DMA_irq_cb);

	dacbeep_dev->cycles_remaining = 50;

	/* Configure the DMA system to use DMA1 Stream5 */
	DMA_DeInit(DMA1_Stream5);
	DMA_InitTypeDef DMA_InitStructure;	
	DMA_InitStructure.DMA_Channel = DMA_Channel_7;  
	DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)&DAC->DHR12R1;
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

	/* Enable transfer complete interrupt for this stream */
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

	DMA_ITConfig(DMA1_Stream5, DMA_IT_TC, ENABLE);
	TIM_Cmd(TIM6, ENABLE);

	return 0;
}

static void PIOS_DACBEEP_DMA_irq_cb()
{	
	struct pios_dacbeep_dev * dacbeep_dev = g_dacbeep_dev;

	bool valid = PIOS_DACBEEP_validate(dacbeep_dev);
	PIOS_Assert(valid);

	if (dacbeep_dev->cycles_remaining == 0) {
		TIM_Cmd(TIM6, DISABLE);
		DMA_ITConfig(DMA1_Stream5, DMA_IT_TC, DISABLE);
	} else {
		dacbeep_dev->cycles_remaining--;
	}
}
#endif /* PIOS_INCLUDE_DAC_BEEPS */

/**
  * @}
  * @}
  */
