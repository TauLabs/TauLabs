/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
  * @addtogroup   PIOS_DAC DAC common methods
 * @brief PIOS interface for DAC Beep implementation
 * @{
 *
 * @file       pios_dac_common.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      Core methods required for the DAC
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
#if defined(PIOS_INCLUDE_DAC_BEEPS) || defined(PIOS_INCLUDE_FSK)

static void (*pios_dac_irq_cb)(void);

//! The DAC hardware on STM32F4 doesn't have much flexibility
int32_t PIOS_DAC_COMMON_Init(void (*irq_cb_method)(void))
{
	// Save IRQ callback reference
	pios_dac_irq_cb = irq_cb_method;

	// Initialize GPIO for DAC
	GPIO_InitTypeDef gpio_init;
	gpio_init.GPIO_Pin  = GPIO_Pin_4;
	gpio_init.GPIO_Mode = GPIO_Mode_AN;
	gpio_init.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOA, &gpio_init);

	// Enable all the timers
	TIM_TimeBaseInitTypeDef TIM6_TimeBase;
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM6, ENABLE);
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_DAC, ENABLE);
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_DMA1, ENABLE);

	// Configure TIM6 as the time base for DMA->DAC clock
	TIM_TimeBaseStructInit(&TIM6_TimeBase); 
	TIM6_TimeBase.TIM_Period        = 1000; // This will be loaded in the higher level driver
	TIM6_TimeBase.TIM_Prescaler     = 0;
	TIM6_TimeBase.TIM_ClockDivision = 0;
	TIM6_TimeBase.TIM_CounterMode   = TIM_CounterMode_Up;
	TIM_TimeBaseInit(TIM6, &TIM6_TimeBase);
	TIM_SelectOutputTrigger(TIM6, TIM_TRGOSource_Update);
	TIM_Cmd(TIM6, ENABLE);

	// Enable DAC to be clocked by TIM6
	DAC_InitTypeDef DAC_INIT;
	DAC_StructInit(&DAC_INIT);
	DAC_DeInit();
	DAC_INIT.DAC_Trigger        = DAC_Trigger_T6_TRGO;
	DAC_INIT.DAC_WaveGeneration = DAC_WaveGeneration_None;
	DAC_INIT.DAC_OutputBuffer   = DAC_OutputBuffer_Enable;
	DAC_Init(DAC_Channel_1, &DAC_INIT);

	// Skip configuring DMA since that is specific to the lower level

	// Enable IRQ when DMA transfer is complete
	NVIC_InitTypeDef NVIC_InitStructure;
	NVIC_InitStructure.NVIC_IRQChannel = DMA1_Stream5_IRQn;
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_LOW;
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
	NVIC_Init(&NVIC_InitStructure);

	return 0;
}

// Calls into specific streaming driver IRQ
void DMA1_Stream5_IRQHandler(void)
{
#if defined(PIOS_INCLUDE_CHIBIOS)
	CH_IRQ_PROLOGUE();
#endif /* defined(PIOS_INCLUDE_CHIBIOS) */

	if (DMA_GetITStatus(DMA1_Stream5, DMA_IT_TCIF5)) {
		DMA_ClearITPendingBit(DMA1_Stream5, DMA_IT_TCIF5);

		// call the higher level IRQ callback
		pios_dac_irq_cb();
	}

#if defined(PIOS_INCLUDE_FREERTOS)
	portEND_SWITCHING_ISR((rx_need_yield || tx_need_yield) ? pdTRUE : pdFALSE);
#endif	/* defined(PIOS_INCLUDE_FREERTOS) */

#if defined(PIOS_INCLUDE_CHIBIOS)
	CH_IRQ_EPILOGUE();
#endif /* defined(PIOS_INCLUDE_CHIBIOS) */
}

#endif /*  PIOS_INCLUDE_DAC_BEEPS || PIOS_INCLUDE_FSK  */

