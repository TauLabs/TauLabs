/**
 ******************************************************************************
 * @addtogroup PIOS PIOS
 * @{
 * @addtogroup
 * @brief
 * @{
 *
 * @file       pios_dma_hooks.c
 * @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
 * @brief
 * @see        The GNU Public License (GPL) Version 3
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
#include <pios_dma.h>
#if defined(PIOS_INCLUDE_DMA_CB_SUBSCRIBING_FUNCTION)

static void PIOS_DMA_11_irq_handler(void);
static void PIOS_DMA_12_irq_handler(void);
static void PIOS_DMA_13_irq_handler(void);
static void PIOS_DMA_14_irq_handler(void);
static void PIOS_DMA_15_irq_handler(void);
static void PIOS_DMA_16_irq_handler(void);
static void PIOS_DMA_17_irq_handler(void);
static void PIOS_DMA_21_irq_handler(void);
static void PIOS_DMA_22_irq_handler(void);
static void PIOS_DMA_23_irq_handler(void);
static void PIOS_DMA_24_irq_handler(void);
static void PIOS_DMA_25_irq_handler(void);

void DMA1_Channel1_IRQHandler() __attribute__ ((alias ("PIOS_DMA_11_irq_handler")));
void DMA1_Channel2_IRQHandler() __attribute__ ((alias ("PIOS_DMA_12_irq_handler")));
void DMA1_Channel3_IRQHandler() __attribute__ ((alias ("PIOS_DMA_13_irq_handler")));
void DMA1_Channel4_IRQHandler() __attribute__ ((alias ("PIOS_DMA_14_irq_handler")));
void DMA1_Channel5_IRQHandler() __attribute__ ((alias ("PIOS_DMA_15_irq_handler")));
void DMA1_Channel6_IRQHandler() __attribute__ ((alias ("PIOS_DMA_16_irq_handler")));
void DMA1_Channel7_IRQHandler() __attribute__ ((alias ("PIOS_DMA_17_irq_handler")));
void DMA2_Channel1_IRQHandler() __attribute__ ((alias ("PIOS_DMA_21_irq_handler")));
void DMA2_Channel2_IRQHandler() __attribute__ ((alias ("PIOS_DMA_22_irq_handler")));
void DMA2_Channel3_IRQHandler() __attribute__ ((alias ("PIOS_DMA_23_irq_handler")));
void DMA2_Channel4_IRQHandler() __attribute__ ((alias ("PIOS_DMA_24_irq_handler")));
void DMA2_Channel5_IRQHandler() __attribute__ ((alias ("PIOS_DMA_25_irq_handler")));

/**
 * @brief calls the handlers associated with a given index (from PIOS_DMA_CHANNELS)
 * @param[in] DMA channel index (PIOS_DMA_CHANNELS)
 */
static inline void PIOS_DMA_Mapper(uint32_t index)
{
	if (pios_dma_handler_map[index][0] == NULL )
		PIOS_DMA_Default_Handler();
	for (uint32_t i = 0; i < PIOS_DMA_MAX_HANDLERS_PER_CHANNEL; ++i) {
		if (pios_dma_handler_map[index][i] == NULL )
			return;
		else {
			pios_dma_handler_map[index][i]();
		}
	}
}

static void PIOS_DMA_11_irq_handler(void){
	PIOS_DMA_Mapper(0);
}
static void PIOS_DMA_12_irq_handler(void){
	PIOS_DMA_Mapper(1);
}
static void PIOS_DMA_13_irq_handler(void){
	PIOS_DMA_Mapper(2);
}
static void PIOS_DMA_14_irq_handler(void){
	PIOS_DMA_Mapper(3);
}
static void PIOS_DMA_15_irq_handler(void){
	PIOS_DMA_Mapper(4);
}
static void PIOS_DMA_16_irq_handler(void){
	PIOS_DMA_Mapper(5);
}
static void PIOS_DMA_17_irq_handler(void){
	PIOS_DMA_Mapper(6);
}
static void PIOS_DMA_21_irq_handler(void){
	PIOS_DMA_Mapper(7);
}
static void PIOS_DMA_22_irq_handler(void){
	PIOS_DMA_Mapper(8);
}
static void PIOS_DMA_23_irq_handler(void){
	PIOS_DMA_Mapper(9);
}
static void PIOS_DMA_24_irq_handler(void){
	PIOS_DMA_Mapper(10);
}
static void PIOS_DMA_25_irq_handler(void){
	PIOS_DMA_Mapper(11);
}
#endif /* PIOS_INCLUDE_DMA_CB_SUBSCRIBING_FUNCTION */
