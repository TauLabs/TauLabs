/**
 ******************************************************************************
 * @addtogroup PIOS PIOS
 * @{
 * @addtogroup
 * @brief
 * @{
 *
 * @file       pios_dma.c
 * @author     The Tau Labs Team, http://www.taulabls.org Copyright (C) 2013.
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
#define HANDLER(X)      if(pios_dma_handler_map[X][0]==NULL)PIOS_DMA_Default_Handler();for(uint8_t i=0;i<PIOS_DMA_MAX_HANDLERS_PER_CHANNEL;++i){if(pios_dma_handler_map[X][i]==NULL){return;}else{pios_dma_handler_map[X][i]();}}
#include <pios_dma.h>

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

static void PIOS_DMA_11_irq_handler(void){
        HANDLER(0);
}
static void PIOS_DMA_12_irq_handler(void){
        HANDLER(1);
}
static void PIOS_DMA_13_irq_handler(void){
        HANDLER(2);
}
static void PIOS_DMA_14_irq_handler(void){
        HANDLER(3);
}
static void PIOS_DMA_15_irq_handler(void){
        HANDLER(4);
}
static void PIOS_DMA_16_irq_handler(void){
        HANDLER(5);
}
static void PIOS_DMA_17_irq_handler(void){
        HANDLER(6);
}
static void PIOS_DMA_21_irq_handler(void){
        HANDLER(7);
}
static void PIOS_DMA_22_irq_handler(void){
        HANDLER(8);
}
static void PIOS_DMA_23_irq_handler(void){
        HANDLER(9);
}
static void PIOS_DMA_24_irq_handler(void){
        HANDLER(10);
}
static void PIOS_DMA_25_irq_handler(void){
        HANDLER(11);
}
