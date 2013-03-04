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

#include <pios_dma_priv.h>

static void PIOS_DMA_Default_Handler(void);

funcPtr pios_dma_handler_map[PIOS_DMA_MAX_CHANNELS]={PIOS_DMA_Default_Handler};

const static DMA_Channel_TypeDef * dma_channels[] = PIOS_DMA_CHANNELS;

static void PIOS_DMA_Default_Handler(void){
        while(true){};
}

void PIOS_DMA_Install_Hook(DMA_Channel_TypeDef *channel, void * function){
        for(uint8_t i=0;i<PIOS_DMA_MAX_CHANNELS;++i){
                if(dma_channels[i]==channel){
                        pios_dma_handler_map[i]=function;
                }
        }
}
