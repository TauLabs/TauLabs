/**
 ******************************************************************************
 * @addtogroup PIOS PIOS
 * @{
 * @addtogroup
 * @brief
 * @{
 *
 * @file       pios_dma.h
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
#ifndef PIOS_DMA_H_
#define PIOS_DMA_H_

#include <pios.h>
#if defined(PIOS_INCLUDE_DMA_CB_SUBSCRIBING_FUNCTION)
extern void PIOS_DMA_Default_Handler();
extern int8_t PIOS_DMA_Install_Interrupt_handler(DMA_Channel_TypeDef *channel, void * function);
extern void (*pios_dma_handler_map[PIOS_DMA_MAX_CHANNELS][PIOS_DMA_MAX_HANDLERS_PER_CHANNEL])(void);
#endif /* PIOS_INCLUDE_DMA_CB_SUBSCRIBING_FUNCTION */
#endif /* PIOS_DMA_H_ */
