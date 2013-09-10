/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_BRUSHLESS Brushless gimbal driver control
 * @{
 *
 * @file       pios_brushless_priv.h
 * @author     Tau Labs, http://github.com/TauLabs Copyright (C) 2013.
 * @brief      Brushless gimbal controller
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

#ifndef PIOS_BRUSHLESS_PRIV_H
#define PIOS_BRUSHLESS_PRIV_H

#include <pios.h>
#include <pios_tim_priv.h>

struct pios_brushless_cfg {
	TIM_TimeBaseInitTypeDef tim_base_init;
	TIM_OCInitTypeDef tim_oc_init;
	GPIO_InitTypeDef gpio_init;
	uint32_t remap;
	const struct stm32_gpio enables[3];
	const struct pios_tim_channel * channels;
	uint8_t num_channels;
};

extern int32_t PIOS_Brushless_Init(const struct pios_brushless_cfg * cfg);

#endif /* PIOS_BRUSHLESS_PRIV_H */

/**
 * @}
 * @}
 */
