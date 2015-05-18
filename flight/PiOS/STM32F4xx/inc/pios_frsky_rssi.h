/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS driver for PWM RSSI input
 * @brief Input RSSI PWM
 * @{
 *
 * @file       pios_frsky_rssi.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2015.
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

#ifndef PIOS_FRSKY_RSSI_H
#define PIOS_FRSKY_RSSI_H

#include <pios_stm32.h>
#include <stm32f4xx_tim.h>

struct pios_frsky_rssi_cfg {
	struct pios_tim_clock_cfg clock_cfg;
	struct pios_tim_channel channels[2];
	TIM_ICInitTypeDef ic1;
	TIM_ICInitTypeDef ic2;
};

int32_t PIOS_FrSkyRssi_Init(const struct pios_frsky_rssi_cfg * cfg_in);
uint16_t PIOS_FrSkyRssi_Get();

#endif /* PIOS_FRSKY_RSSI_H */
