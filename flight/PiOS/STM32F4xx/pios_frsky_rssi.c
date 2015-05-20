/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS driver for FrSky RSSI input
 * @brief Driver for FrSky RSSI, which is a PWM signal with a frequency of
 * about 110kHz. The driver uses two capture-compare channels of the timer
 * to measure the PWM pulse width. In contrast to the normal PWM driver, this
 * procedure does not require any interrupts.
 * @{
 *
 * @file       pios_frsky_rssi.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015.
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

#include "pios_config.h"

#if defined(PIOS_INCLUDE_FRSKY_RSSI)

#include "pios.h"
#include <pios_stm32.h>
#include "stm32f4xx_tim.h"
#include "pios_tim_priv.h"

#include "pios_frsky_rssi_priv.h"

static const struct pios_frsky_rssi_cfg * cfg = NULL;

int32_t PIOS_FrSkyRssi_Init(const struct pios_frsky_rssi_cfg * cfg_in)
{
	PIOS_Assert(cfg_in);
	cfg = cfg_in;

	// this only works with the advanced timers
	if ((cfg->channels[0].timer != TIM1) && (cfg->channels[0].timer != TIM8))
		PIOS_Assert(0);

	// both channels need to be on the same timer
	if (cfg->channels[0].timer != cfg->channels[1].timer)
		PIOS_Assert(0);

	// we can only use channels 1 and 2
	if ((cfg->channels[0].timer_chan != TIM_Channel_1) && (cfg->channels[0].timer_chan != TIM_Channel_2))
		PIOS_Assert(0);

	if ((cfg->channels[1].timer_chan != TIM_Channel_1) && (cfg->channels[1].timer_chan != TIM_Channel_2))
		PIOS_Assert(0);

	// Configure timer clock
	PIOS_TIM_InitClock(&cfg->clock_cfg);

	// Setup channels
	uintptr_t tim_id;
	if (PIOS_TIM_InitChannels(&tim_id, cfg->channels, 2, NULL, (uintptr_t)cfg)) {
		return -1;
	}

	// Configure the input capture channels
	TIM_ICInitTypeDef TIM_ICInitStructure = cfg->ic1;
	TIM_ICInitStructure.TIM_Channel = cfg->channels[0].timer_chan;
	TIM_ICInit(cfg->channels[0].timer, &TIM_ICInitStructure);

	TIM_ICInitStructure = cfg->ic2;
	TIM_ICInitStructure.TIM_Channel = cfg->channels[1].timer_chan;
	TIM_ICInit(cfg->channels[1].timer, &TIM_ICInitStructure);

	// slave mode and trigger configuration
	TIM_SelectSlaveMode(cfg->channels[0].timer, TIM_SlaveMode_Reset);

	switch(cfg->channels[0].timer_chan) {
		case TIM_Channel_1:
			TIM_SelectInputTrigger(cfg->channels[0].timer, TIM_TS_TI1FP1);
			break;
		case TIM_Channel_2:
			TIM_SelectInputTrigger(cfg->channels[0].timer, TIM_TS_TI2FP2);
			break;
	}

	// Enable CC channels
	TIM_CCxCmd(cfg->channels[0].timer, cfg->channels[0].timer_chan, TIM_CCx_Enable);
	TIM_CCxCmd(cfg->channels[1].timer, cfg->channels[1].timer_chan, TIM_CCx_Enable);

	return 0;
}

uint16_t PIOS_FrSkyRssi_Get()
{
	uint16_t raw_rssi;
	if (cfg == NULL)
		return 0;

	// test if no new capture occured
	if ((cfg->channels[0].timer->SR & 0x02) == 0)
		return 0;

	raw_rssi = cfg->channels[0].timer->CCR1;

	return raw_rssi;
}
#endif /* PIOS_INCLUDE_FRSKY_RSSI */



