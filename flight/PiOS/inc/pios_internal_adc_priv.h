/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_INTERNAL_ADC ADC Functions
 * @brief PIOS interface for INTERNAL ADC port
 * @{
 *
 * @file       pios_internal_adc_priv.h
 * @author     The Tau Labs Team, http://www.taulabs.org Copyright (C) 2013.
 * @brief      ADC private definitions.
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

#ifndef PIOS_INTERNAL_ADC_PRIV_H
#define PIOS_INTERNAL_ADC_PRIV_H

#include <pios.h>
#include <pios_stm32.h>
#include <pios_internal_adc.h>
#include <fifo_buffer.h>

extern const struct pios_adc_driver pios_internal_adc_driver;

struct adc_pin {
	GPIO_TypeDef *port;
	uint32_t pin;
	uint8_t adc_channel;
	bool is_master_channel;
};

struct pios_internal_adc_cfg {
	ADC_TypeDef* adc_dev_master;
	ADC_TypeDef* adc_dev_slave;
	TIM_TypeDef* timer;
	struct stm32_dma dma;
	uint32_t half_flag;
	uint32_t full_flag;
	uint16_t max_downsample;
	uint32_t oversampling;
	uint8_t number_of_used_pins;
	struct adc_pin *adc_pins;
};

extern int32_t PIOS_INTERNAL_ADC_Init(uint32_t * internal_adc_id, const struct pios_internal_adc_cfg * cfg);

#endif /* PIOS_INTERNAL_ADC_PRIV_H */

/**
 * @}
 * @}
 */

