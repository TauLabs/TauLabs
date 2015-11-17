/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_INTERNAL_ADC Internal ADC Functions
 * @{
 *
 * @file       pios_internal_adc_light_priv.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

#ifndef PIOS_INTERNAL_ADC_LIGHT_PRIV_H
#define PIOS_INTERNAL_ADC_LIGHT_PRIV_H

#include <pios.h>
#include <pios_stm32.h>
#include <pios_internal_adc.h>
#include <pios_internal_adc_priv.h>
#include <fifo_buffer.h>

extern int32_t PIOS_INTERNAL_ADC_LIGHT_Init(uint32_t *internal_adc_id,
		const struct pios_internal_adc_cfg *cfg,
		uint16_t number_of_used_pins);

#endif /* PIOS_INTERNAL_ADC_LIGHT_PRIV_H */

/**
 * @}
 * @}
 */

