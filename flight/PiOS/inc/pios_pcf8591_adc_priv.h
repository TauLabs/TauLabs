/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_PCF8591_ADC ADC Functions
 * @brief PIOS driver for PCF8591 ADC converter
 * @{
 *
 * @file       pios_pcf8591_adc_priv.h
 * @author     The Tau Labs Team, http://www.taulabs.org Copyright (C) 2013.
 * @brief      PCF8591 ADC private definitions.
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

#ifndef PIOS_PCF8591_ADC_PRIV_H_
#define PIOS_PCF8591_ADC_PRIV_H_

#include <pios.h>
#include <pios_stm32.h>
#include <pios_pcf8591_adc.h>

extern const struct pios_adc_driver pios_pcf8591_adc_driver;

struct pios_pcf8591_adc_cfg {
	uint8_t	i2c_adress;
	bool	use_auto_increment;
	bool	enable_dac;
	uint8_t	adc_input_type;
};

extern int32_t PIOS_PCF8591_ADC_Init(uint32_t * pcf8591_adc_id, const struct pios_pcf8591_adc_cfg * cfg);

#endif /* PIOS_PCF8591_ADC_PRIV_H_ */
