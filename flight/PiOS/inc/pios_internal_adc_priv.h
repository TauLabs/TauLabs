/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_INTERNAL_ADC Internal ADC Functions
 * @{
 *
 * @file       pios_internal_adc_priv.h
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

#ifndef PIOS_INTERNAL_ADC_PRIV_H
#define PIOS_INTERNAL_ADC_PRIV_H

#include <pios.h>
#include <pios_stm32.h>
#include <pios_internal_adc.h>
#include <fifo_buffer.h>

extern const struct pios_adc_driver pios_internal_adc_driver;

/**
 * A structure that defines an internal ADC pin
 */
struct adc_pin {
	GPIO_TypeDef *port; 	/**< port of the pin ex:GPIOC */
	uint32_t pin; 		/**< port pin number ex:GPIO_Pin_3 */
	uint8_t adc_channel; 	/**< adc channel corresponding to the pin ex:ADC_Channel_9 */
	bool is_master_channel;	/**< true if this pin belongs to the master adc */
};

/**
 * A structure that defines an internal ADC configuration
 */
struct pios_internal_adc_cfg {
	ADC_TypeDef* adc_dev_master;	/**< ADC master ex:ADC1 */
	ADC_TypeDef* adc_dev_slave;	/**< ADC slave to use in ADC dual mode leave unset or NULL for single mode */
	TIM_TypeDef* timer;		/**<  */
	struct stm32_dma dma;		/**< DMA configuration structure */
	uint32_t half_flag;		/**< DMA half buffer flag ex:DMA1_IT_HT1 */
	uint32_t full_flag;		/**< DMA buffer full flag ex:DMA1_IT_TC1 */
	uint16_t max_downsample;	/**< maximum downsample */
	uint32_t oversampling;		/**< oversampling */
	uint8_t number_of_used_pins;	/**< number of pins to use. Normally the size of the adc_pins array */
	struct adc_pin *adc_pins;	/**< pointer to the array of pins to use */
};

extern int32_t PIOS_INTERNAL_ADC_Init(uint32_t * internal_adc_id, const struct pios_internal_adc_cfg * cfg);

#endif /* PIOS_INTERNAL_ADC_PRIV_H */

/**
 * @}
 * @}
 */

