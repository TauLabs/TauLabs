/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_ADC ADC layer functions
 * @brief Upper level Analog to Digital converter layer
 * @{
 *
 * @file       pios_adc.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      ADC layer functions header
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

#ifndef PIOS_ADC_H
#define PIOS_ADC_H

#include <stdint.h>		/* uint*_t */
#include <stdbool.h>	/* bool */

struct pios_adc_driver {
	void (*init)(uint32_t id);
	int32_t (*get_pin)(uint32_t id, uint32_t pin);
	bool (*available)(uint32_t id, uint32_t device_pin);
#if defined(PIOS_INCLUDE_FREERTOS)
	void (*set_queue)(uint32_t id, xQueueHandle data_queue);
#endif
	uint8_t (*number_of_channels)(uint32_t id);
	float (*lsb_voltage)(uint32_t id);
};

/* Public Functions */
extern int32_t PIOS_ADC_DevicePinGet(uintptr_t adc_id, uint32_t device_pin);
extern bool PIOS_ADC_Available(uintptr_t adc_id, uint32_t device_pin);
#if defined(PIOS_INCLUDE_FREERTOS)
extern void PIOS_ADC_SetQueue(uintptr_t adc_id, xQueueHandle data_queue);
#endif
extern int32_t PIOS_ADC_GetChannelRaw(uint32_t channel);
extern float PIOS_ADC_GetChannelVolt(uint32_t channel);
#endif /* PIOS_ADC_H */

/**
  * @}
  * @}
  */
