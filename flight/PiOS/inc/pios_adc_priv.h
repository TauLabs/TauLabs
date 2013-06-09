/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_ADC ADC Functions
 * @{
 *
 * @file       pios_adc_priv.h
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

#ifndef PIOS_ADC_PRIV_H
#define PIOS_ADC_PRIV_H

#include <pios_adc.h>

extern int32_t PIOS_ADC_Init(uintptr_t * adc_id, const struct pios_adc_driver * driver, uint32_t lower_id);

#endif /* PIOS_ADC_PRIV_H */

/**
 * @}
 * @}
 */

