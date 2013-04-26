/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_PCF8591_ADC ADC Functions
 * @brief PCF8591 ADC PIOS interface
 * @{
 *
 * @file       pios_pcf8591_adc.h
 * @author     The Tau Labs Team, http://www.taulabs.org Copyright (C) 2013.
 * @brief      PCF8591 ADC functions header.
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

#ifndef PIOS_PCF8591_ADC_H_
#define PIOS_PCF8591_ADC_H_

#define PIOS_PCF8591_DAC_ENABLE							0x40
#define PIOS_PCF8591_ADC_CH0							0x00
#define PIOS_PCF8591_ADC_CH1							0x01
#define PIOS_PCF8591_ADC_CH2							0x02
#define PIOS_PCF8591_ADC_CH3							0x03
#define PIOS_PCF8591_ADC_AUTO_INCREMENT					0x04
#define PIOS_PCF8591_ADC_SINGLE_ENDED					0x00
#define PIOS_PCF8591_ADC_THREE_DIFF_INPUTS				0x10
#define PIOS_PCF8591_ADC_SINGLE_ENDED_AND_DIFF_MIXED	0X20
#define PIOS_PCF8591_ADC_TWO_DIFF_INPUTS				0X30
#define PIOS_PCF8591_NUMBER_OF_ADC_CHANNELS				4
#define PIOS_PCF8591_CHANNELS { PIOS_PCF8591_ADC_CH0, PIOS_PCF8591_ADC_CH1, PIOS_PCF8591_ADC_CH2, PIOS_PCF8591_ADC_CH3 }

#endif /* PIOS_PCF8591_ADC_H_ */
