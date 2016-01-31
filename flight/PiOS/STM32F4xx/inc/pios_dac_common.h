/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
  * @addtogroup   PIOS_DAC DAC common methods
 * @brief PIOS interface for DAC Beep implementation
 * @{
 *
 * @file       pios_dac_common.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      Core methods required for the DAC
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

#ifndef PIOS_DAC_COMMON_H
#define PIOS_DAC_COMMON_H

int32_t PIOS_DAC_COMMON_Init(void (*irq_cb_method)(void));

#endif /* PIOS_DAC_COMMON_H */