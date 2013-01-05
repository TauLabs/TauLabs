/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MS5611 MS5611 Functions
 * @brief Hardware functions to deal with the altitude pressure sensor
 * @{
 *
 * @file       pios_ms5611_priv.h  
 * @author     PhoenixPilot, http://github.com/PhoenixPilot Copyright (C) 2013.
 * @brief      MS5611 functions header.
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

#ifndef PIOS_MS5611_PRIV_H
#define PIOS_MS5611_PRIV_H

struct pios_ms5611_cfg {
	uint32_t oversampling;
	uint32_t temperature_interleaving;
};

int32_t PIOS_MS5611_Init(const struct pios_ms5611_cfg * cfg, int32_t i2c_device);

#endif /* PIOS_MS5611_PRIV_H */

/** 
  * @}
  * @}
  */
