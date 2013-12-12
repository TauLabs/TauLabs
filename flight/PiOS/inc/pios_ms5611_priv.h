/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MS5611 MS5611 Functions
 * @brief Hardware functions to deal with the altitude pressure sensor
 * @{
 *
 * @file       pios_ms5611_priv.h  
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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

//! The valid oversampling rates
enum pios_ms5611_osr {
	MS5611_OSR_256   = 0,
	MS5611_OSR_512   = 2,
	MS5611_OSR_1024  = 4,
	MS5611_OSR_2048  = 6,
	MS5611_OSR_4096  = 8,
};

//! Configuration structure for the MS5611 driver
struct pios_ms5611_cfg {
	//! The oversampling setting for the baro, higher produces
	//! less frequenct cleaner data
	enum pios_ms5611_osr oversampling;

	//! How many samples of pressure for each temperature measurement
	uint32_t temperature_interleaving;
};

int32_t PIOS_MS5611_Init(const struct pios_ms5611_cfg * cfg, int32_t i2c_device);
int32_t PIOS_MS5611_SPI_Init(uint32_t spi_id, uint32_t slave_num, const struct pios_ms5611_cfg *cfg);

#endif /* PIOS_MS5611_PRIV_H */

/** 
  * @}
  * @}
  */
