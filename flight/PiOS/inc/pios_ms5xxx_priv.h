/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MS5XXX MS5XXX Functions
 * @brief Hardware functions to deal with the altitude pressure sensor
 * @{
 *
 * @file       pios_ms5xxx_priv.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      MS5XXX functions header.
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

#ifndef PIOS_MS5XXX_PRIV_H
#define PIOS_MS5XXX_PRIV_H

#define MS5XXX_RESET            0x1E
#define MS5XXX_CALIB_ADDR       0xA2  /* First sample is factory stuff */
#define MS5XXX_CALIB_LEN        16
#define MS5XXX_ADC_READ         0x00
#define MS5XXX_PRES_ADDR        0x40
#define MS5XXX_TEMP_ADDR        0x50
#define MS5XXX_ADC_MSB          0xF6

//#define PIOS_MS5XXX_OVERSAMPLING oversampling

//! The valid oversampling rates
enum pios_ms5xxx_osr {
	MS5XXX_OSR_256   = 0,
	MS5XXX_OSR_512   = 2,
	MS5XXX_OSR_1024  = 4,
	MS5XXX_OSR_2048  = 6,
	MS5XXX_OSR_4096  = 8,
};

//! The valid MS5XXX models
enum PIOS_MS5XXX_MODEL {
	PIOS_MS5M_MS5611 = 0x01, // Start at 1 so that 0 is undefined. This will be used as a trap for unconfigured config structs.
	PIOS_MS5M_MS5637,
	PIOS_MS5M_MS5803_01,
	PIOS_MS5M_MS5803_02,
};

//! Valid MS5xxx Addresses
enum MS5XXX_I2C_ADDRESS {
	MS5XXX_I2C_ADDR_0x76 = 0x76,
	MS5XXX_I2C_ADDR_0x77 = 0x77,
};

//! Configuration structure for the MS5XXX driver
struct pios_ms5xxx_cfg {
	//! The oversampling setting for the baro, higher produces
	//! less frequenct cleaner data
	enum pios_ms5xxx_osr oversampling;

	//! How many samples of pressure for each temperature measurement
	uint32_t temperature_interleaving;
	
	//! MS5XXX model
	enum PIOS_MS5XXX_MODEL pios_ms5xxx_model;
};

/* Public Functions */
int32_t PIOS_MS5XXX_Test(void);

#if defined(PIOS_INCLUDE_I2C)
int32_t PIOS_MS5XXX_I2C_Init(int32_t i2c_bus_id, enum MS5XXX_I2C_ADDRESS i2c_address, const struct pios_ms5xxx_cfg *cfg);
#endif  // defined(PIOS_INCLUDE_I2C)

#if defined(PIOS_INCLUDE_SPI)
int32_t PIOS_MS5XXX_SPI_Init(uint32_t spi_bus_id, uint32_t slave_num, const struct pios_ms5xxx_cfg *cfg);
#endif  // defined(PIOS_INCLUDE_SPI)

#endif /* PIOS_MS5XXX_PRIV_H */

/** 
  * @}
  * @}
  */
