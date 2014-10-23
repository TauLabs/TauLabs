/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_BMP085 BMP085 Functions
 * @brief Hardware functions to deal with the altitude pressure sensor
 * @{
 *
 * @file       pios_bmp085_priv.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      BMP085 functions header.
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

#ifndef PIOS_BMP085_PRIV_H
#define PIOS_BMP085_PRIV_H

//! The valid oversampling rates
enum pios_bmp085_osr {
    BMP085_OSR_0  = 0,
    BMP085_OSR_1  = 1,
    BMP085_OSR_2  = 2,
    BMP085_OSR_3  = 3,
};

//! Configuration structure for the BMP085 driver
struct pios_bmp085_cfg {
    //! The oversampling setting for the baro, higher produces
    //! less frequenct cleaner data
    enum pios_bmp085_osr oversampling;

    //! How many samples of pressure for each temperature measurement
    uint32_t temperature_interleaving;
};

int32_t PIOS_BMP085_Init(const struct pios_bmp085_cfg *cfg, int32_t i2c_device);


#endif /* PIOS_BMP085_PRIV_H */

/** 
  * @}
  * @}
  */
