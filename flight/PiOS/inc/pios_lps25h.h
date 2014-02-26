/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MS5611 MS5611 Functions
 * @brief Hardware functions to deal with the altitude pressure sensor
 * @{
 *
 * @file       pios_ms5611.h  
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
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

#ifndef PIOS_LPS25H_H
#define PIOS_LPS25H_H

#include <stdint.h>

// sampling rate
enum pios_lps25h_odr {
	LPS25H_ODR_1HZ   = 1,
	LPS25H_ODR_7HZ   = 2,
	LPS25H_ODR_12HZ  = 3,
	LPS25H_ODR_25HZ  = 4
};

enum pios_lps25h_addr {
	LPS25H_I2C_ADDR_SA0_LOW   = 0x5c,
	LPS25H_I2C_ADDR_SA0_HIGH  = 0x5d
};

struct pios_lps25h_cfg {
	enum pios_lps25h_addr i2c_addr;
	enum pios_lps25h_odr odr;
};

int32_t PIOS_LPS25H_Init(const struct pios_lps25h_cfg *cfg, int32_t i2c_device);

#endif /* PIOS_LPS25H_H */

/** 
  * @}
  * @}
  */
