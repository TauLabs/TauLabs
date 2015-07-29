/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_HMC5883 HMC5883 Functions
 * @brief Deals with the hardware interface to the magnetometers
 * @{
 *
 * @file       pios_hmc5883.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      HMC5883 functions header.
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

#ifndef PIOS_PX4FLOW_H
#define PIOS_PX4FLOW_H

#include <stdint.h>
#include <stdbool.h>

/* PX4FLOW Addresses */
#define PIOS_PX4FLOW_I2C_7_BIT_ADDR                              0x42
#define PIOS_PX4FLOW_FRAMECOUNTER_LSB                            0x00
#define PIOS_PX4FLOW_FRAMECOUNTER_SINCE_LAST_I2C_READING_LSB     0x16

struct Rotation {
	// Units are [deg*100]
	int16_t roll_D100;
	int16_t pitch_D100;
	int16_t yaw_D100;
};

struct pios_px4flow_cfg {
	struct Rotation rotation;
};

struct pios_px4flow_data {
	int16_t x_dot;
	int16_t y_dot;
	int16_t x;
	int16_t y;
};

/* Public Functions */
extern int32_t PIOS_PX4Flow_Init(const struct pios_px4flow_cfg *cfg, const uint32_t i2c_id);
extern int32_t PIOS_PX4Flow_Test(void);
extern int32_t PIOS_PX4Flow_SetRotation(const struct Rotation rotation);
extern bool PIOS_PX4Flow_IRQHandler();
#endif /* PIOS_PX4FLOW_H */

/** 
  * @}
  * @}
  */
