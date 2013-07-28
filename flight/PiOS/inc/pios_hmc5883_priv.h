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

#ifndef PIOS_HMC5883_H
#define PIOS_HMC5883_H

#include <stdint.h>
#include <stdbool.h>

/* HMC5883 Addresses */
#define PIOS_HMC5883_I2C_ADDR			0x1E
#define PIOS_HMC5883_I2C_READ_ADDR      0x3D
#define PIOS_HMC5883_I2C_WRITE_ADDR     0x3C
#define PIOS_HMC5883_CONFIG_REG_A		(uint8_t)0x00
#define PIOS_HMC5883_CONFIG_REG_B		(uint8_t)0x01
#define PIOS_HMC5883_MODE_REG			(uint8_t)0x02
#define PIOS_HMC5883_DATAOUT_XMSB_REG		0x03
#define PIOS_HMC5883_DATAOUT_XLSB_REG		0x04
#define PIOS_HMC5883_DATAOUT_ZMSB_REG		0x05
#define PIOS_HMC5883_DATAOUT_ZLSB_REG		0x06
#define PIOS_HMC5883_DATAOUT_YMSB_REG		0x07
#define PIOS_HMC5883_DATAOUT_YLSB_REG		0x08
#define PIOS_HMC5883_DATAOUT_STATUS_REG		0x09
#define PIOS_HMC5883_DATAOUT_IDA_REG		0x0A
#define PIOS_HMC5883_DATAOUT_IDB_REG		0x0B
#define PIOS_HMC5883_DATAOUT_IDC_REG		0x0C

/* Output Data Rate */
#define PIOS_HMC5883_ODR_0_75		0x00
#define PIOS_HMC5883_ODR_1_5		0x04
#define PIOS_HMC5883_ODR_3			0x08
#define PIOS_HMC5883_ODR_7_5		0x0C
#define PIOS_HMC5883_ODR_15			0x10
#define PIOS_HMC5883_ODR_30			0x14
#define PIOS_HMC5883_ODR_75			0x18

/* Measure configuration */
#define PIOS_HMC5883_MEASCONF_NORMAL		0x00
#define PIOS_HMC5883_MEASCONF_BIAS_POS		0x01
#define PIOS_HMC5883_MEASCONF_BIAS_NEG		0x02

/* Gain settings */
#define PIOS_HMC5883_GAIN_0_88			0x00
#define PIOS_HMC5883_GAIN_1_3			0x20
#define PIOS_HMC5883_GAIN_1_9			0x40
#define PIOS_HMC5883_GAIN_2_5			0x60
#define PIOS_HMC5883_GAIN_4_0			0x80
#define PIOS_HMC5883_GAIN_4_7			0xA0
#define PIOS_HMC5883_GAIN_5_6			0xC0
#define PIOS_HMC5883_GAIN_8_1			0xE0

/* Modes */
#define PIOS_HMC5883_MODE_CONTINUOUS	0x00
#define PIOS_HMC5883_MODE_SINGLE		0x01
#define PIOS_HMC5883_MODE_IDLE			0x02
#define PIOS_HMC5883_MODE_SLEEP			0x03

/* Sensitivity Conversion Values */
enum pios_hmc5883_sensitivity {
	PIOS_HMC5883_Sensitivity_0_88Ga = 1370, // LSB/Ga
	PIOS_HMC5883_Sensitivity_1_3Ga  = 1090, // LSB/Ga
	PIOS_HMC5883_Sensitivity_1_9Ga  = 820, // LSB/Ga
	PIOS_HMC5883_Sensitivity_2_5Ga  = 660, // LSB/Ga
	PIOS_HMC5883_Sensitivity_4_0Ga  = 440, // LSB/Ga
	PIOS_HMC5883_Sensitivity_4_7Ga  = 390, // LSB/Ga
	PIOS_HMC5883_Sensitivity_5_6Ga  = 330, // LSB/Ga
	PIOS_HMC5883_Sensitivity_8_1Ga  = 230, // LSB/Ga  --> NOT RECOMMENDED
};

enum pios_hmc5883_orientation {
	// clockwise rotation from board forward while looking at top side
	// 0 degree is chip mark on upper left corner
	PIOS_HMC5883_TOP_0DEG,
	PIOS_HMC5883_TOP_90DEG,
	PIOS_HMC5883_TOP_180DEG,
	PIOS_HMC5883_TOP_270DEG,
	// clockwise rotation from board forward while looking at bottom side
	// 0 degree is chip mark on upper left corner
	PIOS_HMC5883_BOTTOM_0DEG,
	PIOS_HMC5883_BOTTOM_90DEG,
	PIOS_HMC5883_BOTTOM_180DEG,
	PIOS_HMC5883_BOTTOM_270DEG
};

struct pios_hmc5883_cfg {
	const struct pios_exti_cfg * exti_cfg; /* Pointer to the EXTI configuration */
	uint8_t M_ODR;		/* OUTPUT DATA RATE --> here below the relative define (See datasheet page 11 for more details) */
	uint8_t Meas_Conf;	/* Measurement Configuration,: Normal, positive bias, or negative bias --> here below the relative define */
	uint8_t Gain;		/* Gain Configuration, select the full scale --> here below the relative define (See datasheet page 11 for more details) */
	uint8_t Mode;
	enum pios_hmc5883_orientation Default_Orientation;
};

struct pios_hmc5883_data {
	int16_t mag_x;
	int16_t mag_y;
	int16_t mag_z;
};

/* Public Functions */
extern int32_t PIOS_HMC5883_Init(uint32_t i2c_id, const struct pios_hmc5883_cfg * cfg);
extern int32_t PIOS_HMC5883_Test(void);
extern int32_t PIOS_HMC5883_SetOrientation(enum pios_hmc5883_orientation orientation);
extern bool PIOS_HMC5883_IRQHandler();
#endif /* PIOS_HMC5883_H */

/** 
  * @}
  * @}
  */
