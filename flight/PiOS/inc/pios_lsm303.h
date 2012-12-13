/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_LSM303 LSM303 Functions
 * @brief Deals with the hardware interface to the LSM303 3-axis accelerometer and 3-axis magnetometer
 * @{
 *
 * @file       pios_lsm303.h
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @brief      LSM303 3-axis accelerometer and 3-axis magnetometer function headers
 * @see        The GNU Public License (GPL) Version 3
 *
 ******************************************************************************
 */
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

#ifndef PIOS_PIOS_LSM303_H
#define PIOS_PIOS_LSM303_H

#include "pios.h"


// register addresses
#define PIOS_LSM303_CTRL_REG1_A			0x20
#define PIOS_LSM303_CTRL_REG2_A			0x21
#define PIOS_LSM303_CTRL_REG3_A			0x22
#define PIOS_LSM303_CTRL_REG4_A			0x23
#define PIOS_LSM303_CTRL_REG5_A			0x24
#define PIOS_LSM303_CTRL_REG6_A			0x25 // DLHC only
#define PIOS_LSM303_HP_FILTER_RESET_A	0x25 // DLH, DLM only
#define PIOS_LSM303_REFERENCE_A			0x26
#define PIOS_LSM303_STATUS_REG_A		0x27

#define PIOS_LSM303_OUT_X_L_A			0x28
#define PIOS_LSM303_OUT_X_H_A			0x29
#define PIOS_LSM303_OUT_Y_L_A			0x2A
#define PIOS_LSM303_OUT_Y_H_A			0x2B
#define PIOS_LSM303_OUT_Z_L_A			0x2C
#define PIOS_LSM303_OUT_Z_H_A			0x2D

#define PIOS_LSM303_FIFO_CTRL_REG_A		0x2E // DLHC only
#define PIOS_LSM303_FIFO_SRC_REG_A		0x2F // DLHC only

#define PIOS_LSM303_INT1_CFG_A			0x30
#define PIOS_LSM303_INT1_SRC_A			0x31
#define PIOS_LSM303_INT1_THS_A			0x32
#define PIOS_LSM303_INT1_DURATION_A		0x33
#define PIOS_LSM303_INT2_CFG_A			0x34
#define PIOS_LSM303_INT2_SRC_A			0x35
#define PIOS_LSM303_INT2_THS_A			0x36
#define PIOS_LSM303_INT2_DURATION_A		0x37

#define PIOS_LSM303_CLICK_CFG_A			0x38 // DLHC only
#define PIOS_LSM303_CLICK_SRC_A			0x39 // DLHC only
#define PIOS_LSM303_CLICK_THS_A			0x3A // DLHC only
#define PIOS_LSM303_TIME_LIMIT_A		0x3B // DLHC only
#define PIOS_LSM303_TIME_LATENCY_A		0x3C // DLHC only
#define PIOS_LSM303_TIME_WINDOW_A		0x3D // DLHC only

#define PIOS_LSM303_CRA_REG_M			0x00
#define PIOS_LSM303_CRB_REG_M			0x01
#define PIOS_LSM303_MR_REG_M			0x02

#define PIOS_LSM303_OUT_X_H_M			0x03
#define PIOS_LSM303_OUT_X_L_M			0x04
#define PIOS_LSM303_OUT_Y_H_M			0x05	// Attention: the addresses of the Y and Z magnetometer output registers
#define PIOS_LSM303_OUT_Y_L_M			0x06	// are reversed on the DLM and DLHC relative to the DLH.
#define PIOS_LSM303_OUT_Z_H_M			0x07
#define PIOS_LSM303_OUT_Z_L_M			0x08
#define PIOS_LSM303_SR_REG_M			0x09
#define PIOS_LSM303_IRA_REG_M			0x0A
#define PIOS_LSM303_IRB_REG_M			0x0B
#define PIOS_LSM303_IRC_REG_M			0x0C

#define PIOS_LSM303_WHO_AM_I_M			0x0F // DLM only

#define PIOS_LSM303_TEMP_OUT_H_M		0x31 // DLHC only
#define PIOS_LSM303_TEMP_OUT_L_M		0x32 // DLHC only

#define PIOS_LSM303DLH_OUT_Y_H_M		0x05
#define PIOS_LSM303DLH_OUT_Y_L_M		0x06
#define PIOS_LSM303DLH_OUT_Z_H_M		0x07
#define PIOS_LSM303DLH_OUT_Z_L_M		0x08

#define PIOS_LSM303DLM_OUT_Z_H_M		0x05
#define PIOS_LSM303DLM_OUT_Z_L_M		0x06
#define PIOS_LSM303DLM_OUT_Y_H_M		0x07
#define PIOS_LSM303DLM_OUT_Y_L_M		0x08

#define PIOS_LSM303DLHC_OUT_Z_H_M		0x05
#define PIOS_LSM303DLHC_OUT_Z_L_M		0x06
#define PIOS_LSM303DLHC_OUT_Y_H_M		0x07
#define PIOS_LSM303DLHC_OUT_Y_L_M		0x08

/* Ctrl1 flags */
#define PIOS_LSM303_CTRL1_1344HZ		0x90
#define PIOS_LSM303_CTRL1_400HZ			0x70
#define PIOS_LSM303_CTRL1_PD			0x08
#define PIOS_LSM303_CTRL1_ZEN			0x04
#define PIOS_LSM303_CTRL1_YEN			0x02
#define PIOS_LSM303_CTRL1_XEN			0x01

/* Ctrl3 flags */
#define PIOS_LSM303_CTRL3_I1_CLICK		0x80
#define PIOS_LSM303_CTRL3_I1_AOI1		0x40
#define PIOS_LSM303_CTRL3_I1_AOI2		0x20
#define PIOS_LSM303_CTRL3_I1_DRDY1		0x10
#define PIOS_LSM303_CTRL3_I1_DRDY2		0x08
#define PIOS_LSM303_CTRL3_I1_WTM		0x04
#define PIOS_LSM303_CTRL3_I1_OVERRUN	0x02

/* Ctrl4 flags */
#define PIOS_LSM303_CTRL4_BDU			0x80
#define PIOS_LSM303_CTRL4_BLE			0x40
#define PIOS_LSM303_CTRL4_HR			0x04
#define PIOS_LSM303_CTRL4_SIM			0x02

/* Ctrl5 flags */
#define PIOS_LSM303_CTRL5_BOOT			0x80
#define PIOS_LSM303_CTRL5_FIFO_EN		0x40
#define PIOS_LSM303_CTRL5_LIR_INT1		0x08
#define PIOS_LSM303_CTRL5_D4D_INT1		0x04
#define PIOS_LSM303_CTRL5_LIR_INT2		0x02
#define PIOS_LSM303_CTRL5_D4D_INT2		0x01

/* Ctrl6 flags */
#define PIOS_LSM303_CTRL6_I2_CLICK_EN	0x80
#define PIOS_LSM303_CTRL6_I2_INT1		0x40
#define PIOS_LSM303_CTRL6_I2_INT2		0x20
#define PIOS_LSM303_CTRL6_BOOT_I1		0x10
#define PIOS_LSM303_CTRL6_P2_ACT		0x08
#define PIOS_LSM303_CTRL6_H_LACTIVE		0x02

/* Fifo Ctrl flags */
#define PIOS_LSM303_FIFO_MODE_BYPASS	0x00
#define PIOS_LSM303_FIFO_MODE_FIFO		0x40
#define PIOS_LSM303_FIFO_MODE_STREAM	0x80
#define PIOS_LSM303_FIFO_MODE_TRIGGER	0xc0




enum pios_lsm303_accel_range {
	PIOS_LSM303_ACCEL_2_G = 0x00,
	PIOS_LSM303_ACCEL_4_G = 0x10,
	PIOS_LSM303_ACCEL_8_G = 0x20,
	PIOS_LSM303_ACCEL_16_G = 0x30,
};


// device types
enum pios_lsm303_devicetype
{
	PIOS_LSM303_DEVICE_INVALID = 0,
	PIOS_LSM303DLH_DEVICE,
	PIOS_LSM303DLM_DEVICE,
	PIOS_LSM303DLHC_DEVICE,
	PIOS_LSM303_DEVICE_AUTO,
};

// SA0_A states
enum pios_lsm303_sa0_state
{
	PIOS_LSM303_SA0_A_LOW = 0,
	PIOS_LSM303_SA0_A_HIGH,
	PIOS_LSM303_SA0_A_AUTO,
};

struct pios_lsm303_accel_data {
	int16_t accel_x;
	int16_t accel_y;
	int16_t accel_z;
};

struct pios_lsm303_mag_data {
	int16_t mag_x;
	int16_t mag_y;
	int16_t mag_z;
	int16_t temperature;
};

struct pios_lsm303_cfg {
	const struct pios_exti_cfg * exti_cfg; /* Pointer to the EXTI configuration */

	enum pios_lsm303_devicetype devicetype;
	enum pios_lsm303_sa0_state sa0_state;
	enum pios_lsm303_accel_range accel_range;
};

/* Public Functions */
extern int32_t PIOS_LSM303_Init(uint32_t i2c_id, const struct pios_lsm303_cfg * new_cfg);
extern xQueueHandle PIOS_LSM303_GetQueue_Accel();
extern xQueueHandle PIOS_LSM303_GetQueue_Mag();	//mag not implemented yet
extern int32_t PIOS_LSM303_ReadData_Accel(struct pios_lsm303_accel_data * buffer);
extern int32_t PIOS_LSM303_ReadData_Mag(struct pios_lsm303_mag_data * buffer);	//mag not implemented yet
extern int32_t PIOS_LSM303_ReadID();
extern uint8_t PIOS_LSM303_Test_Accel();
extern uint8_t PIOS_LSM303_Test_Mag();
extern float PIOS_LSM303_GetScale_Accel();
extern float PIOS_LSM303_GetScale_Mag();	//mag not implemented yet
extern bool PIOS_LSM303_IRQHandler(void);

#endif /* PIOS_PIOS_LSM303_H */

/** 
  * @}
  * @}
  */
