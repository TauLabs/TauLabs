/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MPU60X0 MPU60X0 Functions
 * @brief Deals with the hardware interface to the 3-axis gyro
 * @{
 *
 * @file       PIOS_MPU60X0.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      MPU60X0 3-axis gyor function headers
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

#ifndef PIOS_MPU60X0_H
#define PIOS_MPU60X0_H

#include "pios.h"

/* MPU60X0 Addresses */
#define PIOS_MPU60X0_SMPLRT_DIV_REG       0X19
#define PIOS_MPU60X0_DLPF_CFG_REG         0X1A
#define PIOS_MPU60X0_GYRO_CFG_REG         0X1B
#define PIOS_MPU60X0_ACCEL_CFG_REG        0X1C
#define PIOS_MPU60X0_FIFO_EN_REG          0x23
#define PIOS_MPU60X0_SLV0_ADDR_REG        0x25
#define PIOS_MPU60X0_SLV0_REG_REG         0x26
#define PIOS_MPU60X0_SLV0_CTRL_REG        0x27
#define PIOS_MPU60X0_SLV4_ADDR_REG        0x31
#define PIOS_MPU60X0_SLV4_REG_REG         0x32
#define PIOS_MPU60X0_SLV4_DO_REG          0x33
#define PIOS_MPU60X0_SLV4_CTRL_REG        0x34
#define PIOS_MPU60X0_SLV4_DI_REG          0x35
#define PIOS_MPU60X0_I2C_MST_STATUS_REG   0x36
#define PIOS_MPU60X0_INT_CFG_REG          0x37
#define PIOS_MPU60X0_INT_EN_REG           0x38
#define PIOS_MPU60X0_INT_STATUS_REG       0x3A
#define PIOS_MPU60X0_ACCEL_X_OUT_MSB      0x3B
#define PIOS_MPU60X0_ACCEL_X_OUT_LSB      0x3C
#define PIOS_MPU60X0_ACCEL_Y_OUT_MSB      0x3D
#define PIOS_MPU60X0_ACCEL_Y_OUT_LSB      0x3E
#define PIOS_MPU60X0_ACCEL_Z_OUT_MSB      0x3F
#define PIOS_MPU60X0_ACCEL_Z_OUT_LSB      0x40
#define PIOS_MPU60X0_TEMP_OUT_MSB         0x41
#define PIOS_MPU60X0_TEMP_OUT_LSB         0x42
#define PIOS_MPU60X0_GYRO_X_OUT_MSB       0x43
#define PIOS_MPU60X0_GYRO_X_OUT_LSB       0x44
#define PIOS_MPU60X0_GYRO_Y_OUT_MSB       0x45
#define PIOS_MPU60X0_GYRO_Y_OUT_LSB       0x46
#define PIOS_MPU60X0_GYRO_Z_OUT_MSB       0x47
#define PIOS_MPU60X0_GYRO_Z_OUT_LSB       0x48
#define PIOS_MPU60X0_SIGNAL_PATH_RESET    0x68
#define PIOS_MPU60X0_USER_CTRL_REG        0x6A
#define PIOS_MPU60X0_PWR_MGMT_REG         0x6B
#define PIOS_MPU60X0_FIFO_CNT_MSB         0x72
#define PIOS_MPU60X0_FIFO_CNT_LSB         0x73
#define PIOS_MPU60X0_FIFO_REG             0x74
#define PIOS_MPU60X0_WHOAMI               0x75

/* FIFO enable for storing different values */
#define PIOS_MPU60X0_FIFO_TEMP_OUT        0x80
#define PIOS_MPU60X0_FIFO_GYRO_X_OUT      0x40
#define PIOS_MPU60X0_FIFO_GYRO_Y_OUT      0x20
#define PIOS_MPU60X0_FIFO_GYRO_Z_OUT      0x10
#define PIOS_MPU60X0_ACCEL_OUT            0x08

/* Interrupt Configuration */
#define PIOS_MPU60X0_INT_ACTL             0x80
#define PIOS_MPU60X0_INT_OPEN             0x40
#define PIOS_MPU60X0_INT_LATCH_EN         0x20
#define PIOS_MPU60X0_INT_CLR_ANYRD        0x10

#define PIOS_MPU60X0_INTEN_OVERFLOW       0x10
#define PIOS_MPU60X0_INTEN_DATA_RDY       0x01

/* Interrupt status */
#define PIOS_MPU60X0_INT_STATUS_OVERFLOW  0x10
#define PIOS_MPU60X0_INT_STATUS_IMU_RDY   0X04
#define PIOS_MPU60X0_INT_STATUS_DATA_RDY  0X01

/* User control functionality */
#define PIOS_MPU60X0_USERCTL_FIFO_EN      0X40
#define PIOS_MPU60X0_USERCTL_I2C_MST_EN   0X20
#define PIOS_MPU60X0_USERCTL_DIS_I2C      0X10
#define PIOS_MPU60X0_USERCTL_FIFO_RST     0X02
#define PIOS_MPU60X0_USERCTL_GYRO_RST     0X01

/* Power management and clock selection */
#define PIOS_MPU60X0_PWRMGMT_IMU_RST      0X80
#define PIOS_MPU60X0_PWRMGMT_INTERN_CLK   0X00
#define PIOS_MPU60X0_PWRMGMT_PLL_X_CLK    0X01
#define PIOS_MPU60X0_PWRMGMT_PLL_Y_CLK    0X02
#define PIOS_MPU60X0_PWRMGMT_PLL_Z_CLK    0X03
#define PIOS_MPU60X0_PWRMGMT_STOP_CLK     0X07

/* I2C master status register bits */
#define PIOS_MPU60X0_I2C_MST_SLV4_DONE    0x40
#define PIOS_MPU60X0_I2C_MST_LOST_ARB     0x20
#define PIOS_MPU60X0_I2C_MST_SLV4_NACK    0x10
#define PIOS_MPU60X0_I2C_MST_SLV0_NACK    0x01

/* I2C SLV register bits */
#define PIOS_MPU60X0_I2CSLV_EN            0x80
#define PIOS_MPU60X0_I2CSLV_BYTE_SW       0x40
#define PIOS_MPU60X0_I2CSLV_REG_DIS       0x20
#define PIOS_MPU60X0_I2CSLV_GRP           0x10

enum pios_mpu60x0_range {
	PIOS_MPU60X0_SCALE_250_DEG  = 0x00,
	PIOS_MPU60X0_SCALE_500_DEG  = 0x08,
	PIOS_MPU60X0_SCALE_1000_DEG = 0x10,
	PIOS_MPU60X0_SCALE_2000_DEG = 0x18
};

enum pios_mpu60x0_filter {
	PIOS_MPU60X0_LOWPASS_256_HZ = 0x00,
	PIOS_MPU60X0_LOWPASS_188_HZ = 0x01,
	PIOS_MPU60X0_LOWPASS_98_HZ  = 0x02,
	PIOS_MPU60X0_LOWPASS_42_HZ  = 0x03,
	PIOS_MPU60X0_LOWPASS_20_HZ  = 0x04,
	PIOS_MPU60X0_LOWPASS_10_HZ  = 0x05,
	PIOS_MPU60X0_LOWPASS_5_HZ   = 0x06
};

enum pios_mpu60x0_accel_range {
	PIOS_MPU60X0_ACCEL_2G = 0x00,
	PIOS_MPU60X0_ACCEL_4G = 0x08,
	PIOS_MPU60X0_ACCEL_8G = 0x10,
	PIOS_MPU60X0_ACCEL_16G = 0x18
};

enum pios_mpu60x0_orientation { // clockwise rotation from board forward
	PIOS_MPU60X0_TOP_0DEG    = 0x00,
	PIOS_MPU60X0_TOP_90DEG   = 0x01,
	PIOS_MPU60X0_TOP_180DEG  = 0x02,
	PIOS_MPU60X0_TOP_270DEG  = 0x03,
	PIOS_MPU60X0_BOTTOM_0DEG  = 0x04,
	PIOS_MPU60X0_BOTTOM_90DEG  = 0x05,
	PIOS_MPU60X0_BOTTOM_180DEG  = 0x06,
	PIOS_MPU60X0_BOTTOM_270DEG  = 0x07,
};

struct pios_mpu60x0_data {
	int16_t gyro_x;
	int16_t gyro_y;
	int16_t gyro_z;
#if defined(PIOS_MPU6000_ACCEL) || defined(PIOS_MPU6050_ACCEL)
	int16_t accel_x;
	int16_t accel_y;
	int16_t accel_z;
#endif /* PIOS_MPU6000_ACCEL || PIOS_MPU6050_ACCEL */
	int16_t temperature;
};

struct pios_mpu60x0_cfg {
	const struct pios_exti_cfg *exti_cfg; /* Pointer to the EXTI configuration */

	uint16_t default_samplerate;	/* Sample to use in Hz (See datasheet page 32 for more details) */
	uint8_t interrupt_cfg;			/* Interrupt configuration (See datasheet page 35 for more details) */
	uint8_t interrupt_en;			/* Interrupt configuration (See datasheet page 35 for more details) */
	uint8_t User_ctl;				/* User control settings (See datasheet page 41 for more details)  */
	uint8_t Pwr_mgmt_clk;			/* Power management and clock selection (See datasheet page 32 for more details) */
	enum pios_mpu60x0_filter default_filter;
	enum pios_mpu60x0_orientation orientation;
};

#endif /* PIOS_MPU60X0_H */

/** 
  * @}
  * @}
  */
