/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MPU9250 MPU9250 Functions
 * @brief Deals with the hardware interface to the 3-axis gyro
 * @{
 *
 * @file       pios_mpu9250.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      MPU9250 9-DOF chip function headers
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

#ifndef PIOS_MPU9250_H
#define PIOS_MPU9250_H

#include "pios.h"
#include "pios_mpu60x0.h"

/* MPU9250 I2C Addresses */
#define PIOS_MPU9250_I2C_ADD_A0_LOW       0x68
#define PIOS_MPU9250_I2C_ADD_A0_HIGH      0x69

enum pios_mpu9250_gyro_filter {
	PIOS_MPU9250_GYRO_LOWPASS_250_HZ  = 0x00, // do not use, sample rat div. cannot be used
	PIOS_MPU9250_GYRO_LOWPASS_184_HZ  = 0x01,
	PIOS_MPU9250_GYRO_LOWPASS_92_HZ   = 0x02,
	PIOS_MPU9250_GYRO_LOWPASS_41_HZ   = 0x03,
	PIOS_MPU9250_GYRO_LOWPASS_20_HZ   = 0x04,
	PIOS_MPU9250_GYRO_LOWPASS_10_HZ   = 0x05,
	PIOS_MPU9250_GYRO_LOWPASS_5_HZ    = 0x06,
	PIOS_MPU9250_GYRO_LOWPASS_3600_HZ = 0x07 // do not use, sample rat div. cannot be used
};

enum pios_mpu9250_accel_filter {
	PIOS_MPU9250_ACCEL_LOWPASS_460_HZ  = 0x00, // do not use, sample rat div. cannot be used
	PIOS_MPU9250_ACCEL_LOWPASS_184_HZ  = 0x01,
	PIOS_MPU9250_ACCEL_LOWPASS_92_HZ   = 0x02,
	PIOS_MPU9250_ACCEL_LOWPASS_41_HZ   = 0x03,
	PIOS_MPU9250_ACCEL_LOWPASS_20_HZ   = 0x04,
	PIOS_MPU9250_ACCEL_LOWPASS_10_HZ   = 0x05,
	PIOS_MPU9250_ACCEL_LOWPASS_5_HZ    = 0x06,
	PIOS_MPU9250_ACCEL_LOWPASS_3600_HZ = 0x07 // do not use, sample rat div. cannot be used
};

struct pios_mpu9250_cfg {
	const struct pios_exti_cfg *exti_cfg; /* Pointer to the EXTI configuration */

	uint16_t default_samplerate;	/* Sample to use in Hz (See datasheet page 32 for more details) */
	uint8_t interrupt_cfg;			/* Interrupt configuration (See datasheet page 35 for more details) */
	uint8_t interrupt_en;			/* Interrupt configuration (See datasheet page 35 for more details) */
	uint8_t User_ctl;				/* User control settings (See datasheet page 41 for more details)  */
	uint8_t Pwr_mgmt_clk;			/* Power management and clock selection (See datasheet page 32 for more details) */
	enum pios_mpu9250_gyro_filter gyro_filter;
	enum pios_mpu9250_accel_filter accel_filter;
	enum pios_mpu60x0_orientation orientation;
};

/* Public Functions */
extern int32_t PIOS_MPU9250_Init(uint32_t i2c_id, uint8_t i2c_addr, bool use_mag, const struct pios_mpu9250_cfg * new_cfg);
extern uint8_t PIOS_MPU9250_Test();
extern int32_t PIOS_MPU9250_Probe(uint32_t i2c_id, uint8_t i2c_addr);
extern int32_t PIOS_MPU9250_SetGyroRange(enum pios_mpu60x0_range);
extern int32_t PIOS_MPU9250_SetAccelRange(enum pios_mpu60x0_accel_range);
extern int32_t PIOS_MPU9250_SetSampleRate(uint16_t samplerate_hz);
extern void PIOS_MPU9250_SetLPF(enum pios_mpu60x0_filter filter);
extern bool PIOS_MPU9250_IRQHandler(void);

#endif /* PIOS_MPU9250_H */

/** 
  * @}
  * @}
  */
