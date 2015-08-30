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

enum pios_mpu9250_gyro_filter {
	PIOS_MPU9250_GYRO_LOWPASS_250_HZ = 0x00,
	PIOS_MPU9250_GYRO_LOWPASS_184_HZ = 0x01,
	PIOS_MPU9250_GYRO_LOWPASS_92_HZ  = 0x02,
	PIOS_MPU9250_GYRO_LOWPASS_41_HZ  = 0x03,
	PIOS_MPU9250_GYRO_LOWPASS_20_HZ  = 0x04,
	PIOS_MPU9250_GYRO_LOWPASS_10_HZ  = 0x05,
	PIOS_MPU9250_GYRO_LOWPASS_5_HZ   = 0x06
};

enum pios_mpu9250_accel_filter {
	PIOS_MPU9250_ACCEL_LOWPASS_460_HZ = 0x00,
	PIOS_MPU9250_ACCEL_LOWPASS_184_HZ = 0x01,
	PIOS_MPU9250_ACCEL_LOWPASS_92_HZ  = 0x02,
	PIOS_MPU9250_ACCEL_LOWPASS_41_HZ  = 0x03,
	PIOS_MPU9250_ACCEL_LOWPASS_20_HZ  = 0x04,
	PIOS_MPU9250_ACCEL_LOWPASS_10_HZ  = 0x05,
	PIOS_MPU9250_ACCEL_LOWPASS_5_HZ   = 0x06
};

enum pios_mpu9250_orientation { // clockwise rotation from board forward
	PIOS_MPU9250_TOP_0DEG       = 0x00,
	PIOS_MPU9250_TOP_90DEG      = 0x01,
	PIOS_MPU9250_TOP_180DEG     = 0x02,
	PIOS_MPU9250_TOP_270DEG     = 0x03,
	PIOS_MPU9250_BOTTOM_0DEG    = 0x04,
	PIOS_MPU9250_BOTTOM_90DEG   = 0x05,
	PIOS_MPU9250_BOTTOM_180DEG  = 0x06,
	PIOS_MPU9250_BOTTOM_270DEG  = 0x07
};


struct pios_mpu9250_cfg {
	const struct pios_exti_cfg *exti_cfg; /* Pointer to the EXTI configuration */

	uint16_t default_samplerate;	/* Sample to use in Hz (See RM datasheet page 12 for more details) */
	uint8_t interrupt_cfg;			/* Interrupt configuration (See RM datasheet page 20 for more details) */
	bool use_magnetometer;			/* Use magnetometer or not - for example when external mag. is used */
	enum pios_mpu9250_gyro_filter default_gyro_filter;
	enum pios_mpu9250_accel_filter default_accel_filter;
	enum pios_mpu9250_orientation orientation;
};

/* Public Functions */
extern int32_t PIOS_MPU9250_SPI_Init(uint32_t spi_id, uint32_t slave_num, const struct pios_mpu9250_cfg *new_cfg);
extern int32_t PIOS_MPU9250_Test();
extern int32_t PIOS_MPU9250_SetGyroRange(enum pios_mpu60x0_range range);
extern int32_t PIOS_MPU9250_SetAccelRange(enum pios_mpu60x0_accel_range);
extern int32_t PIOS_MPU9250_SetSampleRate(uint16_t samplerate_hz);
extern void PIOS_MPU9250_SetGyroLPF(enum pios_mpu9250_gyro_filter filter);
extern void PIOS_MPU9250_SetAccelLPF(enum pios_mpu9250_accel_filter filter);
extern bool PIOS_MPU9250_IRQHandler(void);
extern void PIOS_MPU9250_SetGyroDownSamling(const uint8_t *gyro_downsampling);

#endif /* PIOS_MPU69250_H */

/** 
  * @}
  * @}
  */
