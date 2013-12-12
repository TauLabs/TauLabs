/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MPU6050 MPU6050 Functions
 * @brief Deals with the hardware interface to the 3-axis gyro
 * @{
 *
 * @file       PIOS_MPU6050.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      MPU6050 3-axis gyor function headers
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

#ifndef PIOS_MPU6050_H
#define PIOS_MPU6050_H

#include "pios.h"
#include "pios_mpu60x0.h"

/* MPU6050 I2C Addresses */
#define PIOS_MPU6050_I2C_ADD_A0_LOW       0x68
#define PIOS_MPU6050_I2C_ADD_A0_HIGH      0x69

/* Public Functions */
extern int32_t PIOS_MPU6050_Init(uint32_t i2c_id, uint8_t i2c_addr, const struct pios_mpu60x0_cfg *new_cfg);
extern uint8_t PIOS_MPU6050_Test();
extern int32_t PIOS_MPU6050_Probe(uint32_t i2c_id, uint8_t i2c_addr);
extern void PIOS_MPU6050_SetGyroRange(enum pios_mpu60x0_range);

#if defined(PIOS_MPU6050_ACCEL)
extern void PIOS_MPU6050_SetAccelRange(enum pios_mpu60x0_accel_range);
#endif /* PIOS_MPU6000_ACCEL */

extern void PIOS_MPU6050_SetSampleRate(uint16_t samplerate_hz);
extern void PIOS_MPU6050_SetLPF(enum pios_mpu60x0_filter filter);
extern void PIOS_MPU6050_SetPassThrough(bool passThrough);
extern bool PIOS_MPU6050_IRQHandler(void);

#endif /* PIOS_MPU6050_H */

/** 
  * @}
  * @}
  */
