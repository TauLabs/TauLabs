/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_SERVO RC Servo Functions
 * @{
 *
 * @file       pios_servo.h  
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 * @brief      RC Servo functions header.
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

#ifndef PIOS_SERVO_H
#define PIOS_SERVO_H

enum pwm_mode {PWM_MODE_1US, PWM_MODE_12US};

/* Public Functions */
extern void PIOS_Servo_SetMode(const uint16_t * update_rates, enum pwm_mode *pwm_mdoe, uint8_t banks);
extern void PIOS_Servo_Set(uint8_t Servo, uint16_t Position);
extern void PIOS_Servo_HPWM_Set(uint8_t servo, float position);
extern void PIOS_Servo_HPWM_Update();

#endif /* PIOS_SERVO_H */

/**
  * @}
  * @}
  */
