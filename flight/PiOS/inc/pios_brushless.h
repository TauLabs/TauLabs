/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_BRUSHLESS Brushless gimbal driver control
 * @{
 *
 * @file       pios_brushless.h
 * @author     Tau Labs, http://github.com/TauLabs Copyright (C) 2013.
 * @brief      Brushless gimbal controller
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

#ifndef PIOS_BRUSHLESS_H
#define PIOS_BRUSHLESS_H

/* Public Functions */

//! Set the speed of a channel in deg / s
extern int32_t PIOS_Brushless_SetSpeed(uint32_t channel, float speed, float dT);

//! Set the phase offset for a channel relative to integrated position
extern int32_t PIOS_Brushless_SetPhaseLag(uint32_t channel, float phase);

//! Set the update rate in hz
extern int32_t PIOS_Brushless_SetUpdateRate(uint32_t rate);

//! Set the amplitude scale in %
extern int32_t PIOS_Brushless_SetScale(uint8_t roll, uint8_t pitch, uint8_t yaw);

//! Max acceleration
extern int32_t PIOS_Brushless_SetMaxAcceleration(float roll, float pitch, float yaw);

#endif /* PIOS_BRUSHLESS_H */

/**
  * @}
  * @}
  */
