/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_SERVO Servo Functions
 * @brief PIOS interface to read and write from servo PWM ports
 * @{
 *
 * @file       pios_servo_priv.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Servo private structures.
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

#ifndef PIOS_HCSR04_PRIV_H
#define PIOS_HCSR04_PRIV_H

#include <pios_stm32.h>

#include <pios_tim_priv.h>

struct pios_hcsr04_cfg {
    TIM_ICInitTypeDef tim_ic_init;
    const struct pios_tim_channel *channels;
    uint8_t num_channels;
    struct stm32_gpio trigger;
};

extern int32_t PIOS_HCSR04_Init(uintptr_t *hcsr04_id, const struct pios_hcsr04_cfg *cfg);

#endif /* PIOS_HCSR04_PRIV_H */

/**
 * @}
 * @}
 */
