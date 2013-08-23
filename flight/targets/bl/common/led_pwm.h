/**
 ******************************************************************************
 * @file       led_pwm.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup Bootloader
 * @{
 * @addtogroup Bootloader
 * @{
 * @brief LED PWM emulation for the Tau Labs unified bootloader
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

#ifndef LED_PWM_H_
#define LED_PWM_H_

#include <stdint.h>		/* uint*_t */
#include <stdbool.h>		/* bool */

struct led_pwm_state {
	uint32_t uptime_us;

	bool pwm_1_enabled;
	uint32_t pwm_1_period_us;
	uint32_t pwm_1_sweep_steps;

	bool pwm_2_enabled;
	uint32_t pwm_2_period_us;
	uint32_t pwm_2_sweep_steps;
};

extern void led_pwm_config(struct led_pwm_state * leds, uint32_t pwm_1_period_us, uint32_t pwm_1_sweep_steps, uint32_t pwm_2_period_us, uint32_t pwm_2_sweep_steps);

extern void led_pwm_add_ticks(struct led_pwm_state *leds, uint32_t elapsed_us);

extern bool led_pwm_update_leds(const struct led_pwm_state *leds);

#endif	/* LED_PWM_H_ */

/**
 * @}
 * @}
 */
