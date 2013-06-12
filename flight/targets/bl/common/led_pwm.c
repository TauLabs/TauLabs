/**
 ******************************************************************************
 * @file       led_pwm.c
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

#include "pios.h"		/* PIOS_LED_* -- FIXME: include is too coarse */
#include "led_pwm.h"		/* API definition */

static uint32_t led_pwm_on_p(uint32_t pwm_period, uint32_t pwm_sweep_steps, uint32_t uptime) {
	/* 0 - pwm_sweep_steps */
	uint32_t curr_step = (uptime / pwm_period) % pwm_sweep_steps;

	/* fraction of pwm_period */
	uint32_t pwm_duty = pwm_period * curr_step / pwm_sweep_steps;

	/* ticks once per full sweep */
	uint32_t curr_sweep = (uptime / (pwm_period * pwm_sweep_steps));

	/* reverse direction in odd sweeps */
	if (curr_sweep & 1) {
		pwm_duty = pwm_period - pwm_duty;
	}

	return ((uptime % pwm_period) > pwm_duty) ? 1 : 0;
}

void led_pwm_config(struct led_pwm_state *leds,
			uint32_t pwm_1_period_us,
			uint32_t pwm_1_sweep_steps,
			uint32_t pwm_2_period_us,
			uint32_t pwm_2_sweep_steps)
{
	/* PWM #1 */
	if (pwm_1_period_us > 0) {
		leds->pwm_1_enabled     = true;
		leds->pwm_1_period_us   = pwm_1_period_us;
		leds->pwm_1_sweep_steps = pwm_1_sweep_steps;
	} else {
		leds->pwm_1_enabled     = false;
	}


	/* PWM #2 */
	if (pwm_2_period_us > 0) {
		leds->pwm_2_enabled     = true;
		leds->pwm_2_period_us   = pwm_2_period_us;
		leds->pwm_2_sweep_steps = pwm_2_sweep_steps;
	} else {
		leds->pwm_2_enabled     = false;
	}
}

void led_pwm_add_ticks(struct led_pwm_state *leds, uint32_t elapsed_us)
{
	leds->uptime_us += elapsed_us;
}

bool led_pwm_update_leds(const struct led_pwm_state *leds)
{
	/*
	 * Compute states of emulated PWM timers
	 */
	bool pwm_1_led_state = true; /* forces LED on when pwm1 is disabled */
	if (leds->pwm_1_enabled) {
		pwm_1_led_state = led_pwm_on_p(leds->pwm_1_period_us,
					leds->pwm_1_sweep_steps,
					leds->uptime_us);
	}

	bool pwm_2_led_state = false; /* forces LED off when pwm2 is disabled */
	if (leds->pwm_2_enabled) {
		pwm_2_led_state = led_pwm_on_p(leds->pwm_2_period_us,
					leds->pwm_2_sweep_steps,
					leds->uptime_us);
	}

	/*
	 * Toggle the LEDs based on the 2 emulated PWM timers
	 */

	if (pwm_1_led_state) {
		PIOS_LED_On(PIOS_LED_HEARTBEAT);
	} else {
		PIOS_LED_Off(PIOS_LED_HEARTBEAT);
	}

	if (pwm_2_led_state) {
		PIOS_LED_On(PIOS_LED_HEARTBEAT);
	} else {
		PIOS_LED_Off(PIOS_LED_HEARTBEAT);
	}

	return true;
}

/**
 * @}
 * @}
 */
