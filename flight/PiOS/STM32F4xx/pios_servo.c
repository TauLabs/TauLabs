/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_SERVO RC Servo Functions
 * @brief Code to do set RC servo output
 * @{
 *
 * @file       pios_servo.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014-2015
 * @brief      RC Servo routines (STM32 dependent)
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

/* Project Includes */
#include "pios.h"
#include "pios_servo_priv.h"
#include "pios_tim_priv.h"
#include "misc_math.h"

/* Private Function Prototypes */
static const struct pios_servo_cfg * servo_cfg;

//! The counter rate for the channel, used to calculate compare values.
static enum pwm_mode *output_channel_resolution;  // The clock rate for that timer
#if defined(PIOS_INCLUDE_HPWM)
enum SYNC_PWM {SYNC_PWM_FALSE, SYNC_PWM_TRUE};
static enum SYNC_PWM *output_channel_mode;
#endif

/* Private constant definitions */
#define PWM_MODE_1US_RATE   1000000
#define PWM_MODE_80NS_RATE  12000000

/**
* Initialise Servos
*/
int32_t PIOS_Servo_Init(const struct pios_servo_cfg * cfg)
{
	uintptr_t tim_id;
	if (PIOS_TIM_InitChannels(&tim_id, cfg->channels, cfg->num_channels, NULL, 0)) {
		return -1;
	}

	/* Store away the requested configuration */
	servo_cfg = cfg;

	/* Configure the channels to be in output compare mode */
	for (uint8_t i = 0; i < cfg->num_channels; i++) {
		const struct pios_tim_channel * chan = &cfg->channels[i];

		/* Set up for output compare function */
		switch(chan->timer_chan) {
			case TIM_Channel_1:
				TIM_OC1Init(chan->timer, (TIM_OCInitTypeDef*)&cfg->tim_oc_init);
				TIM_OC1PreloadConfig(chan->timer, TIM_OCPreload_Enable);
				break;
			case TIM_Channel_2:
				TIM_OC2Init(chan->timer, (TIM_OCInitTypeDef*)&cfg->tim_oc_init);
				TIM_OC2PreloadConfig(chan->timer, TIM_OCPreload_Enable);
				break;
			case TIM_Channel_3:
				TIM_OC3Init(chan->timer, (TIM_OCInitTypeDef*)&cfg->tim_oc_init);
				TIM_OC3PreloadConfig(chan->timer, TIM_OCPreload_Enable);
				break;
			case TIM_Channel_4:
				TIM_OC4Init(chan->timer, (TIM_OCInitTypeDef*)&cfg->tim_oc_init);
				TIM_OC4PreloadConfig(chan->timer, TIM_OCPreload_Enable);
				break;
		}

		TIM_ARRPreloadConfig(chan->timer, ENABLE);
		TIM_CtrlPWMOutputs(chan->timer, ENABLE);
		TIM_Cmd(chan->timer, ENABLE);
	}

	output_channel_resolution = PIOS_malloc(servo_cfg->num_channels * sizeof(typeof(output_channel_resolution)));
	if (output_channel_resolution == NULL) {
		return -1;
	}
	memset(output_channel_resolution, 0, servo_cfg->num_channels * sizeof(typeof(output_channel_resolution)));
#if defined(PIOS_INCLUDE_HPWM)
	/* Allocate memory for frequency table */
	output_channel_mode = PIOS_malloc(servo_cfg->num_channels * sizeof(typeof(output_channel_mode)));
	if (output_channel_mode == NULL) {
		return -1;
	}
	memset(output_channel_mode, 0, servo_cfg->num_channels * sizeof(typeof(output_channel_mode)));
#endif

	return 0;
}

/**
 * @brief PIOS_Servo_SetMode Sets the PWM output frequency and resolution.
 * A speed of 0 indicates Synchronous updates (e.g. OneShot), in which case
 * the period is set to the maximal value. Otherwise the prescalar will be
 * determined by the PWM mode to set the resolution, and the period will be
 * calculated based on the speed. The information required to convert from us
 * to compare value is cached for each channel (not timer) to facilitate 
 * PIOS_Servo_Set and PIOS_SERVO_HPWM_Set.
 * @param speeds array of rates in Hz
 * @param pwm_mode mode of the timer (1us to 1/12us)
 * @param banks maximum number of banks
 */
void PIOS_Servo_SetMode(const uint16_t * speeds, const enum pwm_mode *pwm_mode, uint8_t banks)
{
	if (!servo_cfg) {
		return;
	}

	TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure = servo_cfg->tim_base_init;
	TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1;
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;

	uint8_t set = 0;

	for (uint8_t i = 0; (i < servo_cfg->num_channels) && (set < banks); i++) {
		bool new = true;
		const struct pios_tim_channel * chan = &servo_cfg->channels[i];

		/* See if any previous channels use that same timer */
		for (uint8_t j = 0; (j < i) && new; j++) {
			new = new && (chan->timer != servo_cfg->channels[j].timer);
		}

		if (new) {

			uint32_t clk_rate = 0;

			// Based on PWM mode determine the desired output period (which sets the
			// channel resolution)
			if (pwm_mode[set] == PWM_MODE_1US) {
				clk_rate = PWM_MODE_1US_RATE; // Default output timer frequency in hertz
			} else if (pwm_mode[set] == PWM_MODE_80NS) {
				clk_rate = PWM_MODE_80NS_RATE; // Default output timer frequency in hertz
			}

			if (speeds[set] == 0) {
				// Use a maximally long period because we don't want pulses actually repeating
				// without new data arriving.
				TIM_TimeBaseStructure.TIM_Period = 0xFFFFFFFF;

			} else {
				// Note: this can be extended with a new PWM mode that is lower resolution
				// for very long periods
				TIM_TimeBaseStructure.TIM_Period = (clk_rate / speeds[set]) - 1;
			}

			/* Choose the correct prescaler value for the APB the timer is attached */
			if (chan->timer==TIM6 || chan->timer==TIM7) {
				// These timers cannot be used here.
				return;
			} else if (chan->timer==TIM1 || chan->timer==TIM8 || chan->timer==TIM9 || chan->timer==TIM10 || chan->timer==TIM11 ) {
				TIM_TimeBaseStructure.TIM_Prescaler = (PIOS_PERIPHERAL_APB2_CLOCK / clk_rate) - 1;
			} else {
				TIM_TimeBaseStructure.TIM_Prescaler = (PIOS_PERIPHERAL_APB1_CLOCK / clk_rate) - 1;
			}

			// Configure this timer appropriately.
			TIM_TimeBaseInit(chan->timer, &TIM_TimeBaseStructure);	

			/* Configure frequency scaler for all channels that use the same timer */
			for (uint8_t j=0; (j < servo_cfg->num_channels); j++) {
				if (chan->timer == servo_cfg->channels[j].timer) {
#if defined(PIOS_INCLUDE_HPWM)
					/* save the frequency for these channels */
					output_channel_mode[j] = (speeds[set] == 0) ? SYNC_PWM_TRUE : SYNC_PWM_FALSE;
					output_channel_resolution[j] = pwm_mode[set];
#endif
				}
			}

			set++;
		}
	}
}

/**
* Set servo position for HPWM
* \param[in] Servo Servo number (0-num_channels)
* \param[in] Position Servo position in microseconds
*/
#if defined(PIOS_INCLUDE_HPWM)
void PIOS_Servo_Set(uint8_t servo, float position)
{
	/* Make sure servo exists */
	if (!servo_cfg || servo >= servo_cfg->num_channels) {
		return;
	}

	const struct pios_tim_channel * chan = &servo_cfg->channels[servo];

	/* recalculate the position value based on timer clock rate */
	/* position is in us. Note: if the set of channel resolutions */
	/* stop all being multiples of 1MHz we might need to refactor */
	/* the math a bit to preserve precision */
	uint32_t us_to_count = 0;
	switch(output_channel_resolution[servo]) {
	case PWM_MODE_1US:
		us_to_count = PWM_MODE_1US_RATE / 1000000;
		break;
	case PWM_MODE_80NS:
		us_to_count = PWM_MODE_80NS_RATE / 1000000;
		break;
	}
	position = position * us_to_count;

	/* stop the timer in OneShot (Synchronous) mode */
	if (output_channel_mode[servo] == SYNC_PWM_TRUE) {
		TIM_Cmd(chan->timer, DISABLE);
	}

	/* Update the position */
	switch(chan->timer_chan) {
		case TIM_Channel_1:
			TIM_SetCompare1(chan->timer, position);
			break;
		case TIM_Channel_2:
			TIM_SetCompare2(chan->timer, position);
			break;
		case TIM_Channel_3:
			TIM_SetCompare3(chan->timer, position);
			break;
		case TIM_Channel_4:
			TIM_SetCompare4(chan->timer, position);
			break;
	}
}
#else
/**
* Set servo position
* \param[in] Servo Servo number (0-num_channels)
* \param[in] Position Servo position in microseconds
*/
void PIOS_Servo_Set(uint8_t servo, uint16_t position)
{
	/* Make sure servo exists */
	if (!servo_cfg || servo >= servo_cfg->num_channels) {
		return;
	}

	const struct pios_tim_channel * chan = &servo_cfg->channels[servo];

	/* recalculate the position value based on timer clock rate */
	/* position is in us */
	/* clk_rate is in count per second */

	/* Update the position */
	switch(chan->timer_chan) {
		case TIM_Channel_1:
			TIM_SetCompare1(chan->timer, position);
			break;
		case TIM_Channel_2:
			TIM_SetCompare2(chan->timer, position);
			break;
		case TIM_Channel_3:
			TIM_SetCompare3(chan->timer, position);
			break;
		case TIM_Channel_4:
			TIM_SetCompare4(chan->timer, position);
			break;
	}
}
#endif /* PIOS_INCLUDE_HPWM */

#if defined(PIOS_INCLUDE_HPWM)
/**
* Update the timer for HPWM/OneShot
*/
void PIOS_Servo_Update()
{
	if (!servo_cfg) {
		return;
	}

	for (uint8_t i = 0; i < servo_cfg->num_channels; i++) {
		const struct pios_tim_channel * chan = &servo_cfg->channels[i];

		/* Check for channels that are using synchronous output */
		/* Look for a disabled timer using synchronous output */
		if (!(chan->timer->CR1 & TIM_CR1_CEN) &&
		     (output_channel_mode[i] == SYNC_PWM_TRUE)) {
			/* enable it again and reinitialize it */
			TIM_Cmd(chan->timer, ENABLE);
			TIM_GenerateEvent(chan->timer, TIM_EventSource_Update);
		}
	}
}

#endif /* PIOS_INCLUDE_HPWM */
