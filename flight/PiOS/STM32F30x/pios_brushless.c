/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_BRUSHLESS Brushless gimbal driver control
 * @{
 *
 * @file       pios_brushless.c
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

/* Project Includes */
#include "pios.h"
#include "pios_brushless_priv.h"
#include "pios_tim_priv.h"

#include "physical_constants.h"
#include "sin_lookup.h"
#include "misc_math.h"

/* Private Function Prototypes */
static int32_t PIOS_Brushless_SetPhase(uint32_t channel, float phase_deg);
static void PIOS_BRUSHLESS_Task(void* parameters);

// Private variables
static const struct pios_brushless_cfg * brushless_cfg;
static xTaskHandle taskHandle;

#define NUM_BGC_CHANNELS 3
#define STACK_SIZE_BYTES 400
#define TASK_PRIORITY  (tskIDLE_PRIORITY+4)

/**
* Initialise Servos
*/
int32_t PIOS_Brushless_Init(const struct pios_brushless_cfg * cfg)
{
	uintptr_t tim_id;
	if (PIOS_TIM_InitChannels(&tim_id, cfg->channels, cfg->num_channels, NULL, 0)) {
		return -1;
	}

	/* Store away the requested configuration */
	brushless_cfg = cfg;

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

	for (uint8_t i = 0 ; i < NUM_BGC_CHANNELS; i++) {
		// Enable the available enable lines
		if (cfg->enables[i].gpio) {
			GPIO_Init(cfg->enables[i].gpio, (GPIO_InitTypeDef *) &cfg->enables[i].init);
			GPIO_SetBits(cfg->enables[i].gpio, cfg->enables[i].init.GPIO_Pin);
		}
	}

	// Start main task
	xTaskCreate(PIOS_BRUSHLESS_Task, (signed char*)"PIOS_BRUSHLESS", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);

	return 0;
}

static float    phases[NUM_BGC_CHANNELS];      /*! current phase for each output channel */
static float    phase_lag[NUM_BGC_CHANNELS];   /*! offset of phase which from module, provides damping */
static float    speeds[NUM_BGC_CHANNELS];      /*! speed for each of the channels */
static float    scales[NUM_BGC_CHANNELS];      /*! fractional scale for each channel. sets power */
static float    accel_limit[NUM_BGC_CHANNELS]; /*! slew rate limit (deg/s^2) */
static int16_t  scale;                         /*! amplitude of sine wave */
static int32_t  center;                        /*! center value of sine wave */

/**
* Set the servo update rate (Max 500Hz)
* \param[in] rate in Hz
*/
int32_t PIOS_Brushless_SetUpdateRate(uint32_t rate)
{
	if (!brushless_cfg) {
		return -1;
	}

	// Set some default reasonable parameters
	TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure = brushless_cfg->tim_base_init;
	center = TIM_TimeBaseStructure.TIM_Period / 2;
	scale = center;

	return 0;
}

/**
* Set servo position
* \param[in] channel The brushless output channel
* \param[in] speed The desired speed (integrated by internal task)
* \
*/
int32_t PIOS_Brushless_SetSpeed(uint32_t channel, float speed, float dT)
{
	if (channel >= NUM_BGC_CHANNELS)
		return -1;

	float diff;
	// Limit the slew rate 
	if (accel_limit[channel])
		diff = bound_sym(speed - speeds[channel], accel_limit[channel] * dT);
	else
		diff = speed - speeds[channel];
	speeds[channel] += diff;

	return 0;
}

/**
 * Set the phase offset for a channel relative to integrated position
 * @param[in] channel The brushless output channel
 * @param[in] phase The phase lag for a channel (for damping)
 */
int32_t PIOS_Brushless_SetPhaseLag(uint32_t channel, float phase)
{
	if (channel >= NUM_BGC_CHANNELS)
		return -1;

	phase_lag[channel] = phase;

	return 0;
}

//! Set the amplitude scale in %
int32_t PIOS_Brushless_SetScale(uint8_t roll, uint8_t pitch, uint8_t yaw)
{
	scales[0] = (float) roll / 100.0f;
	scales[1] = (float) pitch / 100.0f;
	scales[2] = (float) yaw / 100.0f;

	return 0;
}

//! Set the maximum change in velocity per second
int32_t PIOS_Brushless_SetMaxAcceleration(float roll, float pitch, float yaw)
{
	accel_limit[0] = roll;
	accel_limit[1] = pitch;
	accel_limit[2] = yaw;

	return 0;
}

/**
 * PIOS_Brushless_SetPhase set the phase for one of the channel outputs
 * @param[in] channel The channel to set
 * @param[in] phase_deg The phase in degrees to use
 */
static int32_t PIOS_Brushless_SetPhase(uint32_t channel, float phase_deg)
{
	const int32_t PIN_PER_MOTOR  = 3;

	/* Make sure a valid channel */
	if (channel >= NUM_BGC_CHANNELS)
		return -1;

	/* Check enough outputs are registered */
	if (!brushless_cfg || (PIN_PER_MOTOR * (channel + 1)) > brushless_cfg->num_channels) {
		return -2;
	}

	// Get the first output index
	for (int32_t idx = channel * PIN_PER_MOTOR; idx < (channel + 1) * PIN_PER_MOTOR; idx++) {

		// sin lookup expects between 0 and 360
		while (phase_deg > 360)
			phase_deg -= 360;

		int32_t position = scales[channel] * (center + scale * sinf(phase_deg * DEG2RAD));

		/* Update the position */
		const struct pios_tim_channel * chan = &brushless_cfg->channels[idx];
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

		phase_deg += 120;
	}

	return 0;
}

/**
 * Called whenver the PWM output timer wraps around which is quite frequenct (e.g. 30khz) to
 * update the phase on the outputs based on the current rate
 */
static void PIOS_BRUSHLESS_Task(void* parameters)
{
	const uint32_t TICK_DELAY = 1;

	portTickType lastSysTime = xTaskGetTickCount();

	while (1) {

		vTaskDelayUntil(&lastSysTime, TICK_DELAY);

		const float dT = TICKS2MS(TICK_DELAY) * 0.001f;

		for (int channel = 0; channel < NUM_BGC_CHANNELS; channel++) {

			// Update phase and keep within [0 360)
			phases[channel] += speeds[channel] * dT;
			if (phases[channel] < 0)
				phases[channel] += 360;
			if (phases[channel] >= 360)
				phases[channel] -= 360;

			PIOS_Brushless_SetPhase(channel, phases[channel] + phase_lag[channel]);
		}
	}
}