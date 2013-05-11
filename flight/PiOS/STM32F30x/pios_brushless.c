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
	uint32_t tim_id;
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

	// PB10 and PB1 enabled and on for SparkyBGC
	// TODO put into configuration
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
	GPIO_SetBits(GPIOB, GPIO_InitStructure.GPIO_Pin);

	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
	GPIO_SetBits(GPIOB, GPIO_InitStructure.GPIO_Pin);

	// Start main task
	xTaskCreate(PIOS_BRUSHLESS_Task, (signed char*)"PIOS_BRUSHLESS", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);

	return 0;
}

float    phases[NUM_BGC_CHANNELS];
float    speeds[NUM_BGC_CHANNELS];
float    scales[NUM_BGC_CHANNELS];
int16_t  scale = 30;
int32_t  center = 300;

/**
* Set the servo update rate (Max 500Hz)
* \param[in] rate in Hz
*/
void PIOS_Brushless_SetUpdateRate(uint32_t rate)
{
	if (!brushless_cfg) {
		return;
	}

	TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure = brushless_cfg->tim_base_init;
	TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1;
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;

	for(uint8_t i = 0; i < brushless_cfg->num_channels; i++) {
		bool new = true;
		const struct pios_tim_channel * chan = &brushless_cfg->channels[i];

		/* See if any previous channels use that same timer */
		for(uint8_t j = 0; (j < i) && new; j++)
			new &= chan->timer != brushless_cfg->channels[j].timer;

		if(new) {
			// Choose the correct prescaler value for the APB the timer is attached
			if (chan->timer==TIM2 || chan->timer==TIM3 || chan->timer==TIM4 || chan->timer==TIM6 || chan->timer==TIM7 ){
				//those timers run at double APB1 speed if APB1 prescaler is != 1 which is usually the case
				TIM_TimeBaseStructure.TIM_Prescaler = (PIOS_PERIPHERAL_APB1_CLOCK / 36000000 * 2) - 1;
			}
			else {
				TIM_TimeBaseStructure.TIM_Prescaler = (PIOS_PERIPHERAL_APB2_CLOCK / 36000000) - 1;
			}

			TIM_TimeBaseStructure.TIM_Period = ((36000000 / rate) - 1);
			TIM_TimeBaseInit(chan->timer, &TIM_TimeBaseStructure);
		}
	}

	// Set some default reasonable parameters
	center = TIM_TimeBaseStructure.TIM_Period / 2;
	scale = center;
}

/**
* Set servo position
* \param[in] Servo Servo number (0-7)
* \param[in] Position Servo position in microseconds
*/
void PIOS_Brushless_SetSpeed(uint32_t channel, float speed)
{
	// TODO: Store the speed for that output in a structure and
	// use the timer overflow event to time incrementing

	phases[channel] += speed;
	PIOS_Brushless_SetPhase(channel, phases[channel]);
}

//! Set the amplitude scale in %
void PIOS_Brushless_SetScale(uint8_t roll, uint8_t pitch, uint8_t yaw)
{
	scales[0] = (float) roll / 100.0f;
	scales[1] = (float) pitch / 100.0f;
	scales[2] = (float) yaw / 100.0f;
}


/**
 * PIOS_Brushless_SetPhase set the phase for one of the channel outputs
 * @param[in] channel The channel to set
 * @param[in] phase_deg The phase in degrees to use
 */
static int32_t PIOS_Brushless_SetPhase(uint32_t channel, float phase_deg)
{
	/* Make sure all the channels exist */
	if (!brushless_cfg || (3 * (channel + 1)) > brushless_cfg->num_channels) {
		return -1;
	}

	// Get the first output index
	for (int32_t idx = channel * 3; idx < (channel + 1) * 3; idx++) {

		// sin lookup expects between 0 and 360
		if (phase_deg > 360)
			phase_deg -= 360;

		int32_t position = center + scale * scales[channel] * sinf(phase_deg * DEG2RAD);

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
	while (1) {

		vTaskDelay(1);

		const float dT = 0.001f;

		for (int channel = 0; channel < NUM_BGC_CHANNELS; channel++) {

			// Update phase and keep within [0 360)
			phases[channel] += speeds[channel] * dT;
			if (phases[channel] < 0)
				phases[channel] += 360;
			if (phases[channel] >= 360)
				phases[channel] -= 360;

			PIOS_Brushless_SetPhase(channel, phases[channel]);
		}
	}
}
