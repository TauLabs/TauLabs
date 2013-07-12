/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_HCSR04 HCSR04 Functions
 * @{
 *
 * @file       pios_hcsr04.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      HCSR04 sonar Sensor Routines
 *
 * @see        The GNU Public License (GPL) Version 3
 *
 ******************************************************************************/
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

#include "pios.h"

#ifdef PIOS_INCLUDE_HCSR04
#include "pios_hcsr04_priv.h"

/* Private constants */
#define HCSR04_TASK_PRIORITY (tskIDLE_PRIORITY + configMAX_PRIORITIES - 1) // max priority
#define HCSR04_TASK_STACK (512 / 4)
#define UPDATE_PERIOD_MS 100
/* Private methods */
static void PIOS_HCSR04_Task(void *parameters);

/* Local Variables */
/* 100 ms timeout without updates on channels */
const static uint32_t HCSR04_SUPERVISOR_TIMEOUT = 100000;

enum pios_hcsr04_dev_magic {
    PIOS_HCSR04_DEV_MAGIC = 0xab3029aa,
};

struct pios_hcsr04_dev {
	const struct pios_hcsr04_cfg *cfg;
	xTaskHandle task;
	xQueueHandle queue;

	uint8_t CaptureState[1];
	uint16_t RiseValue[1];
	uint16_t FallValue[1];
	uint32_t CaptureValue[1];
	uint32_t CapCounter[1];
	uint32_t us_since_update[1];
	enum pios_hcsr04_dev_magic magic;
};

static struct pios_hcsr04_dev *dev;

/**
 * @brief Allocate a new device
 */
#if defined(PIOS_INCLUDE_FREERTOS)
static struct pios_hcsr04_dev *PIOS_HCSR04_alloc(void)
{
	struct pios_hcsr04_dev *hcsr04_dev;

	hcsr04_dev = (struct pios_hcsr04_dev *)pvPortMalloc(sizeof(*hcsr04_dev));
	if (!hcsr04_dev) {
		return NULL;
	}

	hcsr04_dev->queue = xQueueCreate(1, sizeof(struct pios_sensor_sonar_data));
	if (hcsr04_dev->queue == NULL) {
		vPortFree(hcsr04_dev);
		return NULL;
	}

	hcsr04_dev->magic = PIOS_HCSR04_DEV_MAGIC;
	return hcsr04_dev;
}
#else
#error requires FreeRTOS
#endif /* if defined(PIOS_INCLUDE_FREERTOS) */

/**
 * @brief Validate the handle to the device
 * @returns 0 for valid device or <0 otherwise
 */
static bool PIOS_HCSR04_validate(struct pios_hcsr04_dev *dev)
{
	if (dev == NULL)
		return -1;
	if (dev->magic != PIOS_HCSR04_DEV_MAGIC)
		return -2;
	return 0;
}

static void PIOS_HCSR04_tim_overflow_cb(uintptr_t id, uintptr_t context, uint8_t channel, uint16_t count);
static void PIOS_HCSR04_tim_edge_cb(uintptr_t id, uintptr_t context, uint8_t channel, uint16_t count);
const static struct pios_tim_callbacks tim_callbacks = {
	.overflow = PIOS_HCSR04_tim_overflow_cb,
	.edge     = PIOS_HCSR04_tim_edge_cb,
};

/**
 * Initialises all the pins
 */
int32_t PIOS_HCSR04_Init(uintptr_t *hcsr04_id, const struct pios_hcsr04_cfg *cfg)
{
	PIOS_DEBUG_Assert(hcsr04_id);
	PIOS_DEBUG_Assert(cfg);

	struct pios_hcsr04_dev *hcsr04_dev;

	hcsr04_dev = (struct pios_hcsr04_dev *)PIOS_HCSR04_alloc();
	if (!hcsr04_dev) {
		return -1;
	}

	/* Bind the configuration to the device instance */
	hcsr04_dev->cfg = cfg;
	dev  = hcsr04_dev;

	/* Flush counter variables */
	hcsr04_dev->CaptureState[0] = 0;
	hcsr04_dev->RiseValue[0]    = 0;
	hcsr04_dev->FallValue[0]    = 0;
	hcsr04_dev->CaptureValue[0] = -1;

	uintptr_t tim_id;
	if (PIOS_TIM_InitChannels(&tim_id, cfg->channels, cfg->num_channels, &tim_callbacks, (uintptr_t)hcsr04_dev)) {
		return -1;
	}

	/* Configure the channels to be in capture/compare mode */

	const struct pios_tim_channel *chan   = &cfg->channels[0];

	/* Configure timer for input capture */
	TIM_ICInitTypeDef TIM_ICInitStructure = cfg->tim_ic_init;
	TIM_ICInitStructure.TIM_Channel = chan->timer_chan;
	TIM_ICInit(chan->timer, &TIM_ICInitStructure);

	/* Enable the Capture Compare Interrupt Request */
	switch (chan->timer_chan) {
	case TIM_Channel_1:
		TIM_ITConfig(chan->timer, TIM_IT_CC1, ENABLE);
		break;
	case TIM_Channel_2:
		TIM_ITConfig(chan->timer, TIM_IT_CC2, ENABLE);
		break;
	case TIM_Channel_3:
		TIM_ITConfig(chan->timer, TIM_IT_CC3, ENABLE);
		break;
	case TIM_Channel_4:
		TIM_ITConfig(chan->timer, TIM_IT_CC4, ENABLE);
		break;
	}

	// Need the update event for that timer to detect timeouts
	TIM_ITConfig(chan->timer, TIM_IT_Update, ENABLE);


	GPIO_Init(hcsr04_dev->cfg->trigger.gpio, (GPIO_InitTypeDef *)&hcsr04_dev->cfg->trigger.init);

	*hcsr04_id = (uintptr_t)hcsr04_dev;

	portBASE_TYPE result = xTaskCreate(PIOS_HCSR04_Task, (const signed char *)"pios_hcsr04",
	                                   HCSR04_TASK_STACK, NULL, HCSR04_TASK_PRIORITY,
	                                   &hcsr04_dev->task);
	PIOS_Assert(result == pdPASS);

	PIOS_SENSORS_Register(PIOS_SENSOR_SONAR, hcsr04_dev->queue);

	return 0;
}

static void PIOS_HCSR04_Trigger(void)
{
	GPIO_SetBits(dev->cfg->trigger.gpio, dev->cfg->trigger.init.GPIO_Pin);
	PIOS_DELAY_WaituS(15);
	GPIO_ResetBits(dev->cfg->trigger.gpio, dev->cfg->trigger.init.GPIO_Pin);
}

/**
 * Get the value of an input channel
 * \param[in] Channel Number of the channel desired
 * \output -1 Channel not available
 * \output >0 Channel value
 */
static int32_t PIOS_HCSR04_Get(void)
{
	return dev->CaptureValue[0];
}

static int32_t PIOS_HCSR04_Completed(void)
{
	return dev->CapCounter[0];
}

static void PIOS_HCSR04_tim_overflow_cb(uintptr_t tim_id, uintptr_t context, uint8_t channel, uint16_t count)
{
	struct pios_hcsr04_dev *hcsr04_dev = (struct pios_hcsr04_dev *)context;

	if (PIOS_HCSR04_validate(dev) != 0) {
		/* Invalid device specified */
		return;
	}

	if (channel >= hcsr04_dev->cfg->num_channels) {
		/* Channel out of range */
		return;
	}

	hcsr04_dev->us_since_update[channel] += count;
	if (hcsr04_dev->us_since_update[channel] >= HCSR04_SUPERVISOR_TIMEOUT) {
		hcsr04_dev->CaptureState[channel]    = 0;
		hcsr04_dev->RiseValue[channel]       = 0;
		hcsr04_dev->FallValue[channel]       = 0;
		hcsr04_dev->CaptureValue[channel]    = -1;
		hcsr04_dev->us_since_update[channel] = 0;
	}
}

static void PIOS_HCSR04_tim_edge_cb(uintptr_t tim_id, uintptr_t context, uint8_t chan_idx, uint16_t count)
{
	/* Recover our device context */
	struct pios_hcsr04_dev *hcsr04_dev = (struct pios_hcsr04_dev *)context;

	if (PIOS_HCSR04_validate(dev) != 0) {
		/* Invalid device specified */
		return;
	}

	if (chan_idx >= hcsr04_dev->cfg->num_channels) {
		/* Channel out of range */
		return;
	}

	const struct pios_tim_channel *chan = &hcsr04_dev->cfg->channels[chan_idx];

	if (hcsr04_dev->CaptureState[chan_idx] == 0) {
		hcsr04_dev->RiseValue[chan_idx] = count;
		hcsr04_dev->us_since_update[chan_idx] = 0;
	} else {
		hcsr04_dev->FallValue[chan_idx] = count;
	}

	// flip state machine and capture value here
	/* Simple rise or fall state machine */
	TIM_ICInitTypeDef TIM_ICInitStructure = hcsr04_dev->cfg->tim_ic_init;
	if (hcsr04_dev->CaptureState[chan_idx] == 0) {
		/* Switch states */
		hcsr04_dev->CaptureState[chan_idx] = 1;

		/* Switch polarity of input capture */
		TIM_ICInitStructure.TIM_ICPolarity = TIM_ICPolarity_Falling;
		TIM_ICInitStructure.TIM_Channel    = chan->timer_chan;
		TIM_ICInit(chan->timer, &TIM_ICInitStructure);
	} else {
		/* Capture computation */
		if (hcsr04_dev->FallValue[chan_idx] > hcsr04_dev->RiseValue[chan_idx]) {
			hcsr04_dev->CaptureValue[chan_idx] = (hcsr04_dev->FallValue[chan_idx] - hcsr04_dev->RiseValue[chan_idx]);
		} else {
			hcsr04_dev->CaptureValue[chan_idx] = ((chan->timer->ARR - hcsr04_dev->RiseValue[chan_idx]) + hcsr04_dev->FallValue[chan_idx]);
		}

		/* Switch states */
		hcsr04_dev->CaptureState[chan_idx] = 0;

		/* Increase supervisor counter */
		hcsr04_dev->CapCounter[chan_idx]++;

		/* Switch polarity of input capture */
		TIM_ICInitStructure.TIM_ICPolarity = TIM_ICPolarity_Rising;
		TIM_ICInitStructure.TIM_Channel    = chan->timer_chan;
		TIM_ICInit(chan->timer, &TIM_ICInitStructure);
	}
}

static void PIOS_HCSR04_Task(void *parameters)
{
	int32_t value = 0, timeout = 10;
	float coeff   = 0.25, height_out = 0, height_in = 0;
	PIOS_HCSR04_Trigger();

	while (1) {
		struct pios_sensor_sonar_data data;
		// Compute the current altitude
		if (PIOS_HCSR04_Completed()) {
			value = PIOS_HCSR04_Get();
			// from 2.5cm to 2.5m
			if ((value > 150) && (value < 15000)) {
				height_in  = value * 0.00034f / 2.0f;
				height_out = (height_out * (1 - coeff)) + (height_in * coeff);

				data.altitude = height_out; // m/us
				data.range = true;
				xQueueSend(dev->queue, (void *)&data, 0);
			} else {
				if (value <= 150) data.altitude = -1;
				data.range = false;
				xQueueSend(dev->queue, (void *)&data, 0);
			}

			timeout = 10;
			PIOS_HCSR04_Trigger();
		}
		if (!(timeout--)) {
			// retrigger
			timeout = 10;
			PIOS_HCSR04_Trigger();
		}

		vTaskDelay(UPDATE_PERIOD_MS / portTICK_RATE_MS);
	}
}

#endif /* PIOS_INCLUDE_HCSR04 */
