/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_INTERNAL_ADC Internal ADC Functions
 * @{
 *
 * @file       pios_internal_adc.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Analog to Digital conversion routines
 * @see        The GNU Public License (GPL) Version 3
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

#include "pios.h"
#include <pios_internal_adc_priv.h>

// Private functions
static void PIOS_INTERNAL_ADC_downsample_data(uint32_t internal_adc_id);
static struct pios_internal_adc_dev * PIOS_INTERNAL_ADC_Allocate();
static bool PIOS_INTERNAL_ADC_validate(struct pios_internal_adc_dev *);
static void PIOS_INTERNAL_ADC_Config(uint32_t internal_adc_id, uint32_t oversampling);
static int32_t PIOS_INTERNAL_ADC_PinGet(uint32_t internal_adc_id, uint32_t pin);
#if defined(PIOS_INCLUDE_FREERTOS)
static void PIOS_INTERNAL_ADC_SetQueue(uint32_t internal_adc_id, xQueueHandle data_queue);
#endif
static uint8_t PIOS_INTERNAL_ADC_Number_of_Channels(uint32_t internal_adc_id);
static bool PIOS_INTERNAL_ADC_Available(uint32_t adc_id, uint32_t device_pin);
static float PIOS_INTERNAL_ADC_LSB_Voltage(uint32_t internal_adc_id);

// Private types
enum pios_internal_adc_dev_magic {
	PIOS_INTERNAL_ADC_DEV_MAGIC = 0x58375124,
};

const struct pios_adc_driver pios_internal_adc_driver = {
		.available	= PIOS_INTERNAL_ADC_Available,
		.get_pin	= PIOS_INTERNAL_ADC_PinGet,
		.set_queue	= PIOS_INTERNAL_ADC_SetQueue,
		.number_of_channels = PIOS_INTERNAL_ADC_Number_of_Channels,
		.lsb_voltage 	= PIOS_INTERNAL_ADC_LSB_Voltage,
};
struct pios_internal_adc_dev {
	const struct pios_internal_adc_cfg * cfg;
	ADCCallback callback_function;
#if defined(PIOS_INCLUDE_FREERTOS)
	xQueueHandle data_queue;
#endif
	volatile int16_t *valid_data_buffer;
	volatile uint8_t adc_oversample;
	uint8_t dma_block_size;
	uint16_t dma_half_buffer_size;
#if defined(PIOS_INCLUDE_ADC)
	int16_t fir_coeffs[PIOS_ADC_MAX_SAMPLES+1]  __attribute__ ((aligned(4)));
	volatile int16_t raw_data_buffer[PIOS_ADC_MAX_SAMPLES]  __attribute__ ((aligned(4)));	// Double buffer that DMA just used
	float downsampled_buffer[PIOS_ADC_NUM_CHANNELS]  __attribute__ ((aligned(4)));
#endif
	enum pios_internal_adc_dev_magic magic;
};

/* Local Variables */
static GPIO_TypeDef *ADC_GPIO_PORT[PIOS_ADC_NUM_PINS] = PIOS_ADC_PORTS;
static const uint32_t ADC_GPIO_PIN[PIOS_ADC_NUM_PINS] = PIOS_ADC_PINS;
static const uint32_t ADC_CHANNEL[PIOS_ADC_NUM_PINS] = PIOS_ADC_CHANNELS;

static ADC_TypeDef *ADC_MAPPING[PIOS_ADC_NUM_PINS] = PIOS_ADC_MAPPING;
static const uint32_t ADC_CHANNEL_MAPPING[PIOS_ADC_NUM_PINS] = PIOS_ADC_CHANNEL_MAPPING;

struct pios_internal_adc_dev * static_adc_dev = NULL;


/**
  * @brief Validates the ADC device
  * \param[in] dev pointer to device structure
  * \return true if valid
  */
static bool PIOS_INTERNAL_ADC_validate(struct pios_internal_adc_dev * dev)
{
	if (dev == NULL)
		return false;

	return (dev->magic == PIOS_INTERNAL_ADC_DEV_MAGIC);
}

/**
  * @brief Allocates an internal ADC device in memory
  * \param[out] pointer to the newly created device
  *
  */
static struct pios_internal_adc_dev * PIOS_INTERNAL_ADC_Allocate()
{
	if(static_adc_dev)
		return NULL;
	struct pios_internal_adc_dev * adc_dev;
	
	adc_dev = (struct pios_internal_adc_dev *)PIOS_malloc(sizeof(*adc_dev));
	if (!adc_dev) return (NULL);
	
	adc_dev->magic = PIOS_INTERNAL_ADC_DEV_MAGIC;

	static_adc_dev = adc_dev;

	return(adc_dev);
}

/**
  * @brief Initializes an internal ADC device
  * \param[out] internal_adc_id handle to the device
  * \param[in] cfg device configuration
  * \return < 0 if deviced initialization failed
  */
int32_t PIOS_INTERNAL_ADC_Init(uint32_t * internal_adc_id, const struct pios_internal_adc_cfg * cfg)
{
	PIOS_DEBUG_Assert(internal_adc_id);
	PIOS_DEBUG_Assert(cfg);

	struct pios_internal_adc_dev * adc_dev;
	adc_dev = PIOS_INTERNAL_ADC_Allocate();
	if (adc_dev == NULL)
		return -1;

	adc_dev->cfg = cfg;
	adc_dev->callback_function = NULL;
	
#if defined(PIOS_INCLUDE_FREERTOS)
	adc_dev->data_queue = NULL;
#endif
	
	/* Setup analog pins */
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_StructInit(&GPIO_InitStructure);
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
	
	/* Enable each ADC pin in the array */
	for (int32_t i = 0; i < PIOS_ADC_NUM_PINS; i++) {
		GPIO_InitStructure.GPIO_Pin = ADC_GPIO_PIN[i];
		GPIO_Init(ADC_GPIO_PORT[i], &GPIO_InitStructure);
	}

	*internal_adc_id = (uint32_t)adc_dev;

	PIOS_INTERNAL_ADC_Config(*internal_adc_id, cfg->oversampling);
	
	return 0;
}

/**
 * @brief Configure the ADC to run at a fixed oversampling
 * @param[in] oversampling the amount of oversampling to run at
 * @param[in] internal_adc_id handle to the device
 */
static void PIOS_INTERNAL_ADC_Config(uint32_t internal_adc_id, uint32_t oversampling)
{	
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *)internal_adc_id;
	if(!PIOS_INTERNAL_ADC_validate(adc_dev))
	{
		return;
	}
	adc_dev->adc_oversample = (oversampling > PIOS_ADC_MAX_OVERSAMPLING) ? PIOS_ADC_MAX_OVERSAMPLING : oversampling;

	ADC_DeInit(ADC1);
	ADC_DeInit(ADC2);
	
	/* Disable interrupts */
	DMA_ITConfig(adc_dev->cfg->dma.rx.channel, adc_dev->cfg->dma.irq.flags, DISABLE);
	
	/* Enable ADC clocks */
	PIOS_ADC_CLOCK_FUNCTION;
	
	/* Map channels to conversion slots depending on the channel selection mask */
	for (int32_t i = 0; i < PIOS_ADC_NUM_PINS; i++) {
		ADC_RegularChannelConfig(ADC_MAPPING[i], ADC_CHANNEL[i],
					 ADC_CHANNEL_MAPPING[i],
					 PIOS_ADC_SAMPLE_TIME);
	}
	
#if (PIOS_ADC_USE_TEMP_SENSOR)
	ADC_TempSensorVrefintCmd(ENABLE);
	ADC_RegularChannelConfig(PIOS_ADC_TEMP_SENSOR_ADC, ADC_Channel_16,
				 PIOS_ADC_TEMP_SENSOR_ADC_CHANNEL,
				 PIOS_ADC_SAMPLE_TIME);
#endif
	// return	
	/* Configure ADCs */
	ADC_InitTypeDef ADC_InitStructure;
	ADC_StructInit(&ADC_InitStructure);
	ADC_InitStructure.ADC_Mode = ADC_Mode_RegSimult;
	ADC_InitStructure.ADC_ScanConvMode = ENABLE;
	ADC_InitStructure.ADC_ContinuousConvMode = ENABLE;
	ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None;
	ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;
	ADC_InitStructure.ADC_NbrOfChannel = ((PIOS_ADC_NUM_CHANNELS + 1) >> 1);
	ADC_Init(ADC1, &ADC_InitStructure);
	
#if (PIOS_ADC_USE_ADC2)
	ADC_Init(ADC2, &ADC_InitStructure);
	
	/* Enable ADC2 external trigger conversion (to synch with ADC1) */
	ADC_ExternalTrigConvCmd(ADC2, ENABLE);
#endif
	
	RCC_ADCCLKConfig(PIOS_ADC_ADCCLK);
		
	/* Enable ADC1->DMA request */
	ADC_DMACmd(ADC1, ENABLE);
	
	/* ADC1 calibration */
	ADC_Cmd(ADC1, ENABLE);
	ADC_ResetCalibration(ADC1);
	while (ADC_GetResetCalibrationStatus(ADC1)) ;
	ADC_StartCalibration(ADC1);
	while (ADC_GetCalibrationStatus(ADC1)) ;
	
#if (PIOS_ADC_USE_ADC2)
	/* ADC2 calibration */
	ADC_Cmd(ADC2, ENABLE);
	ADC_ResetCalibration(ADC2);
	while (ADC_GetResetCalibrationStatus(ADC2)) ;
	ADC_StartCalibration(ADC2);
	while (ADC_GetCalibrationStatus(ADC2)) ;
#endif
	
	/* This makes sure we have an even number of transfers if using ADC2 */
	adc_dev->dma_block_size = ((PIOS_ADC_NUM_CHANNELS + PIOS_ADC_USE_ADC2) >> PIOS_ADC_USE_ADC2) << PIOS_ADC_USE_ADC2;
	adc_dev->dma_half_buffer_size = adc_dev->dma_block_size * adc_dev->adc_oversample;

	/* Configure DMA channel */		
	DMA_InitTypeDef dma_init = adc_dev->cfg->dma.rx.init;
	dma_init.DMA_MemoryBaseAddr = (uint32_t) &adc_dev->raw_data_buffer[0];
	dma_init.DMA_MemoryInc = DMA_MemoryInc_Enable;
	dma_init.DMA_BufferSize = adc_dev->dma_half_buffer_size; /* x2 for double buffer /2 for 32-bit xfr */
	DMA_Init(adc_dev->cfg->dma.rx.channel, &dma_init);
	DMA_Cmd(adc_dev->cfg->dma.rx.channel, ENABLE);
	
	/* Trigger interrupt when for half conversions too to indicate double buffer */
	DMA_ITConfig(adc_dev->cfg->dma.rx.channel, DMA_IT_TC, ENABLE);
        DMA_ITConfig(adc_dev->cfg->dma.rx.channel, DMA_IT_HT, ENABLE);
	
	/* Configure DMA interrupt */
	NVIC_Init((NVIC_InitTypeDef*)&adc_dev->cfg->dma.irq.init);
	
	/* Finally start initial conversion */
	ADC_SoftwareStartConvCmd(ADC1, ENABLE);
	
	/* Use simple averaging filter for now */
	for (int32_t i = 0; i < adc_dev->adc_oversample; i++)
		adc_dev->fir_coeffs[i] = 1;
	adc_dev->fir_coeffs[adc_dev->adc_oversample] = adc_dev->adc_oversample;
	
	/* Enable DMA1 clock */
	RCC_AHBPeriphClockCmd(adc_dev->cfg->dma.ahb_clk, ENABLE);
}

/**
 * Returns value of an ADC Pin
 * \param[in] pin number
 * \param[in] internal_adc_id handle to the device
 *
 * \return ADC pin value - resolution depends on the selected oversampling rate
 * \return -1 if pin doesn't exist
 */
static int32_t PIOS_INTERNAL_ADC_PinGet(uint32_t internal_adc_id, uint32_t pin)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *)internal_adc_id;
	if(!PIOS_INTERNAL_ADC_validate(adc_dev))
	{
		return -2;
	}
	/* Check if pin exists */
	if (pin >= PIOS_ADC_NUM_CHANNELS) {
		return -1;
	}
	
	/* Return last conversion result */
	return adc_dev->downsampled_buffer[pin];
}

#if defined(PIOS_INCLUDE_FREERTOS)
/**
 * @brief Register a queue to add data to when downsampled 
 * \param[in] internal_adc_id handle to the device
 */
static void PIOS_INTERNAL_ADC_SetQueue(uint32_t internal_adc_id, xQueueHandle data_queue)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *)internal_adc_id;
	if(!PIOS_INTERNAL_ADC_validate(adc_dev))
	{
		return;
	}
	adc_dev->data_queue = data_queue;
}
#endif

/**
 * @brief Downsample the data for each of the channels then call
 * callback function if installed
 * \param[in] internal_adc_id handle to the device
 */ 
static void PIOS_INTERNAL_ADC_downsample_data(uint32_t internal_adc_id)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *)internal_adc_id;
	if(!PIOS_INTERNAL_ADC_validate(adc_dev))
	{
		return;
	}
	uint16_t chan;
	uint16_t sample;
	float * downsampled_buffer = &adc_dev->downsampled_buffer[0];
	
	for (chan = 0; chan < PIOS_ADC_NUM_CHANNELS; chan++) {
		int32_t sum = 0;
		for (sample = 0; sample < adc_dev->adc_oversample; sample++) {
			sum += adc_dev->valid_data_buffer[chan + sample * adc_dev->dma_block_size] * adc_dev->fir_coeffs[sample];
		}
		downsampled_buffer[chan] = (float) sum / adc_dev->fir_coeffs[adc_dev->adc_oversample];
	}
	
#if defined(PIOS_INCLUDE_FREERTOS)
	if(adc_dev->data_queue) {
		static portBASE_TYPE xHigherPriorityTaskWoken;
		xQueueSendFromISR(adc_dev->data_queue, adc_dev->downsampled_buffer, &xHigherPriorityTaskWoken);
		portEND_SWITCHING_ISR(xHigherPriorityTaskWoken);		
	}
#endif
	if(adc_dev->callback_function)
		adc_dev->callback_function(adc_dev->downsampled_buffer);
}

/**
 * @brief Interrupt for half and full buffer transfer
 * 
 * This interrupt handler swaps between the two halfs of the double buffer to make
 * sure the ahrs uses the most recent data.  Only swaps data when AHRS is idle, but
 * really this is a pretense of a sanity check since the DMA engine is consantly 
 * running in the background.  Keep an eye on the ekf_too_slow variable to make sure
 * it's keeping up.
 */
void PIOS_INTERNAL_ADC_DMA_Handler()
{
	if(!PIOS_INTERNAL_ADC_validate(static_adc_dev))
		return;
	if (DMA_GetFlagStatus(static_adc_dev->cfg->full_flag /*DMA1_IT_TC1*/)) {	// whole double buffer filled
		static_adc_dev->valid_data_buffer = &static_adc_dev->raw_data_buffer[static_adc_dev->dma_half_buffer_size];
		DMA_ClearFlag(static_adc_dev->cfg->full_flag);
		PIOS_INTERNAL_ADC_downsample_data((uint32_t)static_adc_dev);
	}
	else if (DMA_GetFlagStatus(static_adc_dev->cfg->half_flag /*DMA1_IT_HT1*/)) {
		static_adc_dev->valid_data_buffer = &static_adc_dev->raw_data_buffer[0];
		DMA_ClearFlag(static_adc_dev->cfg->half_flag);
		PIOS_INTERNAL_ADC_downsample_data((uint32_t)static_adc_dev);
	}
	else {
		// This should not happen, probably due to transfer errors
		DMA_ClearFlag(static_adc_dev->cfg->dma.irq.flags /*DMA1_FLAG_GL1*/);
	}
}

/**
  * @brief Checks if a given pin is available on the given device
  * \param[in] adc_id handle of the device to read
  * \param[in] device_pin pin to check if available
  * \return true if available
  */
static bool PIOS_INTERNAL_ADC_Available(uint32_t adc_id, uint32_t device_pin) {
	/* Check if pin exists */
	return (!(device_pin >= PIOS_ADC_NUM_CHANNELS));
}


/**
  * @brief Checks the number of available ADC channels on the device
  * \param[in] adc_id handle of the device
  * \return number of ADC channels of the device
  */
static uint8_t PIOS_INTERNAL_ADC_Number_of_Channels(uint32_t internal_adc_id)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *)internal_adc_id;
	if(!PIOS_INTERNAL_ADC_validate(adc_dev))
			return 0;
	return PIOS_ADC_NUM_CHANNELS;
}

/**
 * @brief Gets the least significant bit voltage of the ADC
 */
static float PIOS_INTERNAL_ADC_LSB_Voltage(uint32_t internal_adc_id)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *) internal_adc_id;
	if (!PIOS_INTERNAL_ADC_validate(adc_dev)) {
		return 0;
	}
        return VREF_PLUS / (((uint32_t)1 << 12) - 1);
}
/** 
 * @}
 * @}
 */
