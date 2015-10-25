/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_ADC STM32F103 ADC1 Functions
 * @{
 *
 * @file       pios_internal_adc.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 * @brief      STM32F10x internal ADC PIOS interface
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

#include <pios_internal_adc_priv.h>

#if defined(PIOS_INCLUDE_ADC)

static void PIOS_INTERNAL_ADC_PinConfig(uint32_t internal_adc_id);
static void PIOS_INTERNAL_DMAConfig(uint32_t internal_adc_id);
int32_t PIOS_INTERNAL_ADC_Init(uint32_t *internal_adc_id, const struct pios_internal_adc_cfg *cfg);
static void PIOS_INTERNAL_ADC_Converter_Config(uint32_t internal_adc_id);
static bool PIOS_INTERNAL_ADC_Available(uint32_t internal_adc_id, uint32_t pin);
static int32_t PIOS_INTERNAL_ADC_PinGet(uint32_t internal_adc_id, uint32_t pin);
static uint8_t PIOS_INTERNAL_ADC_NumberOfChannels(uint32_t internal_adc_id);
static float PIOS_INTERNAL_ADC_LSB_Voltage(uint32_t internal_adc_id);

const struct pios_adc_driver pios_internal_adc_driver = {
		.available = PIOS_INTERNAL_ADC_Available,
		.get_pin = PIOS_INTERNAL_ADC_PinGet,
		.set_queue = NULL,
		.number_of_channels = PIOS_INTERNAL_ADC_NumberOfChannels,
		.lsb_voltage = PIOS_INTERNAL_ADC_LSB_Voltage,
};

// Private types
enum pios_internal_adc_dev_magic {
	PIOS_INTERNAL_ADC_DEV_MAGIC = 0xBD37C124,
};

struct pios_internal_adc_dev {
	const struct pios_internal_adc_cfg * cfg;
	uint16_t dma_transfer_size;
	volatile uint16_t *raw_data_buffer;
	enum pios_internal_adc_dev_magic magic;
};

/**
 * @brief Validates an internal ADC device
 * \return true if device is valid
 */
static bool PIOS_INTERNAL_ADC_validate(struct pios_internal_adc_dev * dev)
{
	if (dev == NULL )
		return false;

	return (dev->magic == PIOS_INTERNAL_ADC_DEV_MAGIC);
}

/**
 * @brief Allocates an internal ADC device
 */
static struct pios_internal_adc_dev * PIOS_INTERNAL_ADC_Allocate()
{
	struct pios_internal_adc_dev *adc_dev = (struct pios_internal_adc_dev *)PIOS_malloc(sizeof(*adc_dev));
	if (!adc_dev)
		return (NULL );
	adc_dev->magic = PIOS_INTERNAL_ADC_DEV_MAGIC;
	return (adc_dev);
}

/**
 * @brief Configures the pins used on the ADC device
 * \param[in] handle to the ADC device
 */
static void PIOS_INTERNAL_ADC_PinConfig(uint32_t internal_adc_id)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *) internal_adc_id;
	if (!PIOS_INTERNAL_ADC_validate(adc_dev)) {
		return;
	}
	/* Setup analog pins */
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_StructInit(&GPIO_InitStructure);
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;

	for (int32_t i = 0; i < adc_dev->cfg->number_of_used_pins; i++) {
		if (adc_dev->cfg->adc_pins[i].port == NULL )
			continue;
		GPIO_InitStructure.GPIO_Pin = adc_dev->cfg->adc_pins[i].pin;
		GPIO_Init(adc_dev->cfg->adc_pins[i].port, (GPIO_InitTypeDef*) &GPIO_InitStructure);
	}
}
/**
 * @brief Configures the DMA used on the ADC device
 * \param[in] handle to the ADC device
 */
static void PIOS_INTERNAL_DMAConfig(uint32_t internal_adc_id)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *) internal_adc_id;
	if (!PIOS_INTERNAL_ADC_validate(adc_dev)) {
		return;
	}

	/* Disable interrupts */
	DMA_ITConfig(adc_dev->cfg->dma.rx.channel, adc_dev->cfg->dma.irq.flags, DISABLE);

	/* Configure DMA channel */
	DMA_DeInit(adc_dev->cfg->dma.rx.channel);

	RCC_AHBPeriphClockCmd(adc_dev->cfg->dma.ahb_clk, ENABLE);

	DMA_InitTypeDef DMAInit = adc_dev->cfg->dma.rx.init;

	DMAInit.DMA_PeripheralBaseAddr = (uint32_t) &ADC1->DR;
	DMAInit.DMA_MemoryBaseAddr = (uint32_t) adc_dev->raw_data_buffer;
	DMAInit.DMA_BufferSize = adc_dev->dma_transfer_size;
	DMAInit.DMA_DIR = DMA_DIR_PeripheralSRC;
	DMAInit.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
	DMAInit.DMA_MemoryInc = DMA_MemoryInc_Enable;
	DMAInit.DMA_PeripheralDataSize = DMA_PeripheralDataSize_HalfWord;
	DMAInit.DMA_MemoryDataSize = DMA_MemoryDataSize_HalfWord;
	DMAInit.DMA_Mode = DMA_Mode_Circular;
	DMAInit.DMA_M2M = DMA_M2M_Disable;

	DMA_Init(adc_dev->cfg->dma.rx.channel, &DMAInit); /* channel is actually stream ... */

	/* enable DMA */
	DMA_Cmd(adc_dev->cfg->dma.rx.channel, ENABLE);
}
/**
 * @brief Configures the ADC device
 * \param[in] handle to the ADC device
 */
static void PIOS_INTERNAL_ADC_Converter_Config(uint32_t internal_adc_id)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *) internal_adc_id;

	ADC_DeInit(ADC1);

	RCC_APB2PeriphClockCmd(RCC_APB2Periph_ADC1, ENABLE);

	RCC_ADCCLKConfig(RCC_PCLK2_Div8);

	for (int32_t i = 0; i < adc_dev->cfg->number_of_used_pins; i++) {
		ADC_RegularChannelConfig(ADC1, adc_dev->cfg->adc_pins[i].adc_channel,
					 i + 1, ADC_SampleTime_239Cycles5);
	}

	ADC_InitTypeDef ADC_InitStructure;
	ADC_StructInit(&ADC_InitStructure);

	ADC_InitStructure.ADC_Mode = ADC_Mode_Independent;
	ADC_InitStructure.ADC_ScanConvMode = ENABLE;
	ADC_InitStructure.ADC_ContinuousConvMode = ENABLE;
	ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None;
	ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;
	ADC_InitStructure.ADC_NbrOfChannel = adc_dev->cfg->number_of_used_pins;

	ADC_Init(ADC1, &ADC_InitStructure);

	ADC_Cmd(ADC1, ENABLE);

	PIOS_DELAY_WaituS(10);
	ADC_ResetCalibration(ADC1);
	while(ADC_GetResetCalibrationStatus(ADC1));
	ADC_StartCalibration(ADC1);
	while (ADC_GetCalibrationStatus(ADC1));

	/* Enable ADC->DMA request */
	ADC_DMACmd(ADC1, ENABLE);

	PIOS_DELAY_WaituS(10);
	ADC_SoftwareStartConvCmd(ADC1, ENABLE);
}

/**
 * @brief Init the ADC.
 */
int32_t PIOS_INTERNAL_ADC_Init(uint32_t * internal_adc_id, const struct pios_internal_adc_cfg * cfg)
{
	PIOS_DEBUG_Assert(internal_adc_id); PIOS_DEBUG_Assert(cfg);

	struct pios_internal_adc_dev * adc_dev;
	adc_dev = PIOS_INTERNAL_ADC_Allocate();
	if (adc_dev == NULL )
		return -1;
	adc_dev->cfg = cfg;

	*internal_adc_id = (uint32_t) adc_dev;

	// DMA transfer size in units defined by DMA_PeripheralDataSize, 32bits for dual mode and 16bits for single mode
	adc_dev->dma_transfer_size = adc_dev->cfg->number_of_used_pins;
	adc_dev->raw_data_buffer = PIOS_malloc(adc_dev->dma_transfer_size * sizeof(uint16_t));
	if (adc_dev->raw_data_buffer == NULL )
		return -1;

	PIOS_INTERNAL_ADC_PinConfig((uint32_t) adc_dev);

	PIOS_INTERNAL_DMAConfig((uint32_t) adc_dev);

	PIOS_INTERNAL_ADC_Converter_Config((uint32_t) adc_dev);
	return 0;
}

/**
 * Checks if a pin is available on a certain ADC device
 * @param[in] pin number
 * @param[in] internal_adc_id handler to the device to check
 * @return True if the pin is available.
 */
static bool PIOS_INTERNAL_ADC_Available(uint32_t internal_adc_id, uint32_t pin)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *) internal_adc_id;
	/* Check if pin exists */
	if (pin >= adc_dev->cfg->number_of_used_pins) {
		return false;
	}
	return true;
}

/**
 * Checks the number of channels on a certain ADC device
 * @param[in] internal_adc_id handler to the device to check
 * @return number of channels on the device.
 */
static uint8_t PIOS_INTERNAL_ADC_NumberOfChannels(uint32_t internal_adc_id)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *) internal_adc_id;
	if (!PIOS_INTERNAL_ADC_validate(adc_dev))
		return 0;
	return adc_dev->cfg->number_of_used_pins;

}

/**
 * @brief Gets the value of an ADC pinn
 * @param[in] pin number, acording to the order passed on the configuration
 * @return ADC pin value averaged over the set of samples since the last reading.
 * @return -1 if pin doesn't exist
 */
static int32_t PIOS_INTERNAL_ADC_PinGet(uint32_t internal_adc_id, uint32_t pin)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *) internal_adc_id;

	/* Check if pin exists */
	if (pin >= adc_dev->cfg->number_of_used_pins) {
		return -1;
	}

	/* This results in a half-word load (2 cycles) of the volatile buffer location
	   written by DMA. The buffer is dynamically allocated and thus will always be aligned. 
	   A barrier is not required for this on Cortex-M. See ARM App. Note 321 */
	return adc_dev->raw_data_buffer[pin];
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
#endif /* PIOS_INCLUDE_ADC */
/** 
 * @}
 * @}
 */
