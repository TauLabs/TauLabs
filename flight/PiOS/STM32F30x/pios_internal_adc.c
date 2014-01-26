/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_ADC ADC Functions
 * @{
 *
 * @file       pios_internal_adc.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      STM32F30x internal ADC PIOS interface
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

static void PIOS_INTERNAL_ADC_DMA_Handler1(void);
static void PIOS_INTERNAL_ADC_DMA_Handler2(void);
static void PIOS_INTERNAL_ADC_DMA_Handler3(void);
static void PIOS_INTERNAL_ADC_DMA_Handler4(void);

// Private types
enum pios_internal_adc_dev_magic {
	PIOS_INTERNAL_ADC_DEV_MAGIC = 0x58375124,
};

struct adc_accumulator {
	uint32_t accumulator;
	uint32_t count;
};

struct pios_internal_adc_dev * driver_instances[PIOS_INTERNAL_ADC_MAX_INSTANCES];
static uint8_t current_instances = 0;

struct pios_internal_adc_dev {
	const struct pios_internal_adc_cfg * cfg;
	uint8_t number_used_master_channels;
	uint8_t number_used_slave_channels;
	uint8_t regular_group_size;
	struct adc_accumulator **channel_map;
	struct adc_accumulator *accumulator;
	uint16_t dma_half_buffer_index;
	uint16_t dma_transfer_size;
	uint16_t *raw_data_buffer;
	uint32_t accumulator_increment;
	uint32_t accumulator_scan_size;
	enum pios_internal_adc_dev_magic magic;
};
static void PIOS_ADC_DMA_Handler(struct pios_internal_adc_dev *);

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
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AN;

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

	switch (current_instances) {
	case 1:
		PIOS_DMA_Install_Interrupt_handler(adc_dev->cfg->dma.rx.channel, &PIOS_INTERNAL_ADC_DMA_Handler1);
		break;
	case 2:
		PIOS_DMA_Install_Interrupt_handler(adc_dev->cfg->dma.rx.channel, &PIOS_INTERNAL_ADC_DMA_Handler2);
		break;
	case 3:
		PIOS_DMA_Install_Interrupt_handler(adc_dev->cfg->dma.rx.channel, &PIOS_INTERNAL_ADC_DMA_Handler3);
		break;
	case 4:
		PIOS_DMA_Install_Interrupt_handler(adc_dev->cfg->dma.rx.channel, &PIOS_INTERNAL_ADC_DMA_Handler4);
		break;
	default:
		break;
	}

	/* Configure DMA channel */
	DMA_DeInit(adc_dev->cfg->dma.rx.channel);
	DMA_InitTypeDef DMAInit = adc_dev->cfg->dma.rx.init;
	if (adc_dev->cfg->adc_dev_slave) {
		if (adc_dev->cfg->adc_dev_master == ADC1 )
			DMAInit.DMA_PeripheralBaseAddr = (uint32_t) &ADC1_2->CDR;
		else
			DMAInit.DMA_PeripheralBaseAddr = (uint32_t) &ADC3_4->CDR;

	} else
		DMAInit.DMA_PeripheralBaseAddr = (uint32_t) &adc_dev->cfg->adc_dev_master->DR;

	DMAInit.DMA_MemoryBaseAddr = (uint32_t) adc_dev->raw_data_buffer;
	DMAInit.DMA_BufferSize = adc_dev->dma_transfer_size;
	DMAInit.DMA_DIR = DMA_DIR_PeripheralSRC;
	DMAInit.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
	DMAInit.DMA_MemoryInc = DMA_MemoryInc_Enable;
	if (adc_dev->cfg->adc_dev_slave) {
		DMAInit.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Word;
		DMAInit.DMA_MemoryDataSize = DMA_MemoryDataSize_Word;
	} else {
		DMAInit.DMA_PeripheralDataSize = DMA_PeripheralDataSize_HalfWord;
		DMAInit.DMA_MemoryDataSize = DMA_MemoryDataSize_HalfWord;
	}
	DMAInit.DMA_Mode = DMA_Mode_Circular;
	DMAInit.DMA_M2M = DMA_M2M_Disable;

	DMA_Init(adc_dev->cfg->dma.rx.channel, &DMAInit); /* channel is actually stream ... */

	/* enable DMA */
	DMA_Cmd(adc_dev->cfg->dma.rx.channel, ENABLE);

	/* configure for double-buffered mode and interrupt on every buffer flip */
	DMA_ITConfig(adc_dev->cfg->dma.rx.channel, DMA_IT_TC, ENABLE);
	DMA_ITConfig(adc_dev->cfg->dma.rx.channel, DMA_IT_HT, ENABLE);

	/* Configure DMA interrupt */
	NVIC_InitTypeDef NVICInit = adc_dev->cfg->dma.irq.init;
	NVIC_Init(&NVICInit);
}
/**
 * @brief Configures the ADC device
 * \param[in] handle to the ADC device
 */
static void PIOS_INTERNAL_ADC_Converter_Config(uint32_t internal_adc_id)
{
	struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *) internal_adc_id;

	ADC_DeInit(adc_dev->cfg->adc_dev_master);
	if (adc_dev->cfg->adc_dev_slave)
		ADC_DeInit(adc_dev->cfg->adc_dev_slave);

	if (adc_dev->cfg->adc_dev_master == ADC1 || adc_dev->cfg->adc_dev_master == ADC2 )
		RCC_ADCCLKConfig(RCC_ADC12PLLCLK_Div16);
	else
		RCC_ADCCLKConfig(RCC_ADC34PLLCLK_Div16);

	ADC_VoltageRegulatorCmd(adc_dev->cfg->adc_dev_master, ENABLE);
	PIOS_DELAY_WaituS(10);
	ADC_SelectCalibrationMode(adc_dev->cfg->adc_dev_master, ADC_CalibrationMode_Single);
	ADC_StartCalibration(adc_dev->cfg->adc_dev_master);
	while (ADC_GetCalibrationStatus(adc_dev->cfg->adc_dev_master) != RESET)
		;
	/* Slave calibration */
	if (adc_dev->cfg->adc_dev_slave) {
		ADC_VoltageRegulatorCmd(adc_dev->cfg->adc_dev_slave, ENABLE);
		PIOS_DELAY_WaituS(10);
		ADC_SelectCalibrationMode(adc_dev->cfg->adc_dev_slave, ADC_CalibrationMode_Single);
		ADC_StartCalibration(adc_dev->cfg->adc_dev_slave);
		while (ADC_GetCalibrationStatus(adc_dev->cfg->adc_dev_slave) != RESET)
			;
	}

	if (adc_dev->cfg->adc_dev_master == ADC1 || adc_dev->cfg->adc_dev_master == ADC2)
		RCC_ADCCLKConfig(RCC_ADC12PLLCLK_Div256);
	else
		RCC_ADCCLKConfig(RCC_ADC34PLLCLK_Div256);

	/* Do common ADC init */
	ADC_CommonInitTypeDef ADC_CommonInitStructure;
	ADC_CommonStructInit(&ADC_CommonInitStructure);
	if (adc_dev->cfg->adc_dev_slave) {
		ADC_CommonInitStructure.ADC_Mode = ADC_Mode_RegSimul;
		ADC_CommonInitStructure.ADC_DMAAccessMode = ADC_DMAAccessMode_1;
	} else {
		ADC_CommonInitStructure.ADC_Mode = ADC_Mode_Independent;
		ADC_CommonInitStructure.ADC_DMAAccessMode = ADC_DMAAccessMode_Disabled;
	}

	ADC_CommonInitStructure.ADC_Clock = ADC_Clock_AsynClkMode;
	ADC_CommonInitStructure.ADC_DMAMode = ADC_DMAMode_Circular;
	ADC_CommonInitStructure.ADC_TwoSamplingDelay = 0;
	if (adc_dev->cfg->adc_dev_slave == NULL )
		ADC_DMAConfig(adc_dev->cfg->adc_dev_master, ADC_DMAMode_Circular);
	ADC_CommonInit(adc_dev->cfg->adc_dev_master, &ADC_CommonInitStructure);

	ADC_InitTypeDef ADC_InitStructure;
	ADC_StructInit(&ADC_InitStructure);
	ADC_InitStructure.ADC_Resolution = ADC_Resolution_12b;
	ADC_InitStructure.ADC_ContinuousConvMode = ADC_ContinuousConvMode_Enable;
	ADC_InitStructure.ADC_ExternalTrigConvEvent = ADC_ExternalTrigConvEvent_0;
	ADC_InitStructure.ADC_ExternalTrigEventEdge = ADC_ExternalTrigEventEdge_None;
	ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;

	ADC_InitStructure.ADC_NbrOfRegChannel = adc_dev->regular_group_size;

	ADC_Init(adc_dev->cfg->adc_dev_master, &ADC_InitStructure);

	if (adc_dev->cfg->adc_dev_slave)
		ADC_Init(adc_dev->cfg->adc_dev_slave, &ADC_InitStructure);

	/* Enable DMA request */
	ADC_DMACmd(adc_dev->cfg->adc_dev_master, ENABLE);

	/* Configure input scan
	 * channel_map indexing corresponds to each conversion slot in order
	 * in single channel mode this corresponds to 0,1,2,3...
	 * in dual channel mode this corresponds to 0,2,4...
	 *                                          1,3,5...
	 * channel_map value is a pointer to an accumulator, if the same channel is used multiple times the same accumulator is used
	 *
	 * Input scan is setup to repeat channels if needed, i.e if a channel has more conversions to make the other will repeat conversions
	 * example:
	 * 2 ADC1 pins to convert pinA1, pinA2
	 * 3 ADC2 pins to convert pinB1, pinB2, pinB3
	 * input scan becomes:
	 * pinA1, pinA2, pinA1
	 * pinB1, pinB2, pinB3
	 */
	uint32_t current_index = 0;
	if (!adc_dev->cfg->adc_dev_slave) {
		for (uint32_t i = 0; i < adc_dev->cfg->number_of_used_pins; i++) {
			ADC_RegularChannelConfig(adc_dev->cfg->adc_dev_master, adc_dev->cfg->adc_pins[i].adc_channel, current_index + 1, ADC_SampleTime_61Cycles5); /* XXX this is totally arbitrary... */
			adc_dev->channel_map[current_index] = &adc_dev->accumulator[current_index];
			++current_index;
		}
	} else {
		bool again = true;
		current_index = 0;
		uint32_t acc_index = 0;
		while (again) {
			for (uint32_t i = 0; i < adc_dev->cfg->number_of_used_pins; i++) {
				if (adc_dev->cfg->adc_pins[i].is_master_channel) {
					ADC_RegularChannelConfig(adc_dev->cfg->adc_dev_master, adc_dev->cfg->adc_pins[i].adc_channel, current_index + 1, ADC_SampleTime_61Cycles5); /* XXX this is totally arbitrary... */
					adc_dev->channel_map[acc_index] = &adc_dev->accumulator[i];
					++current_index;
					acc_index += 2;
					if (current_index
					                == (adc_dev->number_used_master_channels > adc_dev->number_used_slave_channels ?
					                                adc_dev->number_used_master_channels : adc_dev->number_used_slave_channels)) {
						again = false;
						break;
					}
				}
			}
		}
		again = true;
		current_index = 0;
		acc_index = 1;
		while (again) {
			for (uint32_t i = 0; i < adc_dev->cfg->number_of_used_pins; i++) {
				if (!adc_dev->cfg->adc_pins[i].is_master_channel) {
					ADC_RegularChannelConfig(adc_dev->cfg->adc_dev_slave, adc_dev->cfg->adc_pins[i].adc_channel, current_index + 1, ADC_SampleTime_61Cycles5); /* XXX this is totally arbitrary... */
					adc_dev->channel_map[acc_index] = &adc_dev->accumulator[i];
					++current_index;
					acc_index += 2;
					if (current_index
					                == (adc_dev->number_used_master_channels > adc_dev->number_used_slave_channels ?
					                                adc_dev->number_used_master_channels : adc_dev->number_used_slave_channels)) {
						again = false;
						break;
					}
				}
			}
		}
	}
	ADC_Cmd(adc_dev->cfg->adc_dev_master, ENABLE);
	if (adc_dev->cfg->adc_dev_slave)
		ADC_Cmd(adc_dev->cfg->adc_dev_slave, ENABLE);
	while (!ADC_GetFlagStatus(adc_dev->cfg->adc_dev_master, ADC_FLAG_RDY))
		;
	if (adc_dev->cfg->adc_dev_slave)
		while (!ADC_GetFlagStatus(adc_dev->cfg->adc_dev_slave, ADC_FLAG_RDY))
			;
	ADC_StartConversion(adc_dev->cfg->adc_dev_master);
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
	adc_dev->number_used_master_channels = 0;
	adc_dev->number_used_slave_channels = 0;

	for (uint8_t i = 0; i < adc_dev->cfg->number_of_used_pins; ++i) {
		if (adc_dev->cfg->adc_pins[i].is_master_channel)
			++adc_dev->number_used_master_channels;
		else
			++adc_dev->number_used_slave_channels;
	}
	adc_dev->regular_group_size = adc_dev->number_used_master_channels > adc_dev->number_used_slave_channels ? adc_dev->number_used_master_channels : adc_dev->number_used_slave_channels;
	if (adc_dev->cfg->adc_dev_slave) {
		// DMA transfer size in units defined by DMA_PeripheralDataSize, 32bits for dual mode and 16bits for single mode
		adc_dev->dma_transfer_size = 2 * adc_dev->cfg->oversampling * adc_dev->regular_group_size;
		// DMA half buffer index (buffer is 16bit array),
		// one should think that the half buffer index was half the transfer size but need to remember that the first is 16bit and the second 32bit
		adc_dev->dma_half_buffer_index = adc_dev->dma_transfer_size;
		adc_dev->accumulator_increment = 2;
		adc_dev->accumulator_scan_size = adc_dev->regular_group_size * 2;
		adc_dev->raw_data_buffer = PIOS_malloc(adc_dev->dma_transfer_size * sizeof(uint32_t));
	} else {
		// DMA transfer size in units defined by DMA_PeripheralDataSize, 32bits for dual mode and 16bits for single mode
		adc_dev->dma_transfer_size = 2 * adc_dev->cfg->oversampling * adc_dev->cfg->number_of_used_pins;
		// DMA half buffer index (buffer is 16bit array),
		// half buffer index is half the transfer size because they are both 16bit based here
		adc_dev->dma_half_buffer_index = adc_dev->cfg->oversampling * adc_dev->cfg->number_of_used_pins;
		;
		adc_dev->accumulator_increment = 1;
		adc_dev->accumulator_scan_size = adc_dev->regular_group_size;
		adc_dev->raw_data_buffer = PIOS_malloc(adc_dev->dma_transfer_size * sizeof(uint16_t));
	}
	if (adc_dev->raw_data_buffer == NULL )
		return -1;

	adc_dev->accumulator = PIOS_malloc(adc_dev->cfg->number_of_used_pins * sizeof(struct adc_accumulator));
	if (adc_dev->accumulator == NULL )
		return -1;
	if (adc_dev->cfg->adc_dev_slave)
		adc_dev->channel_map = PIOS_malloc(adc_dev->regular_group_size * 2 * sizeof(struct adc_accumulator *));
	else
		adc_dev->channel_map = PIOS_malloc(adc_dev->regular_group_size * sizeof(struct adc_accumulator *));
	if (adc_dev->channel_map == NULL )
		return -1;

	driver_instances[current_instances] = adc_dev;
	++current_instances;

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
	int32_t result;
	/* Check if pin exists */
	if (pin >= adc_dev->cfg->number_of_used_pins) {
		return -1;
	}
	result = adc_dev->accumulator[pin].accumulator / (adc_dev->accumulator[pin].count ? : 1);
	adc_dev->accumulator[pin].accumulator = 0;
	adc_dev->accumulator[pin].count = 0;
	return result;
}

/**
 * @brief accumulate the data for each of the channels.
 */
static void accumulate(struct pios_internal_adc_dev *adc_dev, uint16_t *buffer)
{
	uint16_t *sp;
	/*
	 * Accumulate sampled values.
	 */
	for (uint32_t i = 0; i < adc_dev->cfg->oversampling; ++i) {
		sp = buffer + adc_dev->regular_group_size * i * adc_dev->accumulator_increment;
		for (uint8_t scan_index = 0; scan_index < adc_dev->accumulator_scan_size; ++scan_index) {
			adc_dev->channel_map[scan_index]->accumulator += *sp;
			adc_dev->channel_map[scan_index]->count++;
			sp++;
			/*
			 * If the accumulator reaches half-full, rescale in order to
			 * make more space.
			 */
			if (adc_dev->channel_map[scan_index]->accumulator >= (1 << 31)) {
				adc_dev->channel_map[scan_index]->accumulator /= 2;
				adc_dev->channel_map[scan_index]->count /= 2;
			}

		}
	}
}

/**
 * @brief DMA Interrupt handlers
 */
static void PIOS_INTERNAL_ADC_DMA_Handler1(void)
{
	PIOS_ADC_DMA_Handler(driver_instances[0]);
}

static void PIOS_INTERNAL_ADC_DMA_Handler2(void)
{
	PIOS_ADC_DMA_Handler(driver_instances[1]);
}

static void PIOS_INTERNAL_ADC_DMA_Handler3(void)
{
	PIOS_ADC_DMA_Handler(driver_instances[2]);
}

static void PIOS_INTERNAL_ADC_DMA_Handler4(void)
{
	PIOS_ADC_DMA_Handler(driver_instances[3]);
}

/**
 * @brief Interrupt on buffer flip.
 *
 * The hardware is done with the 'other' buffer, so we can pass it to the accumulator.
 */
static void PIOS_ADC_DMA_Handler(struct pios_internal_adc_dev *adc_dev)
{
	if (!PIOS_INTERNAL_ADC_validate(adc_dev))
		PIOS_Assert(0);
	/* terminal count, buffer has flipped */
	if (DMA_GetFlagStatus(adc_dev->cfg->full_flag)) { // whole double buffer filled
		DMA_ClearFlag(adc_dev->cfg->full_flag);
		accumulate(adc_dev, adc_dev->raw_data_buffer + adc_dev->dma_half_buffer_index);
	} else if (DMA_GetFlagStatus(adc_dev->cfg->half_flag /*DMA1_IT_HT1*/)) {
		DMA_ClearFlag(adc_dev->cfg->half_flag);
		accumulate(adc_dev, adc_dev->raw_data_buffer);
	} else {
		// This should not happen, probably due to transfer errors
		DMA_ClearFlag(adc_dev->cfg->dma.irq.flags /*DMA1_FLAG_GL1*/);
	}
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
