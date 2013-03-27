/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_ADC ADC Functions
 * @brief STM32F4xx ADC PIOS interface
 * @{
 *
 * @file       pios_adc.c  
 * @author     Michael Smith Copyright (C) 2011.
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @brief      Analog to Digital converstion routines
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

/*
 * @note This is a stripped-down ADC driver intended primarily for sampling
 * voltage and current values.  Samples are averaged over the period between
 * fetches so that relatively accurate measurements can be obtained without
 * forcing higher-level logic to poll aggressively.
 *
 * @todo This module needs more work to be more generally useful.  It should
 * almost certainly grow callback support so that e.g. voltage and current readings
 * can be shipped out for coulomb counting purposes.  The F1xx interface presumes
 * use with analog sensors, but that implementation largely dominates the ADC
 * resources.  Rather than commit to a new API without a defined use case, we
 * should stick to our lightweight subset until we have a better idea of what's needed.
 */

#include "pios.h"
#include <pios_internal_adc_priv.h>

static struct pios_internal_adc_dev * PIOS_INTERNAL_ADC_Allocate();
static bool PIOS_INTERNAL_ADC_validate(struct pios_internal_adc_dev *);
static void PIOS_INTERNAL_ADC_PinConfig(uint32_t internal_adc_id);
static void PIOS_INTERNAL_DMAConfig(uint32_t internal_adc_id);
static void PIOS_ADC_DMA_Handler(struct pios_internal_adc_dev *);
int32_t PIOS_INTERNAL_ADC_Init(uint32_t *internal_adc_id, const struct pios_internal_adc_cfg *cfg);
static void PIOS_INTERNAL_ADC_Converter_Config(uint32_t internal_adc_id);

void PIOS_INTERNAL_ADC_DMA_Handler1(void);
void PIOS_INTERNAL_ADC_DMA_Handler2(void);
void PIOS_INTERNAL_ADC_DMA_Handler3(void);
void PIOS_INTERNAL_ADC_DMA_Handler4(void);

static void test(void);
static void test2(void);

//void DMA1_Channel1_IRQHandler() __attribute__ ((alias ("test")));//TODO
//void ADC1_2_IRQHandler() __attribute__ ((alias ("test2")));//TODO
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
        ADCCallback callback_function;
#if defined(PIOS_INCLUDE_FREERTOS)
        xQueueHandle data_queue;
#endif
        // volatile int16_t *valid_data_buffer;//TODO
        struct adc_accumulator **channel_map;
        struct adc_accumulator *master_accumulator;
        struct adc_accumulator *slave_accumulator;
        volatile uint8_t adc_oversample;
        uint8_t dma_block_size;
        uint16_t dma_half_buffer_size;
        volatile uint16_t *raw_data_buffer;   // Double buffer that DMA just used
        enum pios_internal_adc_dev_magic magic;
};
static void test(void)//TODO
{
        PIOS_LED_Toggle(PIOS_LED_GREEN_E);
        PIOS_LED_Toggle(PIOS_LED_GREEN_W);
        PIOS_LED_Toggle(PIOS_LED_ORANGE_NE);

}
static void test2(void)//TODO
{
        PIOS_LED_Toggle(PIOS_LED_GREEN_E);
        PIOS_LED_Toggle(PIOS_LED_GREEN_W);
        PIOS_LED_Toggle(PIOS_LED_ORANGE_NE);
        volatile uint16_t result=ADC_GetConversionValue(ADC1);
        ++result;
        ADC_ClearFlag(ADC1, ADC_FLAG_EOC);

}
static bool PIOS_INTERNAL_ADC_validate(struct pios_internal_adc_dev * dev)
{
        if (dev == NULL )
                return false;

        return (dev->magic == PIOS_INTERNAL_ADC_DEV_MAGIC);
}
#if defined(PIOS_INCLUDE_ADC)
static struct pios_internal_adc_dev * PIOS_INTERNAL_ADC_Allocate()
{
        struct pios_internal_adc_dev *adc_dev = (struct pios_internal_adc_dev *) pvPortMalloc(sizeof(*adc_dev));
        if (!adc_dev)
                return (NULL );
        adc_dev->magic = PIOS_INTERNAL_ADC_DEV_MAGIC;
        return (adc_dev);
}
#endif
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
        }
        else
                DMAInit.DMA_PeripheralBaseAddr = (uint32_t) &adc_dev->cfg->adc_dev_master->DR;
        DMAInit.DMA_MemoryBaseAddr = (uint32_t) adc_dev->raw_data_buffer;
        DMAInit.DMA_BufferSize = adc_dev->dma_half_buffer_size;
        DMAInit.DMA_DIR = DMA_DIR_PeripheralSRC;
        DMAInit.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
        DMAInit.DMA_MemoryInc = DMA_MemoryInc_Enable;
        DMAInit.DMA_PeripheralDataSize = DMA_PeripheralDataSize_HalfWord;
        DMAInit.DMA_MemoryDataSize = DMA_MemoryDataSize_HalfWord;
        DMAInit.DMA_Mode = DMA_Mode_Circular;

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

static void PIOS_INTERNAL_ADC_Converter_Config(uint32_t internal_adc_id)
{
        PIOS_LED_Toggle(PIOS_LED_GREEN_E);
        struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *) internal_adc_id;

        ADC_DeInit(adc_dev->cfg->adc_dev_master);
        if (adc_dev->cfg->adc_dev_slave)
                ADC_DeInit(adc_dev->cfg->adc_dev_slave);

        RCC_ADCCLKConfig(RCC_ADC12PLLCLK_Div2);//TODO
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

        /* Do common ADC init */
        ADC_CommonInitTypeDef ADC_CommonInitStructure;
        ADC_CommonStructInit(&ADC_CommonInitStructure);
        if (adc_dev->cfg->adc_dev_slave) {
                ADC_CommonInitStructure.ADC_Mode = ADC_Mode_RegSimul;
                ADC_CommonInitStructure.ADC_DMAAccessMode = ADC_DMAAccessMode_1;
        }
        else {
                ADC_CommonInitStructure.ADC_Mode = ADC_Mode_Independent;
                ADC_CommonInitStructure.ADC_DMAAccessMode = ADC_DMAAccessMode_Disabled;
        }

        ADC_CommonInitStructure.ADC_Clock = ADC_Clock_AsynClkMode;   //TODO
        ADC_CommonInitStructure.ADC_DMAMode = ADC_DMAMode_Circular;
        ADC_CommonInitStructure.ADC_TwoSamplingDelay = 0;
        ADC_DMAConfig(adc_dev->cfg->adc_dev_master,ADC_DMAMode_Circular);
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

        /* Configure input scan */
        uint8_t master_index = 0;
        uint8_t slave_index = 0;
        for (int32_t i = 0; i < adc_dev->cfg->number_of_used_pins; i++) {
                if (adc_dev->cfg->adc_pins[i].is_master_channel) {
                        ADC_RegularChannelConfig(adc_dev->cfg->adc_dev_master, adc_dev->cfg->adc_pins[i].adc_channel,
                                        master_index + 1, ADC_SampleTime_61Cycles5); /* XXX this is totally arbitrary... */
                        ++master_index;
                }
                else {
                        ADC_RegularChannelConfig(adc_dev->cfg->adc_dev_slave, adc_dev->cfg->adc_pins[i].adc_channel,
                                        slave_index + 1, ADC_SampleTime_61Cycles5); /* XXX this is totally arbitrary... */
                        ++slave_index;
                }
        }
      //  ADC_ITConfig(ADC1,ADC_FLAG_EOC,ENABLE);//TODO

        ADC_Cmd(adc_dev->cfg->adc_dev_master, ENABLE);

        while (!ADC_GetFlagStatus(ADC1, ADC_FLAG_RDY));
        ADC_StartConversion(adc_dev->cfg->adc_dev_master);

//        NVIC_InitTypeDef NVICInit;
//        NVICInit.NVIC_IRQChannel = ADC1_2_IRQn;
//        NVICInit.NVIC_IRQChannelSubPriority = 0;
//        NVICInit.NVIC_IRQChannelCmd = ENABLE;
//        NVIC_Init(&NVICInit);
      //  ADC_ITConfig(ADC1,ADC_FLAG_EOS,ENABLE);
        //TODO
}

/**
 * @brief Init the ADC.
 */
int32_t PIOS_INTERNAL_ADC_Init(uint32_t * internal_adc_id, const struct pios_internal_adc_cfg * cfg)
{

//#if defined(PIOS_INCLUDE_ADC)
        PIOS_DEBUG_Assert(internal_adc_id);PIOS_DEBUG_Assert(cfg);

        struct pios_internal_adc_dev * adc_dev;
        adc_dev = PIOS_INTERNAL_ADC_Allocate();
        if (adc_dev == NULL )
                return -1;
        adc_dev->cfg = cfg;
        adc_dev->callback_function = NULL;

        *internal_adc_id = (uint32_t) adc_dev;
        adc_dev->number_used_master_channels = 0;
        adc_dev->number_used_slave_channels = 0;

        for (uint8_t i = 0; i < adc_dev->cfg->number_of_used_pins; ++i) {
                if (adc_dev->cfg->adc_pins[i].is_master_channel)
                        ++adc_dev->number_used_master_channels;
                else
                        ++adc_dev->number_used_slave_channels;
        }
        adc_dev->regular_group_size =
                        adc_dev->number_used_master_channels > adc_dev->number_used_slave_channels ?
                                        adc_dev->number_used_master_channels : adc_dev->number_used_slave_channels;
        if (adc_dev->cfg->adc_dev_slave) {
                adc_dev->raw_data_buffer = pvPortMalloc(
                                2 * adc_dev->cfg->oversampling * adc_dev->regular_group_size * 2 * sizeof(uint16_t));
                adc_dev->dma_half_buffer_size = adc_dev->cfg->oversampling * adc_dev->regular_group_size * 2
                                * sizeof(uint16_t);
        }
        else {
                // 2 * because double buffering
                adc_dev->raw_data_buffer = pvPortMalloc(
                2 * adc_dev->cfg->oversampling * adc_dev->cfg->number_of_used_pins * sizeof(uint16_t));
                adc_dev->dma_half_buffer_size = adc_dev->cfg->oversampling * adc_dev->cfg->number_of_used_pins
                                * sizeof(uint16_t);
        }
        if (adc_dev->raw_data_buffer == NULL )
                return -1;

        adc_dev->master_accumulator = pvPortMalloc(
                        adc_dev->number_used_master_channels * sizeof(struct adc_accumulator));
        if (adc_dev->master_accumulator == NULL )
                return -1;
        if (adc_dev->cfg->adc_dev_slave) {
                adc_dev->slave_accumulator = pvPortMalloc(
                                adc_dev->number_used_slave_channels * sizeof(struct adc_accumulator));
                if (adc_dev->slave_accumulator == NULL )
                        return -1;
        }

        adc_dev->channel_map = pvPortMalloc(adc_dev->cfg->number_of_used_pins * sizeof(struct adc_accumulator *));
        if (adc_dev->channel_map == NULL )
                return -1;
        uint8_t m_index = 0;
        uint8_t s_index = 0;

        for (uint8_t i = 0; i < adc_dev->cfg->number_of_used_pins; ++i) {
                if (adc_dev->cfg->adc_pins[i].is_master_channel) {
                        adc_dev->channel_map[i] = &adc_dev->master_accumulator[m_index++];
                }
                else
                        adc_dev->channel_map[i] = &adc_dev->slave_accumulator[s_index++];
        }

        driver_instances[current_instances] = adc_dev;
        ++current_instances;

        PIOS_INTERNAL_ADC_PinConfig((uint32_t) adc_dev);

        PIOS_INTERNAL_DMAConfig((uint32_t) adc_dev);

        PIOS_INTERNAL_ADC_Converter_Config((uint32_t) adc_dev);
        return 0;

//#endif

        return 0;
}

/**
 * @brief Configure the ADC to run at a fixed oversampling
 * @param[in] oversampling the amount of oversampling to run at
 */
void PIOS_INTERNAL_ADC_Config(uint32_t oversampling)
{
        /* we ignore this */
}

/**
 * Returns value of an ADC Pin
 * @param[in] pin number
 * @return ADC pin value averaged over the set of samples since the last reading.
 * @return -1 if pin doesn't exist
 */
int32_t PIOS_INTERNAL_ADC_PinGet(uint32_t internal_adc_id, uint32_t pin)
{
        struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *) internal_adc_id;
        int32_t result;
      //  return *(adc_dev->raw_data_buffer+1);
        /* Check if pin exists */
        if (pin >= adc_dev->cfg->number_of_used_pins) {
                return -1;
        }
        result = adc_dev->channel_map[pin]->accumulator / (adc_dev->channel_map[pin]->count ? : 1);
        adc_dev->channel_map[pin]->accumulator = 0;
        adc_dev->channel_map[pin]->count = 0;
        return result;
}

/**
 * @brief Set a callback function that is executed whenever
 * the ADC double buffer swaps 
 * @note Not currently supported.
 */
void PIOS_INTERNAL_ADC_SetCallback(ADCCallback new_function)
{
        // XXX might be nice to do something here
}

#if defined(PIOS_INCLUDE_FREERTOS)
/**
 * @brief Register a queue to add data to when downsampled 
 * @note Not currently supported.
 */
void PIOS_INTERNAL_ADC_SetQueue(xQueueHandle data_queue)
{
        // XXX it might make sense? to do this
}
#endif

/**
 * @brief Return the address of the downsampled data buffer
 * @note Not currently supported.
 */
float * PIOS_INTERNAL_ADC_GetBuffer(void)
{
        return NULL ;
}

/**
 * @brief Return the address of the raw data data buffer 
 * @note Not currently supported.
 */
int16_t * PIOS_INTERNAL_ADC_GetRawBuffer(void)
{
        return NULL ;
}

/**
 * @brief Return the amount of over sampling
 * @note Not currently supported (always returns 1)
 */
uint8_t PIOS_INTERNAL_ADC_GetOverSampling(void)
{
        return 1;
}

/**
 * @brief Set the fir coefficients.  Takes as many samples as the 
 * current filter order plus one (normalization)
 *
 * @param new_filter Array of adc_oversampling floats plus one for the
 * filter coefficients
 * @note Not currently supported.
 */
void PIOS_INTERNAL_ADC_SetFIRCoefficients(float * new_filter)
{
        // not implemented
}

/**
 * @brief accumulate the data for each of the channels.
 */
void accumulate(uint32_t internal_adc_id, volatile uint16_t *buffer)
{
        struct pios_internal_adc_dev * adc_dev = (struct pios_internal_adc_dev *) internal_adc_id;
        if (!PIOS_INTERNAL_ADC_validate(adc_dev)) {
                return;
        }

        volatile uint16_t *sp;
        /*
         * Accumulate sampled values.
         */
        uint8_t increment;
        ++increment;
        if (adc_dev->cfg->adc_dev_slave)
                increment = 2;
        else
                increment = 1;
        for (int i = 0; i < adc_dev->cfg->oversampling; ++i) {
                sp = buffer + adc_dev->regular_group_size * i * increment;
                for (int ii = 0; ii < adc_dev->number_used_master_channels; ii++) {
                        adc_dev->master_accumulator[ii].accumulator += *sp;
                        sp = sp + increment;
                        adc_dev->master_accumulator[ii].count++;
                        /*
                         * If the accumulator reaches half-full, rescale in order to
                         * make more space.
                         */
                        if (adc_dev->master_accumulator[ii].accumulator >= (1 << 31)) {
                                adc_dev->master_accumulator[ii].accumulator /= 2;
                                adc_dev->master_accumulator[ii].count /= 2;
                        }
                }

                if (adc_dev->cfg->adc_dev_slave) {
                        sp = buffer + adc_dev->regular_group_size * i * increment + 1;
                        for (int ii = 0; ii < adc_dev->number_used_slave_channels; ii++) {
                                //adc_dev->slave_accumulator[ii].accumulator += *sp;
                                sp = sp + increment;
                                adc_dev->slave_accumulator[ii].count++;
                                /*
                                 * If the accumulator reaches half-full, rescale in order to
                                 * make more space.
                                 */
                                if (adc_dev->slave_accumulator[ii].accumulator >= (1 << 31)) {
                                        adc_dev->slave_accumulator[ii].accumulator /= 2;
                                        adc_dev->slave_accumulator[ii].count /= 2;
                                }
                        }
                }
        }
}

/**
 * @brief Interrupt on buffer flip.
 *
 * The hardware is done with the 'other' buffer, so we can pass it to the accumulator.
 */
void PIOS_INTERNAL_ADC_DMA_Handler1(void)
{
       // test();
        PIOS_ADC_DMA_Handler(driver_instances[0]);
}

void PIOS_INTERNAL_ADC_DMA_Handler2(void)
{
        test();
        test2();
        PIOS_ADC_DMA_Handler(driver_instances[1]);
}

void PIOS_INTERNAL_ADC_DMA_Handler3(void)
{
        PIOS_ADC_DMA_Handler(driver_instances[2]);
}

void PIOS_INTERNAL_ADC_DMA_Handler4(void)
{
        PIOS_ADC_DMA_Handler(driver_instances[3]);
}

static void PIOS_ADC_DMA_Handler(struct pios_internal_adc_dev *adc_dev)
{
        /* terminal count, buffer has flipped */
        if (DMA_GetFlagStatus(adc_dev->cfg->full_flag /*DMA1_IT_TC1*/)) {        // whole double buffer filled
                if (adc_dev->cfg->adc_dev_slave)
                {
                        accumulate((uint32_t) adc_dev, (uint16_t *)((uint32_t)(adc_dev->raw_data_buffer) + (uint32_t)(adc_dev->cfg->oversampling * adc_dev->regular_group_size * 2
                                                                        * sizeof(uint16_t))));
                }
                else {
                        accumulate((uint32_t) adc_dev, adc_dev->raw_data_buffer+adc_dev->cfg->oversampling * adc_dev->cfg->number_of_used_pins);
                }
                DMA_ClearFlag(adc_dev->cfg->full_flag);
        }
        else if (DMA_GetFlagStatus(adc_dev->cfg->half_flag /*DMA1_IT_HT1*/)) {
                accumulate((uint32_t) adc_dev, adc_dev->raw_data_buffer);
                DMA_ClearFlag(adc_dev->cfg->half_flag);
        }
        else {
                // This should not happen, probably due to transfer errors
                DMA_ClearFlag(adc_dev->cfg->dma.irq.flags /*DMA1_FLAG_GL1*/);
        }
}
/** 
 * @}
 * @}
 */
