/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_PCF8591_ADC PCF8591 ADC Functions
 * @{
 *
 * @file       pios_pcf8591_adc.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      PCF8591 ADC driver
 * @see        The GNU Public License (GPL) Version 3
 *
 ******************************************************************************
 */
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
#include <pios_pcf8591_adc_priv.h>

#if defined(PIOS_INCLUDE_PCF8591)

#define PIOS_PCF8591_DAC_ENABLE                                                 0x40
#define PIOS_PCF8591_ADC_CH0                                                    0x00
#define PIOS_PCF8591_ADC_CH1                                                    0x01
#define PIOS_PCF8591_ADC_CH2                                                    0x02
#define PIOS_PCF8591_ADC_CH3                                                    0x03
#define PIOS_PCF8591_ADC_AUTO_INCREMENT                                 0x04
#define PIOS_PCF8591_NUMBER_OF_ADC_CHANNELS                             4
#define PIOS_PCF8591_CHANNELS { PIOS_PCF8591_ADC_CH0, PIOS_PCF8591_ADC_CH1, PIOS_PCF8591_ADC_CH2, PIOS_PCF8591_ADC_CH3 }

// Private functions
static int32_t PIOS_PCF8591_ADC_DevicePinGet(uint32_t adc_id, uint32_t device_pin);
static bool PIOS_PCF8591_ADC_Available(uint32_t adc_id, uint32_t device_pin);
static struct pios_pcf8591_adc_dev * PIOS_PCF8591_ADC_Allocate(void);
static uint8_t PIOS_PCF8591_ADC_Number_of_Channels(uint32_t internal_adc_id);
static bool PIOS_PCF8591_ADC_validate(struct pios_pcf8591_adc_dev *);
static float PIOS_INTERNAL_ADC_LSB_Voltage(uint32_t);

// Private types
enum pios_pcf8591_adc_dev_magic {
        PIOS_PCF8591_ADC_DEV_MAGIC = 0x58376969,
};

const struct pios_adc_driver pios_pcf8591_adc_driver = {
        .available = PIOS_PCF8591_ADC_Available,
        .get_pin = PIOS_PCF8591_ADC_DevicePinGet,
        .number_of_channels = PIOS_PCF8591_ADC_Number_of_Channels,
        .full_range_value = PIOS_INTERNAL_ADC_LSB_Voltage,
};

struct pios_pcf8591_adc_dev {
        const struct pios_pcf8591_adc_cfg * cfg;
        enum pios_pcf8591_adc_dev_magic magic;
};

/* Local Variables */
#define I2C_BUFFER_SIZE	2
static const uint8_t ADC_CHANNEL[] = PIOS_PCF8591_CHANNELS;

/**
 * @brief Validates the ADC device
 * \param[in] dev pointer to device structure
 * \return true if valid
 */
static bool PIOS_PCF8591_ADC_validate(struct pios_pcf8591_adc_dev *dev)
{
        if (dev == NULL)
                return false;

        return (dev->magic == PIOS_PCF8591_ADC_DEV_MAGIC);
}

/**
 * Returns value of an ADC Pin
 * \param[in] device_pin pin number
 * \param[in] adc_id handle to the device
 *
 * \return ADC pin value
 * \return -1 if pin doesn't exist
 */
static int32_t PIOS_PCF8591_ADC_DevicePinGet(uint32_t adc_id, uint32_t device_pin) {
        struct pios_pcf8591_adc_dev * adc_dev = (struct pios_pcf8591_adc_dev *) adc_id;
        if(!PIOS_PCF8591_ADC_validate(adc_dev))
        {
                return -1;
        }
        /* Check if pin exists */
        if (device_pin >= NELEMENTS(ADC_CHANNEL)) {
                return -1;
        }
        uint8_t readBuffer[2];
        uint8_t config;
        config = ADC_CHANNEL[device_pin] | (adc_dev->cfg->use_auto_increment ? PIOS_PCF8591_ADC_AUTO_INCREMENT : 0x00) | adc_dev->cfg->adc_input_type | (adc_dev->cfg->enable_dac ? PIOS_PCF8591_DAC_ENABLE : 0x00);

        struct pios_i2c_txn txn_list[] = {
                {
                        .info = __func__,
                        .addr = (adc_dev->cfg->i2c_address >> 1),
                        .rw = PIOS_I2C_TXN_WRITE,
                        .len = sizeof(config),
                        .buf = &config,
                },
        };
        PIOS_I2C_Transfer(PIOS_I2C_PCF8591_ADAPTER, txn_list, NELEMENTS(txn_list));
        txn_list[0].rw = PIOS_I2C_TXN_READ;
        txn_list[0].len = sizeof(readBuffer);
        txn_list[0].buf = readBuffer;
        PIOS_I2C_Transfer(PIOS_I2C_PCF8591_ADAPTER, txn_list, NELEMENTS(txn_list));
        return readBuffer[1];
}

/**
 * @brief Checks if a given pin is available on the given device
 * \param[in] adc_id handle of the device to read
 * \param[in] device_pin pin to check if available
 * \return true if available
 */
static bool PIOS_PCF8591_ADC_Available(uint32_t adc_id, uint32_t device_pin) {
        /* Check if pin exists */
	return (device_pin < NELEMENTS(ADC_CHANNEL));
}

/**
 * @brief Initializes an PCF8591 ADC device
 * \param[out] pcf8591_adc_id handle to the device
 * \param[in] cfg device configuration
 * \return < 0 if deviced initialization failed
 */
int32_t PIOS_PCF8591_ADC_Init(uint32_t *pcf8591_adc_id, const struct pios_pcf8591_adc_cfg *cfg)
{
        PIOS_DEBUG_Assert(pcf8591_adc_id);
        PIOS_DEBUG_Assert(cfg);

        struct pios_pcf8591_adc_dev *adc_dev;
        adc_dev = PIOS_PCF8591_ADC_Allocate();

        if (adc_dev == NULL)
                return -1;

        adc_dev->cfg = cfg;
        *pcf8591_adc_id = (uint32_t)adc_dev;
        return 0;
}

/**
 * @brief Allocates an internal ADC device in memory
 * \param[out] pointer to the newly created device
 *
 */
static struct pios_pcf8591_adc_dev * PIOS_PCF8591_ADC_Allocate()
{
        struct pios_pcf8591_adc_dev *adc_dev;

        adc_dev = (struct pios_pcf8591_adc_dev *)PIOS_malloc(sizeof(*adc_dev));
        if (!adc_dev) return (NULL);
        adc_dev->magic = PIOS_PCF8591_ADC_DEV_MAGIC;
        return(adc_dev);
}

/**
 * @brief Checks the number of available ADC channels on the device
 * \param[in] adc_id handle of the device
 * \return number of ADC channels of the device
 */
static uint8_t PIOS_PCF8591_ADC_Number_of_Channels(uint32_t adc_id)
{
        struct pios_pcf8591_adc_dev *adc_dev = (struct pios_pcf8591_adc_dev *)adc_id;
        if(!PIOS_PCF8591_ADC_validate(adc_dev)) {
                return 0;
        }
        return NELEMENTS(ADC_CHANNEL);
}

/**
 * @brief Gets the least significant bit voltage of the ADC
 */
static float PIOS_INTERNAL_ADC_LSB_Voltage(uint32_t internal_adc_id)
{
        struct pios_pcf8591_adc_dev *adc_dev = (struct pios_pcf8591_adc_dev *)adc_id;
        if(!PIOS_PCF8591_ADC_validate(adc_dev)) {
                return 0;
        }
        return VREF_PLUS / (((uint32_t)1 << 8) - 1);
}
#endif

/**
 * @}
 * @}
 */
