/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_ADC ADC Functions
 * @{
 *
 * @file       pios_adc.c
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

// Private types
enum pios_adc_dev_magic {
	PIOS_ADC_DEV_MAGIC = 0x58375169,
};

struct pios_adc_dev {
	enum pios_adc_dev_magic magic;
	uint32_t lower_id;
	const struct pios_adc_driver *driver;
};

#if defined(PIOS_INCLUDE_FREERTOS)
struct pios_adc_dev * pios_adc_dev;
#else
#error "PIOS_ADC only works with freeRTOS"
#endif

// Private functions
static struct pios_adc_dev *
PIOS_ADC_Allocate(void);
static bool
PIOS_ADC_validate(struct pios_adc_dev *);

/* Local Variables */
static struct sub_device_list_ {
	uint8_t number_of_devices;
	struct pios_adc_dev * sub_device_pointers[PIOS_ADC_SUB_DRIVER_MAX_INSTANCES];
} sub_device_list;

/**
 * @brief Validates the ADC device
 * \param[in] dev pointer to device structure
 * \return true if valid
 */
static bool PIOS_ADC_validate(struct pios_adc_dev *dev)
{
	if (dev == NULL )
		return false;

	return (dev->magic == PIOS_ADC_DEV_MAGIC);
}

/**
 * @brief Allocates an ADC device in memory
 * \param[out] pointer to the newly created device
 *
 */
static struct pios_adc_dev *PIOS_ADC_Allocate(void)
{
	struct pios_adc_dev *adc_dev;

	adc_dev = (struct pios_adc_dev *)PIOS_malloc(sizeof(*adc_dev));
	if (!adc_dev)
		return (NULL );

	adc_dev->magic = PIOS_ADC_DEV_MAGIC;
	return (adc_dev);
}

/**
 * @brief Initializes an ADC device
 * \param[out] adc_id handle to the device
 * \param[in] driver drive to use with the device
 * \param[in] lower_id handle to the lower level device
 * \return < 0 if deviced initialization failed
 */
int32_t PIOS_ADC_Init(uintptr_t *adc_id, const struct pios_adc_driver *driver, uint32_t lower_id)
{
	PIOS_Assert(adc_id);
	PIOS_Assert(driver);
	if (sub_device_list.number_of_devices >= PIOS_ADC_SUB_DRIVER_MAX_INSTANCES)
		return -1;
	struct pios_adc_dev * adc_dev;

	adc_dev = (struct pios_adc_dev *) PIOS_ADC_Allocate();

	if (!adc_dev)
		return -1;

	adc_dev->driver = driver;
	adc_dev->lower_id = lower_id;
	*adc_id = (uintptr_t) adc_dev;
	sub_device_list.sub_device_pointers[sub_device_list.number_of_devices] = adc_dev;
	sub_device_list.number_of_devices++;
	return 0;
}

/**
 * @brief Gets the ADC value of the given pin of the device
 * \param[in] adc_id pointer to the device to read from
 * \param[in] device_pin pin from device to be read
 * \return the value of the pin or -1 if error
 */
int32_t PIOS_ADC_DevicePinGet(uintptr_t adc_id, uint32_t device_pin)
{
	struct pios_adc_dev * adc_dev = (struct pios_adc_dev *) adc_id;

	if (!PIOS_ADC_validate(adc_dev)) {
		return -1;
	}
	if (adc_dev->driver->get_pin)
		return (adc_dev->driver->get_pin)(adc_dev->lower_id, device_pin);
	else
		return -1;
}

/**
 * @brief Checks if a given pin is available on the given device
 * \param[in] adc_id handle of the device to read
 * \param[in] device_pin pin to check if available
 * \return true if available
 */
bool PIOS_ADC_Available(uintptr_t adc_id, uint32_t device_pin)
{
	struct pios_adc_dev *adc_dev = (struct pios_adc_dev *) adc_id;

	if (!PIOS_ADC_validate(adc_dev)) {
		return false;
	}
	if (adc_dev->driver->available)
		return (adc_dev->driver->available)(adc_dev->lower_id, device_pin);
	else
		return false;
}

/**
 * @brief Set the queue to write to
 * \param[in] adc_id handle to the device
 * \param[in] data_queue handle to the queue to be used
 */
void PIOS_ADC_SetQueue(uintptr_t adc_id, xQueueHandle data_queue)
{
	struct pios_adc_dev *adc_dev = (struct pios_adc_dev *) adc_id;

	if (!PIOS_ADC_validate(adc_dev)) {
		return;
	}
	if (!adc_dev->driver->set_queue)
		return;
	(adc_dev->driver->set_queue)(adc_dev->lower_id, data_queue);
}

/**
 * @brief Reads from an ADC channel
 * this is an abstraction of the lower devices
 * channels are sequentially added from the lower devices available pins
 * Warning this function is not as efficient as directly getting the device channel
 * in the order of initialization
 * \param[in] channel channel to read from
 * \return value of the channel or -1 if error
 */
int32_t PIOS_ADC_GetChannelRaw(uint32_t channel)
{
	uint32_t offset = 0;
	for (uint8_t x = 0; x < sub_device_list.number_of_devices; ++x) {
		struct pios_adc_dev * adc_dev = sub_device_list.sub_device_pointers[x];
		if (!PIOS_ADC_validate(adc_dev)) {
			PIOS_DEBUG_Assert(0);
			continue;
		} else if (adc_dev->driver->number_of_channels) {
			uint32_t num_channels_for_this_device = adc_dev->driver->number_of_channels(adc_dev->lower_id);
			if (adc_dev->driver->get_pin && (channel < offset + num_channels_for_this_device)) {
				return (adc_dev->driver->get_pin)(adc_dev->lower_id, channel - offset);
			} else
				offset += num_channels_for_this_device;
		}
	}
	return -1;
}

/**
 * @brief Reads from an ADC channel and returns value in voltage seen at pin
 * this is an abstraction of the lower devices
 * channels are sequentially added from the lower devices available pins
 * Warning this function is not as efficient as directly getting the device channel
 * in the order of initialization
 * \param[in] channel channel to read from
 * \return value of the channel or -1 if error
 */
float PIOS_ADC_GetChannelVolt(uint32_t channel)
{
	uint32_t offset = 0;
	for (uint8_t x = 0; x < sub_device_list.number_of_devices; ++x) {
		struct pios_adc_dev * adc_dev = sub_device_list.sub_device_pointers[x];
		if (!PIOS_ADC_validate(adc_dev)) {
			PIOS_DEBUG_Assert(0);
			continue;
		} else if (adc_dev->driver->number_of_channels) {
			uint32_t num_channels_for_this_device = adc_dev->driver->number_of_channels(adc_dev->lower_id);
			if (adc_dev->driver->get_pin && (channel < offset + num_channels_for_this_device)) {
				return (float)((adc_dev->driver->get_pin)(adc_dev->lower_id, channel - offset)) * (float)(adc_dev->driver->lsb_voltage)(adc_dev->lower_id);
			} else
				offset += num_channels_for_this_device;
		}
	}
	return -1;
}
/**
 * @}
 * @}
 */
