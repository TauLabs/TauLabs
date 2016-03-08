/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MS5XXX MS5XXX Functions
 * @brief Hardware functions to deal with the altitude pressure sensor
 * @{
 *
 * @file       pios_ms5xxx.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      MS5XXX Pressure Sensor Routines
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

/* Project Includes */
#include "pios.h"

#if defined(PIOS_INCLUDE_MS5XXX) || defined(PIOS_INCLUDE_MS5XXX_SPI)

#include "pios_ms5xxx_priv.h"
#include "pios_semaphore.h"
#include "pios_thread.h"
#include "pios_queue.h"

#include "physical_constants.h"

/* Private constants */
#define MS5XXX_TASK_PRIORITY	PIOS_THREAD_PRIO_HIGHEST
#define MS5XXX_TASK_STACK_BYTES	512

/* Private Variables */

/* Private methods */
static int32_t PIOS_MS5XXX_Read(uint8_t address, uint8_t * buffer, uint8_t len);
static int32_t PIOS_MS5XXX_WriteCommand(uint8_t command);
static void PIOS_MS5XXX_Task(void *parameters);

/* Private types */
//! The valid hardware buses
enum PIOS_BUS_TYPE {
	PIOS_SPI_BUS,
	PIOS_I2C_BUS,
};

/* Local Types */

enum PIOS_MS5XXX_DEV_MAGIC {
	PIOS_MS5XXX_DEV_MAGIC = 0x1c50bcf2, // md5 of `PIOS_MS5XXX_DEV_MAGIC`
};

enum conversion_type {
	PRESSURE_CONV,
	TEMPERATURE_CONV
};

struct ms5xxx_dev {
	const struct pios_ms5xxx_cfg * cfg;
	enum PIOS_BUS_TYPE pios_bus_type; // Bus type can be either SPI or I2C
	uint32_t dev_id; // Valid for both SPI and I2C
	uint32_t slave_num; // Only used for SPI bus chip select
	uint8_t ms5xxx_i2c_addr; // Only used for I2C 7-bit address
	struct pios_thread *task;
	struct pios_queue *queue;

	int64_t pressure_unscaled;
	int64_t temperature_unscaled;
	uint16_t calibration[6];
	enum conversion_type current_conversion_type;
	enum PIOS_MS5XXX_DEV_MAGIC magic;

	struct pios_semaphore *busy;
};

static struct ms5xxx_dev *dev;

/**
 * @brief Allocate a new device
 */
static struct ms5xxx_dev * PIOS_MS5XXX_alloc(void)
{
	struct ms5xxx_dev *ms5xxx_dev;

	ms5xxx_dev = (struct ms5xxx_dev *)PIOS_malloc(sizeof(*ms5xxx_dev));
	if (!ms5xxx_dev) {
		return (NULL);
	}

	memset(ms5xxx_dev, 0, sizeof(*ms5xxx_dev));

	ms5xxx_dev->queue = PIOS_Queue_Create(1, sizeof(struct pios_sensor_baro_data));
	if (ms5xxx_dev->queue == NULL) {
		PIOS_free(ms5xxx_dev);
		return NULL;
	}

	ms5xxx_dev->magic = PIOS_MS5XXX_DEV_MAGIC;

	ms5xxx_dev->busy = PIOS_Semaphore_Create();
	PIOS_Assert(ms5xxx_dev->busy != NULL);

	return ms5xxx_dev;
}

/**
 * @brief Validate the handle to the device
 * @returns 0 for valid device or <0 otherwise
 */
static int32_t PIOS_MS5XXX_Validate(struct ms5xxx_dev *dev)
{
	if (dev == NULL)
		return -1;
	if (dev->magic != PIOS_MS5XXX_DEV_MAGIC)
		return -2;
	if (dev->dev_id == 0)
		return -3;
	return 0;
}


/**
 * @brief PIOS_MS5XXX_Init Initializes the MS5xxx chip
 * @param cfg Configuration structure. Note that it is not deep copied.
 * @return 0 on success. -1 if the allocation fails, -2 if the board
 * setup file does not define the model of chip, and -3 if the write
 * command fails.
 */
int32_t PIOS_MS5XXX_Init(const struct pios_ms5xxx_cfg *cfg)
{
	// Set the pointer to the configuration struct. Note that it is not deep copied.
	dev->cfg = cfg;

	// Check that a model of the chip was chosen
	if (cfg->pios_ms5xxx_model == 0) {
		return -2;
	}

	if (PIOS_MS5XXX_WriteCommand(MS5XXX_RESET) != 0) {
		return -3;
	}

	PIOS_DELAY_WaitmS(20);

	uint8_t data[2];

	/* Calibration parameters */
	for (int i = 0; i < NELEMENTS(dev->calibration); i++) {
		PIOS_MS5XXX_Read(MS5XXX_CALIB_ADDR + i * 2, data, 2);
		dev->calibration[i] = (data[0] << 8) | data[1];
	}

	PIOS_SENSORS_Register(PIOS_SENSOR_BARO, dev->queue);

	dev->task = PIOS_Thread_Create(
			PIOS_MS5XXX_Task, "pios_ms5xxx", MS5XXX_TASK_STACK_BYTES, NULL, MS5XXX_TASK_PRIORITY);
	PIOS_Assert(dev->task != NULL);

	return 0;
}

/**
 * Initialise the MS5XXX sensor
 */
int32_t PIOS_MS5XXX_SPI_Init(uint32_t spi_bus_id, uint32_t slave_num, const struct pios_ms5xxx_cfg *ms5xxx_cfg)
{
	// Allocate the memory
	dev = (struct ms5xxx_dev *)PIOS_MS5XXX_alloc();
	if (dev == NULL) {
		return -1;
	}

	// Do a couple SPI specific things
	dev->pios_bus_type = PIOS_SPI_BUS;
	dev->dev_id = spi_bus_id;
	dev->slave_num = slave_num;

	// Go on to the generic initialization
	return PIOS_MS5XXX_Init(ms5xxx_cfg);
}

int32_t PIOS_MS5XXX_I2C_Init(int32_t i2c_bus_id, enum MS5XXX_I2C_ADDRESS i2c_address, const struct pios_ms5xxx_cfg *ms5xxx_cfg)
{
	// Allocate the memory
	dev = (struct ms5xxx_dev *)PIOS_MS5XXX_alloc();
	if (dev == NULL) {
		return -1;
	}

	// Do a couple I2C specific things
	dev->pios_bus_type = PIOS_I2C_BUS;
	dev->dev_id = i2c_bus_id;
	dev->ms5xxx_i2c_addr = i2c_address;

	// Go on to the generic initialization
	return PIOS_MS5XXX_Init(ms5xxx_cfg);
}

/**
 * Claim the MS5XXX device semaphore.
 * \return 0 if no error
 * \return -1 if timeout before claiming semaphore
 */
static int32_t PIOS_MSXXX_ClaimDeviceSemaphore(void)
{
	PIOS_Assert(PIOS_MS5XXX_Validate(dev) == 0);

	return PIOS_Semaphore_Take(dev->busy, PIOS_SEMAPHORE_TIMEOUT_MAX) == true ? 0 : 1;
}

/**
 * Release the MS5XXX device semaphore.
 * \return 0 if no error
 */
static int32_t PIOS_MS5XXX_ReleaseDeviceSemaphore(void)
{
	PIOS_Assert(PIOS_MS5XXX_Validate(dev) == 0);

	return PIOS_Semaphore_Give(dev->busy) == true ? 0 : 1;
}

/**
 * @brief Claim the SPI bus for the baro communications and select this chip
 * @return 0 if successful, -1 for invalid device, -2 if unable to claim bus
 */
static int32_t PIOS_MS5XXX_ClaimSPIBus(void)
{
#if defined(PIOS_INCLUDE_SPI)
	if (PIOS_MS5XXX_Validate(dev) != 0) {
		return -1;
	} else if (PIOS_SPI_ClaimBus(dev->dev_id) != 0) {
		return -2;
	}

	PIOS_SPI_RC_PinSet(dev->dev_id, dev->slave_num, 0);

	return 0;
#else
	return -2;
#endif
}

/**
 * @brief Release the SPI bus for the baro communications and end the transaction
 * @return 0 if successful
 */
static int32_t PIOS_MS5XXX_ReleaseSPIBus(void)
{
#if defined(PIOS_INCLUDE_SPI)
	if (PIOS_MS5XXX_Validate(dev) != 0) {
		return -1;
	}

	PIOS_SPI_RC_PinSet(dev->dev_id, dev->slave_num, 1);

	return PIOS_SPI_ReleaseBus(dev->dev_id);
#else
	return -2;
#endif
}

/**
* Start the ADC conversion
* \param[in] PRESSURE_CONV or TEMPERATURE_CONV to select which measurement to make
* \return 0 for success, -1 for failure (conversion completed and not read)
*/
static int32_t PIOS_MS5XXX_StartADC(enum conversion_type type)
{
	if (PIOS_MS5XXX_Validate(dev) != 0)
		return -1;

	/* Start the conversion */
	switch (type) {
	case TEMPERATURE_CONV:
		while (PIOS_MS5XXX_WriteCommand(MS5XXX_TEMP_ADDR + dev->cfg->oversampling) != 0)
			continue;
		break;
	case PRESSURE_CONV:
		while (PIOS_MS5XXX_WriteCommand(MS5XXX_PRES_ADDR + dev->cfg->oversampling) != 0)
			continue;
		break;
	default:
		return -1;
	}

	dev->current_conversion_type = type;

	return 0;
}

/**
 * @brief Return the delay for the current osr
 */
static int32_t PIOS_MS5XXX_GetDelay()
{
	if (PIOS_MS5XXX_Validate(dev) != 0)
		return 100;

	switch(dev->cfg->oversampling) {
	case MS5XXX_OSR_256:
		return 2;
	case MS5XXX_OSR_512:
		return 2;
	case MS5XXX_OSR_1024:
		return 3;
	case MS5XXX_OSR_2048:
		return 5;
	case MS5XXX_OSR_4096:
		return 10;
	default:
		break;
	}
	return 10;
}

/**
* Read the ADC conversion value (once ADC conversion has completed)
* \return 0 if successfully read the ADC, -1 if failed
*/
static int32_t PIOS_MS5XXX_ReadADC(void)
{
	if (PIOS_MS5XXX_Validate(dev) != 0)
		return -1;

	uint8_t data[3];

	static int64_t delta_temp;
	static int64_t temperature;

	/* Read and store the 16bit result */
	if (dev->current_conversion_type == TEMPERATURE_CONV) {
		uint32_t raw_temperature;
		/* Read the temperature conversion */
		if (PIOS_MS5XXX_Read(MS5XXX_ADC_READ, data, 3) != 0)
			return -1;

		raw_temperature = (data[0] << 16) | (data[1] << 8) | data[2];

		delta_temp = (int32_t)raw_temperature - (dev->calibration[4] << 8);
		temperature = 2000 + ((delta_temp * dev->calibration[5]) >> 23);
		dev->temperature_unscaled = temperature;

		// second order temperature compensation
		if (temperature < 2000)
			dev->temperature_unscaled -= (delta_temp * delta_temp) >> 31;

	} else {
		int64_t offset;
		int64_t sens;
		uint32_t raw_pressure;

		/* Read the pressure conversion */
		if (PIOS_MS5XXX_Read(MS5XXX_ADC_READ, data, 3) != 0)
			return -1;

		raw_pressure = (data[0] << 16) | (data[1] << 8) | (data[2] << 0);

		offset = ((int64_t)dev->calibration[1] << 16) + (((int64_t)dev->calibration[3] * delta_temp) >> 7);
		sens = (int64_t)dev->calibration[0] << 15;
		sens = sens + ((((int64_t) dev->calibration[2]) * delta_temp) >> 8);

		// second order temperature compensation
		if (temperature < 2000) {
			offset -= (5 * (temperature - 2000) * (temperature - 2000)) >> 1;
			sens -= (5 * (temperature - 2000) * (temperature - 2000)) >> 2;

			if (dev->temperature_unscaled < -1500) {
				offset -= 7 * (temperature + 1500) * (temperature + 1500);
				sens -= (11 * (temperature + 1500) * (temperature + 1500)) >> 1;
			}
		}

		dev->pressure_unscaled = ((((int64_t)raw_pressure * sens) >> 21) - offset) >> 15;
	}
	return 0;
}

/**
* Reads one or more bytes into a buffer
* \param[in] the command indicating the address to read
* \param[out] buffer destination buffer
* \param[in] len number of bytes which should be read
* \return 0 if operation was successful
* \return -1 if dev is invalid
* \return -2 if error during I2C transfer
*/
static int32_t PIOS_MS5XXX_Read(uint8_t address, uint8_t *buffer, uint8_t len)
{
	if (PIOS_MS5XXX_Validate(dev) != 0) {
		return -1;
	}

	int32_t rc = 0;

	switch (dev->pios_bus_type) {
	case PIOS_SPI_BUS:
	{
		if (PIOS_MS5XXX_ClaimSPIBus() != 0) {
			return -2;
		}

		if (PIOS_SPI_TransferByte(dev->dev_id, address) < 0) {
			rc = -3;
			goto out;
		}

		if (PIOS_SPI_TransferBlock(dev->dev_id, NULL, buffer, len, NULL) < 0) {
			rc = -3;
			goto out;
		}

	out:
		PIOS_MS5XXX_ReleaseSPIBus();
		break;
	}
	case PIOS_I2C_BUS:
	{
		const struct pios_i2c_txn txn_list[] = {
			{
				.info = __func__,
					  .addr = dev->ms5xxx_i2c_addr,
					  .rw = PIOS_I2C_TXN_WRITE,
					  .len = 1,
					  .buf = &address,
			},
			{
				.info = __func__,
					  .addr = dev->ms5xxx_i2c_addr,
					  .rw = PIOS_I2C_TXN_READ,
					  .len = len,
					  .buf = buffer,
			}
		};

		rc = PIOS_I2C_Transfer(dev->dev_id, txn_list, NELEMENTS(txn_list));
		break;
	}
	}

	return rc;
}

/**
* Writes one or more bytes to the MS5XXX
* \param[in] address Register address
* \param[in] buffer source buffer
* \return 0 if operation was successful
* \return -1 if dev is invalid
* \return -2 if failed to claim SPI bus
* \return -3 if error during transfer
*/
static int32_t PIOS_MS5XXX_WriteCommand(uint8_t command)
{
	if (PIOS_MS5XXX_Validate(dev) != 0) {
		return -1;
	}

	int32_t rc = 0;

	switch (dev->pios_bus_type) {
	case PIOS_SPI_BUS:
	{
		if (PIOS_MS5XXX_ClaimSPIBus() != 0) {
			return -2;
		}

		if (PIOS_SPI_TransferByte(dev->dev_id, command) < 0) {
			rc = -3;
			goto out;
		}

	out:
		PIOS_MS5XXX_ReleaseSPIBus();
		break;
	}
	case PIOS_I2C_BUS:
	{
		const struct pios_i2c_txn txn_list[] = {
			{
				.info = __func__,
				.addr = dev->ms5xxx_i2c_addr,
				.rw = PIOS_I2C_TXN_WRITE,
				.len = 1,
				.buf = &command,
			 },
		};

		rc = PIOS_I2C_Transfer(dev->dev_id, txn_list, NELEMENTS(txn_list));
		break;
	}
	}

	return rc;
}

/**
* @brief Run self-test operation.
* \return 0 if self-test succeed, -1 if failed
*/
int32_t PIOS_MS5XXX_Test()
{
	if (PIOS_MS5XXX_Validate(dev) != 0) {
		return -1;
	}

	PIOS_MSXXX_ClaimDeviceSemaphore();
	PIOS_MS5XXX_StartADC(TEMPERATURE_CONV);
	PIOS_DELAY_WaitmS(PIOS_MS5XXX_GetDelay());
	PIOS_MS5XXX_ReadADC();
	PIOS_MS5XXX_ReleaseDeviceSemaphore();

	PIOS_MSXXX_ClaimDeviceSemaphore();
	PIOS_MS5XXX_StartADC(PRESSURE_CONV);
	PIOS_DELAY_WaitmS(PIOS_MS5XXX_GetDelay());
	PIOS_MS5XXX_ReadADC();
	PIOS_MS5XXX_ReleaseDeviceSemaphore();

	// check range for sanity according to datasheet
	if (dev->temperature_unscaled < -4000 ||
		dev->temperature_unscaled > 8500 ||
		dev->pressure_unscaled < 1000 ||
		dev->pressure_unscaled > 120000)
		return -1;

	return 0;
}

static void PIOS_MS5XXX_Task(void *parameters)
{
	// init this to 1 in order to force a temperature read on the first run
	uint32_t temp_press_interleave_count = 1;
	int32_t  read_adc_result = 0;

	while (1) {

		--temp_press_interleave_count;
		read_adc_result = 0;

		if (temp_press_interleave_count == 0)
		{
			// Update the temperature data
			PIOS_MSXXX_ClaimDeviceSemaphore();
			PIOS_MS5XXX_StartADC(TEMPERATURE_CONV);
			PIOS_Thread_Sleep(PIOS_MS5XXX_GetDelay());
			read_adc_result = PIOS_MS5XXX_ReadADC();
			PIOS_MS5XXX_ReleaseDeviceSemaphore();

			temp_press_interleave_count = dev->cfg->temperature_interleaving;
			if (temp_press_interleave_count == 0)
				temp_press_interleave_count = 1;
		}

		// Update the pressure data
		PIOS_MSXXX_ClaimDeviceSemaphore();
		PIOS_MS5XXX_StartADC(PRESSURE_CONV);
		PIOS_Thread_Sleep(PIOS_MS5XXX_GetDelay());
		read_adc_result = PIOS_MS5XXX_ReadADC();
		PIOS_MS5XXX_ReleaseDeviceSemaphore();

		// Compute the altitude from the pressure and temperature and send it out
		struct pios_sensor_baro_data data;
		data.temperature = ((float) dev->temperature_unscaled) / 100.0f;
		data.pressure = ((float) dev->pressure_unscaled) / 1000.0f;
		data.altitude = 44330.0f * (1.0f - powf(data.pressure / (STANDARD_AIR_SEA_LEVEL_PRESSURE/1000.0f), (1.0f / 5.255f)));

		if (read_adc_result == 0) {
			PIOS_Queue_Send(dev->queue, &data, 0);
		}
	}
}


#endif

/**
 * @}
 * @}
 */
