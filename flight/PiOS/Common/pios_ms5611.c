/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MS5611 MS5611 Functions
 * @brief Hardware functions to deal with the altitude pressure sensor
 * @{
 *
 * @file       pios_ms5611.c  
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     PhoenixPilot, http://github.com/PhoenixPilot Copyright (C) 2012.
 * @brief      MS5611 Pressure Sensor Routines
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

#if defined(PIOS_INCLUDE_MS5611)

#include "pios_ms5611_priv.h"

/* Private constants */
#define PIOS_MS5611_OVERSAMPLING oversampling
#define MS5611_TASK_PRIORITY	(tskIDLE_PRIORITY + configMAX_PRIORITIES - 1)	// max priority
#define MS5611_TASK_STACK		(512 / 4)

/* Private methods */
static int32_t PIOS_MS5611_Read(uint8_t address, uint8_t * buffer, uint8_t len);
static int32_t PIOS_MS5611_WriteCommand(uint8_t command);
static void PIOS_MS5611_Task(void *parameters);

/* Private types */

/* Local Types */

enum pios_ms5611_dev_magic {
	PIOS_MS5611_DEV_MAGIC = 0xefba8e1d,
};

enum conversion_type {
	PRESSURE_CONV,
	TEMPERATURE_CONV
};

struct ms5611_dev {
	const struct pios_ms5611_cfg * cfg;
	uint32_t i2c_id;
	xTaskHandle task;
	xQueueHandle queue;

	int64_t pressure_unscaled;
	int64_t temperature_unscaled;
	uint16_t calibration[6];
	enum conversion_type current_conversion_type;
	enum pios_ms5611_osr oversampling;
	uint32_t temperature_interleaving;
	int32_t ms5611_read_flag;
	enum pios_ms5611_dev_magic magic;
};

struct ms5611_dev *dev;

/**
 * @brief Allocate a new device
 */
static struct ms5611_dev * PIOS_MS5611_alloc(void)
{
	struct ms5611_dev *ms5611_dev;
	
	ms5611_dev = (struct ms5611_dev *)pvPortMalloc(sizeof(*ms5611_dev));
	if (!ms5611_dev) return (NULL);
	
	ms5611_dev->magic = PIOS_MS5611_DEV_MAGIC;
	
	ms5611_dev->queue = xQueueCreate(1, sizeof(struct pios_ms5611_data));
	if (ms5611_dev->queue == NULL) {
		vPortFree(ms5611_dev);
		return NULL;
	}

	return(ms5611_dev);
}

/**
 * @brief Validate the handle to the i2c device
 * @returns 0 for valid device or -1 otherwise
 */
static int32_t PIOS_MS5611_Validate(struct ms5611_dev *dev)
{
	if (dev == NULL) 
		return -1;
	if (dev->magic != PIOS_MS5611_DEV_MAGIC)
		return -2;
	if (dev->i2c_id == 0)
		return -3;
	return 0;
}

/**
 * Initialise the MS5611 sensor
 */
int32_t PIOS_MS5611_Init(const struct pios_ms5611_cfg *cfg, int32_t i2c_device)
{
	dev = (struct ms5611_dev *) PIOS_MS5611_alloc();
	if (dev == NULL)
		return -1;

	dev->i2c_id = i2c_device;

	dev->oversampling = cfg->oversampling;
	dev->temperature_interleaving = (cfg->temperature_interleaving) == 0 ? 1 : cfg->temperature_interleaving;
	dev->cfg = cfg;

	PIOS_MS5611_WriteCommand(MS5611_RESET);
	PIOS_DELAY_WaitmS(20);			

	uint8_t data[2];

	/* Calibration parameters */
	for (int i = 0; i < NELEMENTS(dev->calibration); i++) {
		PIOS_MS5611_Read(MS5611_CALIB_ADDR + i * 2, data, 2);
		dev->calibration[i] = (data[0] << 8) | data[1];
	}

	portBASE_TYPE result = xTaskCreate(PIOS_MS5611_Task, (const signed char *)"pios_ms5611",
						 MS5611_TASK_STACK, NULL, MS5611_TASK_PRIORITY,
						 &dev->task);
	PIOS_Assert(result == pdPASS);

	return 0;
}

/**
 * Return the queue that receives pressure data
 */
xQueueHandle PIOS_MS5611_GetQueue()
{
	if (PIOS_MS5611_Validate(dev) != 0)
		return NULL;

	return dev->queue;
}

/**
* Start the ADC conversion
* \param[in] PresOrTemp BMP085_PRES_ADDR or BMP085_TEMP_ADDR
* \return 0 for success, -1 for failure (conversion completed and not read)
*/
static int32_t PIOS_MS5611_StartADC(enum conversion_type type)
{
	if (PIOS_MS5611_Validate(dev) != 0)
		return -1;

	/* Start the conversion */
	switch(type) {
	case TEMPERATURE_CONV:
		while (PIOS_MS5611_WriteCommand(MS5611_TEMP_ADDR + dev->oversampling) != 0)
			continue;
		break;
	case PRESSURE_CONV:
		while (PIOS_MS5611_WriteCommand(MS5611_PRES_ADDR + dev->oversampling) != 0)
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
static int32_t PIOS_MS5611_GetDelay() {
	if (PIOS_MS5611_Validate(dev) != 0)
		return 100;

	switch(dev->oversampling) {
		case MS5611_OSR_256:
			return 2;
		case MS5611_OSR_512:
			return 2;
		case MS5611_OSR_1024:
			return 3;
		case MS5611_OSR_2048:
			return 5;
		case MS5611_OSR_4096:
			return 10;
		default:
			break;
	}
	return 10;
}

/**
* Read the ADC conversion value (once ADC conversion has completed)
* \param[in] PresOrTemp BMP085_PRES_ADDR or BMP085_TEMP_ADDR
* \return 0 if successfully read the ADC, -1 if failed
*/
static int32_t PIOS_MS5611_ReadADC(void)
{
	if (PIOS_MS5611_Validate(dev) != 0)
		return -1;

	uint8_t data[3];
	data[0] = 0;
	data[1] = 0;
	data[2] = 0;
	
	static int64_t delta_temp;

	/* Read and store the 16bit result */
	if (dev->current_conversion_type == TEMPERATURE_CONV) {
		uint32_t raw_temperature;
		/* Read the temperature conversion */
		if (PIOS_MS5611_Read(MS5611_ADC_READ, data, 3) != 0)
			return -1;

		raw_temperature = (data[0] << 16) | (data[1] << 8) | data[2];
		
		delta_temp = ((int32_t) raw_temperature) - (dev->calibration[4] << 8);
		dev->temperature_unscaled = 2000l + ((delta_temp * dev->calibration[5]) >> 23);

	} else {	
		int64_t offset;
		int64_t sens;
		uint32_t raw_pressure;

		/* Read the pressure conversion */
		if (PIOS_MS5611_Read(MS5611_ADC_READ, data, 3) != 0)
			return -1;
		raw_pressure = ((data[0] << 16) | (data[1] << 8) | data[2]);
		
		offset = (((int64_t) dev->calibration[1]) << 16) + ((((int64_t) dev->calibration[3]) * delta_temp) >> 7);
		sens = ((int64_t) dev->calibration[0]) << 15;
		sens = sens + ((((int64_t) dev->calibration[2]) * delta_temp) >> 8);
		
		dev->pressure_unscaled = (((((int64_t) raw_pressure) * sens) >> 21) - offset) >> 15; 
	}
	return 0;
}

/**
 * Return the most recently computed temperature in kPa
 */
static float PIOS_MS5611_GetTemperature(void)
{
	if (PIOS_MS5611_Validate(dev) != 0)
		return -1;

	return ((float) dev->temperature_unscaled) / 100.0f;
}

/**
 * Return the most recently computed pressure in kPa
 */
static float PIOS_MS5611_GetPressure(void)
{
	if (PIOS_MS5611_Validate(dev) != 0)
		return -1;

	return ((float) dev->pressure_unscaled) / 1000.0f;
}

/**
* Reads one or more bytes into a buffer
* \param[in] the command indicating the address to read
* \param[out] buffer destination buffer
* \param[in] len number of bytes which should be read
* \return 0 if operation was successful
* \return -1 if error during I2C transfer
*/
static int32_t PIOS_MS5611_Read(uint8_t address, uint8_t * buffer, uint8_t len)
{

	if (PIOS_MS5611_Validate(dev) != 0)
		return -1;

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = MS5611_I2C_ADDR,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = 1,
			.buf = &address,
		}
		,
		{
		 .info = __func__,
		 .addr = MS5611_I2C_ADDR,
		 .rw = PIOS_I2C_TXN_READ,
		 .len = len,
		 .buf = buffer,
		 }
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
* Writes one or more bytes to the MS5611
* \param[in] address Register address
* \param[in] buffer source buffer
* \return 0 if operation was successful
* \return -1 if error during I2C transfer
*/
static int32_t PIOS_MS5611_WriteCommand(uint8_t command)
{
	if (PIOS_MS5611_Validate(dev) != 0)
		return -1;

	const struct pios_i2c_txn txn_list[] = {
		{
		 .info = __func__,
		 .addr = MS5611_I2C_ADDR,
		 .rw = PIOS_I2C_TXN_WRITE,
		 .len = 1,
		 .buf = &command,
		 }
		,
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
* @brief Run self-test operation.
* \return 0 if self-test succeed, -1 if failed
*/
int32_t PIOS_MS5611_Test()
{
	if (PIOS_MS5611_Validate(dev) != 0)
		return -1;

	// TODO: Is there a better way to test this than just checking that pressure/temperature has changed?
	int32_t cur_value = 0;

	cur_value = dev->temperature_unscaled;
	PIOS_MS5611_StartADC(TEMPERATURE_CONV);
	PIOS_DELAY_WaitmS(5);
	PIOS_MS5611_ReadADC();
	if (cur_value == dev->temperature_unscaled)
		return -1;

	cur_value = dev->pressure_unscaled;
	PIOS_MS5611_StartADC(PRESSURE_CONV);
	PIOS_DELAY_WaitmS(26);
	PIOS_MS5611_ReadADC();
	if (cur_value == dev->pressure_unscaled)
		return -1;
	
	return 0;
}

static void PIOS_MS5611_Task(void *parameters)
{
	int32_t temp_press_interleave_count;
	if (PIOS_MS5611_Validate(dev) != 0)
		temp_press_interleave_count = 1;
	else
		temp_press_interleave_count = dev->temperature_interleaving;

	while (1)
	{
		vTaskDelay(PIOS_MS5611_GetDelay() * portTICK_RATE_MS);

		// If device handle isn't validate pause
		if (PIOS_MS5611_Validate(dev) != 0)
			continue;

		temp_press_interleave_count --;
		if(temp_press_interleave_count <= 0)
		{
			// Update the temperature data
			PIOS_MS5611_StartADC(TEMPERATURE_CONV);
			vTaskDelay(PIOS_MS5611_GetDelay());
			PIOS_MS5611_ReadADC();
			
			temp_press_interleave_count = dev->temperature_interleaving;
		}

		// Update the pressure data
		PIOS_MS5611_StartADC(PRESSURE_CONV);
		vTaskDelay(PIOS_MS5611_GetDelay());
		PIOS_MS5611_ReadADC();

		// Compute the altitude from the pressure and temperature and send it out
		struct pios_ms5611_data data;		
		data.temperature = PIOS_MS5611_GetTemperature();
		data.pressure = PIOS_MS5611_GetPressure();
		data.altitude = 44330.0f * (1.0f - powf(data.pressure / MS5611_P0, (1.0f / 5.255f)));

		xQueueSend(dev->queue, (void*)&data, 0);
	}
}


#endif

/**
 * @}
 * @}
 */
