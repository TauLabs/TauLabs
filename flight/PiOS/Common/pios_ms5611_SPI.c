/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MS5611 MS5611 Functions
 * @brief Hardware functions to deal with the altitude pressure sensor
 * @{
 *
 * @file       pios_ms5611_SPI.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Virtual Robotix Network Team, http://www.virtualrobotix.com Copyright (C) 2013.
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

#if defined(PIOS_INCLUDE_MS5611_SPI)

#include "pios_ms5611_SPI_priv.h"

/* Private constants */
#define PIOS_MS5611_OVERSAMPLING oversampling
#define MS5611_TASK_PRIORITY	(tskIDLE_PRIORITY + configMAX_PRIORITIES - 1)	// max priority
#define MS5611_TASK_STACK		(512 / 4)

/* MS5611 Addresses */
#define MS5611_RESET            0x1E
#define MS5611_CALIB_ADDR       0xA2  /* First sample is factory stuff */
#define MS5611_CALIB_LEN        16
#define MS5611_ADC_READ         0x00
#define MS5611_PRES_ADDR        0x40
#define MS5611_TEMP_ADDR        0x50
#define MS5611_ADC_MSB          0xF6
#define MS5611_P0               101.3250f

/* Private methods */
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
	uint32_t spi_id;
	uint32_t slave_num;
	xTaskHandle task;
	xQueueHandle queue;

	int64_t pressure_unscaled;
	int64_t temperature_unscaled;
	uint16_t calibration[6];
	enum conversion_type current_conversion_type;
	enum pios_ms5611_osr oversampling;
	uint32_t temperature_interleaving;
//	int32_t ms5611_read_flag;
	enum pios_ms5611_dev_magic magic;
};

static struct ms5611_dev *dev;

/**
 * @brief Allocate a new device
 */
static struct ms5611_dev * PIOS_MS5611_alloc(void)
{
	struct ms5611_dev *ms5611_dev;
	
	ms5611_dev = (struct ms5611_dev *)pvPortMalloc(sizeof(*ms5611_dev));
	if (!ms5611_dev) return (NULL);

	ms5611_dev->queue = xQueueCreate(1, sizeof(struct pios_sensor_baro_data));
	if (ms5611_dev->queue == NULL) {
		vPortFree(ms5611_dev);
		return NULL;
	}

	ms5611_dev->magic = PIOS_MS5611_DEV_MAGIC;
	
	return(ms5611_dev);
}

/**
 * @brief Validate the handle to the i2c device
 * @returns 0 for valid device or <0 otherwise
 */
static int32_t PIOS_MS5611_Validate(struct ms5611_dev *dev)
{
	if (dev == NULL) 
		return -1;
	if (dev->magic != PIOS_MS5611_DEV_MAGIC)
		return -2;
	if (dev->spi_id == 0)
		return -3;
	return 0;
}

/**
 * @brief Claim the SPI bus for the baro communications and select this chip
 * @return 0 if successful, -1 for invalid device, -2 if unable to claim bus
 */
int32_t PIOS_MS5611_ClaimBus()
{
	if(PIOS_MS5611_Validate(dev) != 0)
		return -1;

	if(PIOS_SPI_ClaimBus(dev->spi_id) != 0)
		return -2;

	PIOS_SPI_RC_PinSet(dev->spi_id,dev->slave_num,0);
	return 0;
}

/**
 * @brief Release the SPI bus for the baro communications and end the transaction
 * @return 0 if successful
 */
int32_t PIOS_MS5611_ReleaseBus()
{
	if(PIOS_MS5611_Validate(dev) != 0)
		return -1;

	PIOS_SPI_RC_PinSet(dev->spi_id,dev->slave_num,1);

	return PIOS_SPI_ReleaseBus(dev->spi_id);
}

/**
 * Initialise the MS5611 sensor
 */
int32_t PIOS_MS5611_Init(uint32_t spi_id, uint32_t slave_num, const struct pios_ms5611_cfg *cfg)
{
	dev = (struct ms5611_dev *) PIOS_MS5611_alloc();
	if (dev == NULL)
		return -1;

	dev->spi_id = spi_id;
	dev->slave_num = slave_num;

	dev->oversampling = cfg->oversampling;
	dev->temperature_interleaving = (cfg->temperature_interleaving) == 0 ? 1 : cfg->temperature_interleaving;
	dev->cfg = cfg;

	/* Configure the MS5611 Sensor */
	PIOS_SPI_SetClockSpeed(dev->spi_id, PIOS_SPI_PRESCALER_256);

	// RESET ms5611
	uint8_t out[] = {MS5611_RESET};

	if(PIOS_MS5611_ClaimBus() < 0)
		return -2;

	if(PIOS_SPI_TransferBlock(dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_MS5611_ReleaseBus();
		return -2;
		}
	PIOS_MS5611_ReleaseBus();

	PIOS_DELAY_WaitmS(20);

	uint8_t data[2];

	/* Calibration parameters */
	for (int i = 0; i < 6; i++) {
		if(PIOS_MS5611_ClaimBus() < 0)
			return -2;

		if(PIOS_SPI_TransferByte(dev->spi_id, MS5611_CALIB_ADDR + i*2) < 0) {
				PIOS_MS5611_ReleaseBus();
				return -3;
				}
		if(PIOS_SPI_TransferBlock(dev->spi_id,NULL,data,sizeof(data),NULL) < 0) {
				PIOS_MS5611_ReleaseBus();
				return -3;
				}
		dev->calibration[i] = (data[0] << 8) | data[1];

		PIOS_MS5611_ReleaseBus();
	}

	portBASE_TYPE result = xTaskCreate(PIOS_MS5611_Task, (const signed char *)"pios_ms5611",
						 MS5611_TASK_STACK, NULL, MS5611_TASK_PRIORITY,
						 &dev->task);
	PIOS_Assert(result == pdPASS);

	PIOS_SENSORS_Register(PIOS_SENSOR_BARO, dev->queue);

	return 0;
}

/**
* Start the ADC conversion
* \param[in] PRESSURE_CONV or TEMPERATURE_CONV to select which measurement to make
* \return 0 for success, -1 for failure (conversion completed and not read)
*/
static int32_t PIOS_MS5611_StartADC(enum conversion_type type)
{
	uint8_t outT[] = {MS5611_TEMP_ADDR + dev->oversampling};
	uint8_t outP[] = {MS5611_PRES_ADDR + dev->oversampling};

	if (PIOS_MS5611_Validate(dev) != 0)
		return -1;

	if(PIOS_MS5611_ClaimBus() < 0)
		return -2;

	/* Start the conversion */
	switch(type) {
	case TEMPERATURE_CONV:
		if(PIOS_SPI_TransferBlock(dev->spi_id,outT,NULL,sizeof(outT),NULL) < 0) {
			PIOS_MS5611_ReleaseBus();
			return -3;
			}
		break;
	case PRESSURE_CONV:
		if(PIOS_SPI_TransferBlock(dev->spi_id,outP,NULL,sizeof(outP),NULL) < 0) {
			PIOS_MS5611_ReleaseBus();
			return -3;
			}
		break;
	default:
		return -1;
	}

	dev->current_conversion_type = type;
	PIOS_MS5611_ReleaseBus();
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
* \return 0 if successfully read the ADC, -1 if failed
*/
static int32_t PIOS_MS5611_ReadADC(void)
{
	if (PIOS_MS5611_Validate(dev) != 0)
		return -1;

	uint8_t out[] = {MS5611_ADC_READ};
	uint8_t Data[] = {0x00, 0x00, 0x00};
	
	static int64_t delta_temp;

	if(PIOS_MS5611_ClaimBus() < 0)
		return -2;

	/* Read and store the 16bit result */
	if (dev->current_conversion_type == TEMPERATURE_CONV) {
		uint32_t raw_temperature;
		/* Read the temperature conversion */
		if(PIOS_SPI_TransferBlock(dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
			PIOS_MS5611_ReleaseBus();
			return -1;
			}
		if(PIOS_SPI_TransferBlock(dev->spi_id,NULL,Data,sizeof(Data),NULL) < 0) {
			PIOS_MS5611_ReleaseBus();
			return -1;
			}

		PIOS_MS5611_ReleaseBus();

		raw_temperature = (Data[0] << 16) | (Data[1] << 8) | Data[2];

		delta_temp = ((int32_t) raw_temperature) - (dev->calibration[4] << 8);
		dev->temperature_unscaled = 2000l + ((delta_temp * dev->calibration[5]) >> 23);

		} else {
		int64_t offset;
		int64_t sens;
		uint32_t raw_pressure;

		/* Read the pressure conversion */
		if(PIOS_SPI_TransferBlock(dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
			PIOS_MS5611_ReleaseBus();
			return -1;
			}
		if(PIOS_SPI_TransferBlock(dev->spi_id,NULL,Data,sizeof(Data),NULL) < 0) {
			PIOS_MS5611_ReleaseBus();
			return -1;
			}

		PIOS_MS5611_ReleaseBus();
		raw_pressure = ((Data[0] << 16) | (Data[1] << 8) | Data[2]);

		offset = (((int64_t) dev->calibration[1]) << 16) + ((((int64_t) dev->calibration[3]) * delta_temp) >> 7);
		sens = ((int64_t) dev->calibration[0]) << 15;
		sens = sens + ((((int64_t) dev->calibration[2]) * delta_temp) >> 8);
		
		dev->pressure_unscaled = (((((int64_t) raw_pressure) * sens) >> 21) - offset) >> 15; 
		}
	return 0;
}

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

	// If device handle isn't validate pause
	while (PIOS_MS5611_Validate(dev) != 0) {
		vTaskDelay(1000);
	}

	while (1) {

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
		struct pios_sensor_baro_data data;		
		data.temperature = ((float) dev->temperature_unscaled) / 100.0f;
		data.pressure = ((float) dev->pressure_unscaled) / 1000.0f;
		data.altitude = 44330.0f * (1.0f - powf(data.pressure / MS5611_P0, (1.0f / 5.255f)));

		xQueueSend(dev->queue, (void*)&data, 0);
	}
}


#endif

/**
 * @}
 * @}
 */
