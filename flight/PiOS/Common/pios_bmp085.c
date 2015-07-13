/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_BMP085 BMP085 Functions
 * @brief Hardware functions to deal with the altitude pressure sensor
 * @{
 *
 * @file       pios_bmp085.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      BMP085 Pressure Sensor Routines
 * @see        The GNU Public License (GPL) Version 3
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

#if defined(PIOS_INCLUDE_BMP085)

#include "pios_bmp085_priv.h"
#include "pios_semaphore.h"
#include "pios_thread.h"
#include "pios_queue.h"

/* Private constants */
#define BMP085_TASK_PRIORITY	PIOS_THREAD_PRIO_HIGHEST
#define BMP085_TASK_STACK		512

/* BMP085 Addresses */
#define BMP085_I2C_ADDR       0x77
#define BMP085_CALIB_ADDR     0xAA
#define BMP085_CALIB_LEN      22
#define BMP085_CTRL_ADDR      0xF4
#define BMP085_PRES_ADDR      0x34
#define BMP085_TEMP_ADDR      0x2E
#define BMP085_ADC_MSB        0xF6
#define BMP085_P0             101.3250f

/* Straight from the datasheet */
static int32_t X1, X2, X3, B3, B5, B6, P;
static uint32_t B4, B7;

/* Private methods */
static int32_t PIOS_BMP085_Read(uint8_t address, uint8_t *buffer, uint8_t len);
static int32_t PIOS_BMP085_WriteCommand(uint8_t address, uint8_t buffer);
static void PIOS_BMP085_Task(void *parameters);

/* Private types */

/* Local Types */

enum pios_bmp085_dev_magic {
	PIOS_BMP085_DEV_MAGIC = 0xefa26e3d,
};

enum conversion_type {
	PRESSURE_CONV,
	TEMPERATURE_CONV
};

struct bmp085_dev {
	const struct pios_bmp085_cfg *cfg;
	uint32_t i2c_id;
	struct pios_thread *task;
	struct pios_queue *queue;

	int64_t pressure_unscaled;
	int64_t temperature_unscaled;
	int16_t AC1;
	int16_t AC2;
	int16_t AC3;
	uint16_t AC4;
	uint16_t AC5;
	uint16_t AC6;
	int16_t B1;
	int16_t B2;
	int16_t MB;
	int16_t MC;
	int16_t MD;

	enum conversion_type current_conversion_type;
	enum pios_bmp085_osr oversampling;
	uint32_t temperature_interleaving;
	int32_t bmp085_read_flag;
	enum pios_bmp085_dev_magic magic;

	struct pios_semaphore *busy;
};

static struct bmp085_dev *dev;

/**
 * @brief Allocate a new device
 */
static struct bmp085_dev *PIOS_BMP085_alloc(void)
{
   struct bmp085_dev *bmp085_dev;

	bmp085_dev = (struct bmp085_dev *)PIOS_malloc(sizeof(*bmp085_dev));
	if (!bmp085_dev) return (NULL);

	bmp085_dev->queue = PIOS_Queue_Create(1, sizeof(struct pios_sensor_baro_data));
	if (bmp085_dev->queue == NULL) {
		PIOS_free(bmp085_dev);
		return NULL;
	}

	bmp085_dev->busy = PIOS_Semaphore_Create();

	bmp085_dev->magic = PIOS_BMP085_DEV_MAGIC;

	return(bmp085_dev);
}

/**
 * @brief Validate the handle to the i2c device
 * @returns 0 for valid device or <0 otherwise
 */
static int32_t PIOS_BMP085_Validate(struct bmp085_dev *dev)
{
	if (dev == NULL)
		return -1;
	if (dev->magic != PIOS_BMP085_DEV_MAGIC)
		return -2;
	if (dev->i2c_id == 0)
		return -3;
	return 0;
}

/**
 * Initialise the BMP085 sensor
 */
int32_t PIOS_BMP085_Init(const struct pios_bmp085_cfg *cfg, int32_t i2c_device)
{
	dev = (struct bmp085_dev *) PIOS_BMP085_alloc();
	if (dev == NULL)
		return -1;

	dev->i2c_id = i2c_device;

	dev->oversampling = cfg->oversampling;
	dev->temperature_interleaving = (cfg->temperature_interleaving) == 0 ? 1 : cfg->temperature_interleaving;
	dev->cfg = cfg;

	uint8_t data[22];
	if (PIOS_BMP085_Read(BMP085_CALIB_ADDR, data, 2) != 0)
		return -2;
	dev->AC1 = (data[0] << 8) | data[1];
	if (PIOS_BMP085_Read(BMP085_CALIB_ADDR, data, 2) != 0)
		return -2;
	dev->AC1 = (data[0] << 8) | data[1];

	if (PIOS_BMP085_Read(BMP085_CALIB_ADDR, data, 22) != 0)
		return -2;
	/* Parameters AC1-AC6 */
	dev->AC1 = (data[0] << 8) | data[1];
	dev->AC2 = (data[2] << 8) | data[3];
	dev->AC3 = (data[4] << 8) | data[5];
	dev->AC4 = (data[6] << 8) | data[7];
	dev->AC5 = (data[8] << 8) | data[9];
	dev->AC6 = (data[10] << 8) | data[11];

	/* Parameters B1, B2 */
	dev->B1  = (data[12] << 8) | data[13];
	dev->B2  = (data[14] << 8) | data[15];

	/* Parameters MB, MC, MD */
	dev->MB  = (data[16] << 8) | data[17];
	dev->MC  = (data[18] << 8) | data[19];
	dev->MD  = (data[20] << 8) | data[21];

	dev->task = PIOS_Thread_Create(
		PIOS_BMP085_Task, "pios_bmp085", BMP085_TASK_STACK, NULL, BMP085_TASK_PRIORITY);
	if (dev->task == NULL)
		return -3;

	PIOS_SENSORS_Register(PIOS_SENSOR_BARO, dev->queue);

	return 0;
}


/**
 * Claim the MS5611 device semaphore.
 * \return 0 if no error
 * \return -1 if timeout before claiming semaphore
 */
static int32_t PIOS_BMP085_ClaimDevice(void)
{
	PIOS_Assert(PIOS_BMP085_Validate(dev) == 0);

	return PIOS_Semaphore_Take(dev->busy, PIOS_SEMAPHORE_TIMEOUT_MAX) == true ? 0 : 1;
}

/**
 * Release the MS5611 device semaphore.
 * \return 0 if no error
 */
static int32_t PIOS_BMP085_ReleaseDevice(void)
{
	PIOS_Assert(PIOS_BMP085_Validate(dev) == 0);

	return PIOS_Semaphore_Give(dev->busy) == true ? 0 : 1;
}


/**
* Start the ADC conversion
* \param[in] PRESSURE_CONV or TEMPERATURE_CONV to select which measurement to make
* \return 0 for success, -1 for failure (conversion completed and not read)
*/
static int32_t PIOS_BMP085_StartADC(enum conversion_type type)
{
	if (PIOS_BMP085_Validate(dev) != 0)
		return -1;

	/* Start the conversion */
	switch(type) {
	case TEMPERATURE_CONV:
		while (PIOS_BMP085_WriteCommand(BMP085_CTRL_ADDR, BMP085_TEMP_ADDR) != 0)
			continue;
		break;
	case PRESSURE_CONV:
		while (PIOS_BMP085_WriteCommand(BMP085_CTRL_ADDR, BMP085_PRES_ADDR + (dev->oversampling << 6)) != 0)
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
static int32_t PIOS_BMP085_GetDelay() {
	if (PIOS_BMP085_Validate(dev) != 0)
		return 100;

	switch(dev->oversampling) {
		case BMP085_OSR_0:
			return 5;
		case BMP085_OSR_1:
			return 8;
		case BMP085_OSR_2:
			return 14;
		case BMP085_OSR_3:
			return 26;
		default:
			break;
	}
	return 26;
}

/**
* Read the ADC conversion value (once ADC conversion has completed)
* \return 0 if successfully read the ADC, -1 if failed
*/
static int32_t PIOS_BMP085_ReadADC(void)
{
	if (PIOS_BMP085_Validate(dev) != 0)
		return -1;

	uint8_t data[3];

	/* Read and store the 16bit result */
	if (dev->current_conversion_type == TEMPERATURE_CONV) {
		uint32_t raw_temperature;
		/* Read the temperature conversion */
		if (PIOS_BMP085_Read(BMP085_ADC_MSB, data, 2) != 0)
			return -1;

		raw_temperature = ((data[0] << 8) | data[1]);

		X1 = (raw_temperature - dev->AC6) * dev->AC5 >> 15;
		X2 = ((int32_t)dev->MC << 11) / (X1 + dev->MD);
		B5 = X1 + X2;
		dev->temperature_unscaled = (B5 + 8) >> 4;//temperature;
	} else {
		uint32_t raw_pressure;

		/* Read the pressure conversion */
		if (PIOS_BMP085_Read(BMP085_ADC_MSB, data, 3) != 0)
			return -1;

		raw_pressure = ((data[0] << 16) | (data[1] << 8) | data[2]) >> (8 - dev->oversampling);

		B6 = B5 - 4000;
		X1 = (dev->B2 * (B6 * B6 >> 12)) >> 11;
		X2 = dev->AC2 * B6 >> 11;
		X3 = X1 + X2;
		B3 = ((((int32_t)dev->AC1 * 4 + X3) << dev->oversampling) + 2) >> 2;
		X1 = dev->AC3 * B6 >> 13;
		X2 = (dev->B1 * (B6 * B6 >> 12)) >> 16;
		X3 = ((X1 + X2) + 2) >> 2;
		B4 = (dev->AC4 * (uint32_t)(X3 + 32768)) >> 15;
		B7 = ((uint32_t)raw_pressure - B3) * (50000 >> dev->oversampling);
		P  = B7 < 0x80000000 ? (B7 * 2) / B4 : (B7 / B4) * 2;
		X1 = (P >> 8) * (P >> 8);
		X1 = (X1 * 3038) >> 16;
		X2 = (-7357 * P) >> 16;
		dev->pressure_unscaled = P + ((X1 + X2 + 3791) >> 4);
	}
	return 0;
}

/*
* Reads one or more bytes into a buffer
* \param[in] the command indicating the address to read
* \param[out] buffer destination buffer
* \param[in] len number of bytes which should be read
* \return 0 if operation was successful
* \return -1 if error during I2C transfer
*/
static int32_t PIOS_BMP085_Read(uint8_t address, uint8_t *buffer, uint8_t len)
{

	if (PIOS_BMP085_Validate(dev) != 0)
		return -1;

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = BMP085_I2C_ADDR,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = 1,
			.buf = &address,
		}
		,
		{
		 .info = __func__,
		 .addr = BMP085_I2C_ADDR,
		 .rw = PIOS_I2C_TXN_READ,
		 .len = len,
		 .buf = buffer,
		 }
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
* Writes one or more bytes to the BMP085
* \param[in] address Register address
* \param[in] buffer source buffer
* \return 0 if operation was successful
* \return -1 if error during I2C transfer
*/
static int32_t PIOS_BMP085_WriteCommand(uint8_t address, uint8_t buffer)
{
	uint8_t data[] = {
		address,
		buffer,
	};

	if (PIOS_BMP085_Validate(dev) != 0)
		return -1;

	const struct pios_i2c_txn txn_list[] = {
		{
		 .info = __func__,
		 .addr = BMP085_I2C_ADDR,
		 .rw = PIOS_I2C_TXN_WRITE,
		 .len = sizeof(data),
		 .buf = data,
		 }
		,
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
* @brief Run self-test operation.
* \return 0 if self-test succeed, -1 if failed
*/
int32_t PIOS_BMP085_Test()
{
	if (PIOS_BMP085_Validate(dev) != 0)
		return -1;

	// TODO: Is there a better way to test this than just checking that pressure/temperature has changed?
	int32_t cur_value = 0;

	PIOS_BMP085_ClaimDevice();
	cur_value = dev->temperature_unscaled;
	PIOS_BMP085_StartADC(TEMPERATURE_CONV);
	PIOS_DELAY_WaitmS(5);
	PIOS_BMP085_ReadADC();
	PIOS_BMP085_ReleaseDevice();
	if (cur_value == dev->temperature_unscaled)
		return -1;

	PIOS_BMP085_ClaimDevice();
	cur_value = dev->pressure_unscaled;
	PIOS_BMP085_StartADC(PRESSURE_CONV);
	PIOS_DELAY_WaitmS(26);
	PIOS_BMP085_ReadADC();
	PIOS_BMP085_ReleaseDevice();


	if (cur_value == dev->pressure_unscaled)
		return -1;

	return 0;
}
static void PIOS_BMP085_Task(void *parameters)
{
	int32_t temp_press_interleave_count = dev->temperature_interleaving;
	int32_t read_adc_result = 0;

	while (1) {

		temp_press_interleave_count --;
		if(temp_press_interleave_count <= 0)
		{
			// Update the temperature data
			PIOS_BMP085_ClaimDevice();
			PIOS_BMP085_StartADC(TEMPERATURE_CONV);
			PIOS_Thread_Sleep(5);
			read_adc_result = PIOS_BMP085_ReadADC();
			PIOS_BMP085_ReleaseDevice();
			temp_press_interleave_count = dev->temperature_interleaving;
		}
		// Update the pressure data
		PIOS_BMP085_ClaimDevice();
		PIOS_BMP085_StartADC(PRESSURE_CONV);
		PIOS_Thread_Sleep(PIOS_BMP085_GetDelay());
		read_adc_result = PIOS_BMP085_ReadADC();
		PIOS_BMP085_ReleaseDevice();

		// Compute the altitude from the pressure and temperature and send it out
		struct pios_sensor_baro_data data;
		data.temperature = ((float) dev->temperature_unscaled) / 10.0f;
		data.pressure = ((float) dev->pressure_unscaled) / 1000.0f;
		data.altitude = 44330.0f * (1.0f - powf(data.pressure / BMP085_P0, (1.0f / 5.255f)));

		if (read_adc_result == 0) {
			PIOS_Queue_Send(dev->queue, (void*)&data, 0);
		}
	}
}

#endif /* PIOS_INCLUDE_BMP085 */
