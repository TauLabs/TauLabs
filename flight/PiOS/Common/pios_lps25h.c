/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_LPS25H LPS25H Functions
 * @brief Hardware functions to deal with the altitude pressure sensor
 * @{
 *
 * @file       pios_lps25h.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      LPS25H Pressure Sensor Routines
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

#define PIOS_INCLUDE_LPS25H
#if defined(PIOS_INCLUDE_LPS25H)

#include "pios_lps25h.h"
#include "pios_sensors.h"

/* Private constants */
#define LPS25H_TASK_PRIORITY	(tskIDLE_PRIORITY + configMAX_PRIORITIES - 1)	// max priority
#define LPS25H_TASK_STACK		(512 / 4)

#define ST_PRESS_LSB_PER_KPA         40960UL
#define ST_PRESS_LSB_PER_CELSIUS       480UL

#define LPS25H_DEVICE_ID               0xbd

#define LPS25H_WAI_EXP                 0xbd
#define LPS25H_ODR_ADDR                0x20
#define LPS25H_ODR_MASK                0x70
#define LPS25H_ODR_AVL_1HZ_VAL         0x01
#define LPS25H_ODR_AVL_7HZ_VAL         0x02
#define LPS25H_ODR_AVL_13HZ_VAL        0x03
#define LPS25H_ODR_AVL_25HZ_VAL        0x04
#define LPS25H_PW_ADDR                 0x20
#define LPS25H_PW_MASK                 0x80
#define LPS25H_FS_ADDR                 0x00
#define LPS25H_FS_MASK                 0x00
#define LPS25H_FS_AVL_1260_VAL         0x00
#define LPS25H_FS_AVL_1260_GAIN        ST_PRESS_KPASCAL_NANO_SCALE
#define LPS25H_FS_AVL_TEMP_GAIN        ST_PRESS_CELSIUS_NANO_SCALE
#define LPS25H_BDU_ADDR                0x20
#define LPS25H_BDU_MASK                0x04
#define LPS25H_DRDY_IRQ_ADDR           0x23
#define LPS25H_DRDY_IRQ_INT1_MASK      0x01
#define LPS25H_DRDY_IRQ_INT2_MASK      0x10
#define LPS25H_MULTIREAD_BIT           true
#define LPS25H_TEMP_OFFSET             42500

#define LPS25H_PRESS_OUT_XL_ADDR       0x28
#define LPS25H_PRESS_OUT_L_ADDR        0x29
#define LPS25H_PRESS_OUT_H_ADDR        0x2a

#define LPS25H_TEMP_OUT_L_ADDR        0x2b
#define LPS25H_TEMP_OUT_H_ADDR        0x2c


#define LPS25H_RES_CONF                0x10
#define LPS25H_STATUS_REG              0x27
#define LPS25H_CTRL_REG1               0x20
#define LPS25H_CTRL_REG2               0x21
#define LPS25H_FIFO_CTRL               0x2e
#define LPS25H_WHO_AM_I                0x0f

/* Commands */
#define LPS25H_RESET                   0x84
#define LPS25H_ONE_SHOT_ENABLE         0x01

/* Private methods */
static int32_t PIOS_LPS25H_Read(uint8_t address, uint8_t * buffer, uint8_t len);
static int32_t PIOS_LPS25H_Write(uint8_t address, uint8_t value);
static int32_t PIOS_LPS25H_StartADC(void);
static int32_t PIOS_LPS25H_ReadADC(void);

static void PIOS_LPS25H_Task(void *parameters);

/* Private types */

/* Local Types */

enum pios_LPS25H_dev_magic {
	PIOS_LPS25H_DEV_MAGIC = 0x01ab6acd,
};


struct LPS25H_dev {
	const struct pios_lps25h_cfg * cfg;

	uint32_t i2c_id;
	xTaskHandle task;
	xQueueHandle queue;

	uint8_t i2c_addr;
	int64_t pressure_unscaled;
	int32_t temperature_unscaled;
	enum pios_LPS25H_dev_magic magic;

#if defined(PIOS_INCLUDE_FREERTOS)
	xSemaphoreHandle busy;
#else
	bool busy;
#endif
};

static struct LPS25H_dev *dev;

/**
 * @brief Allocate a new device
 */
static struct LPS25H_dev * PIOS_LPS25H_alloc(void)
{
	struct LPS25H_dev *LPS25H_dev;

	LPS25H_dev = (struct LPS25H_dev *)pvPortMalloc(sizeof(*LPS25H_dev));
	if (!LPS25H_dev)
		return (NULL);

	memset(LPS25H_dev, 0, sizeof(*LPS25H_dev));

	LPS25H_dev->queue = xQueueCreate(1, sizeof(struct pios_sensor_baro_data));
	if (LPS25H_dev->queue == NULL) {
		vPortFree(LPS25H_dev);
		return NULL;
	}

	LPS25H_dev->magic = PIOS_LPS25H_DEV_MAGIC;

#if defined(PIOS_INCLUDE_FREERTOS)
	vSemaphoreCreateBinary(LPS25H_dev->busy);
	PIOS_Assert(LPS25H_dev->busy != NULL);
#else
	LPS25H_dev->busy = false;
#endif

	return LPS25H_dev;
}

/**
 * @brief Validate the handle to the i2c device
 * @returns 0 for valid device or <0 otherwise
 */
static int32_t PIOS_LPS25H_Validate(struct LPS25H_dev *dev)
{
	if (dev == NULL)
		return -1;
	if (dev->magic != PIOS_LPS25H_DEV_MAGIC)
		return -2;
	return 0;
}

/**
 * @brief Return the delay for the current odr
 */
static int32_t PIOS_LPS25H_GetDelay()
{
	switch(dev->cfg->odr) {
	case LPS25H_ODR_1HZ:
		return 1000;
	case LPS25H_ODR_7HZ:
		return 143;
	case LPS25H_ODR_12HZ:
		return 85;
	case LPS25H_ODR_25HZ:
		return 40;
	}
	return 100;
}

/**
 * Initialise the LPS25H sensor
 */
int32_t PIOS_LPS25H_Init(const struct pios_lps25h_cfg *cfg, int32_t i2c_device)
{
	uint8_t tmp;

	dev = (struct LPS25H_dev *)PIOS_LPS25H_alloc();
	if (dev == NULL)
		return -1;

	dev->i2c_id = i2c_device;
	dev->cfg = cfg;
	dev->i2c_addr = cfg->i2c_addr;

	// Reset the baro
	if (PIOS_LPS25H_Write(LPS25H_CTRL_REG2, LPS25H_RESET) != 0)
		return -2;

	// Give it some time to reset
	PIOS_DELAY_WaitmS(50);

	// Check the ID
	if (PIOS_LPS25H_Read(LPS25H_WHO_AM_I, &tmp, 1) != 0)
		return -3;

	if (tmp != LPS25H_DEVICE_ID)
		return -4;

	// Averaging configuration: 32 for pressure, 16 for temperature
	if (PIOS_LPS25H_Write(LPS25H_RES_CONF, 0x05) != 0)
		return -2;

	// Configure FIFO, use 4 sample moving average
	if (PIOS_LPS25H_Write(LPS25H_FIFO_CTRL, 0xc3) != 0)
		return -2;

	// Power on device and set ODR
	if (PIOS_LPS25H_Write(LPS25H_CTRL_REG1, 0x80 | cfg->odr << 4) != 0)
		return -2;

	PIOS_SENSORS_Register(PIOS_SENSOR_BARO, dev->queue);

	portBASE_TYPE result = xTaskCreate(PIOS_LPS25H_Task, (const signed char *)"pios_LPS25H",
					LPS25H_TASK_STACK, NULL, LPS25H_TASK_PRIORITY,
					&dev->task);
	PIOS_Assert(result == pdPASS);

	return 0;
}

/**
 * Claim the LPS25H device semaphore.
 * \return 0 if no error
 * \return -1 if timeout before claiming semaphore
 */
static int32_t PIOS_LPS25H_ClaimDevice(void)
{
	PIOS_Assert(PIOS_LPS25H_Validate(dev) == 0);

#if defined(PIOS_INCLUDE_FREERTOS)
	if (xSemaphoreTake(dev->busy, 0xffff) != pdTRUE)
		return -1;
#else
	uint32_t timeout = 0xffff;
	while ((dev->busy == true) && --timeout);
	if (timeout == 0) //timed out
		return -1;

	PIOS_IRQ_Disable();
	if (dev->busy == true) {
		PIOS_IRQ_Enable();
		return -1;
	}
	dev->busy = true;
	PIOS_IRQ_Enable();
#endif
	return 0;
}

/**
 * Release the LPS25H device semaphore.
 * \return 0 if no error
 */
static int32_t PIOS_LPS25H_ReleaseDevice(void)
{
	PIOS_Assert(PIOS_LPS25H_Validate(dev) == 0);

#if defined(PIOS_INCLUDE_FREERTOS)
	xSemaphoreGive(dev->busy);
#else
	PIOS_IRQ_Disable();
	dev->busy = false;
	PIOS_IRQ_Enable();
#endif

	return 0;
}

/**
 * @brief Start the ADC conversion
 * @return 0 if no error
 */
static int32_t PIOS_LPS25H_StartADC(void)
{
	if (PIOS_LPS25H_Write(LPS25H_CTRL_REG2, LPS25H_ONE_SHOT_ENABLE) != 0)
		return -1;
	return 0;
}

/**
 * @brief Read pressure and temperature
 * @return 0 if no error
 */
static int32_t PIOS_LPS25H_ReadADC(void)
{
	uint8_t tmp, tmp1, tmp2;
	// make sure data is available
	if (PIOS_LPS25H_Read(LPS25H_STATUS_REG, &tmp, 1) != 0)
		return -1;

	if (tmp & 0x03 != 0x03)
		return -2;

	// read pressure data LSB
	if (PIOS_LPS25H_Read(LPS25H_PRESS_OUT_XL_ADDR, &tmp, 1) != 0)
		return -1;

	// read pressure data middle byte
	if (PIOS_LPS25H_Read(LPS25H_PRESS_OUT_L_ADDR, &tmp1, 1) != 0)
		return -1;

	// read pressure data MSB
	if (PIOS_LPS25H_Read(LPS25H_PRESS_OUT_H_ADDR, &tmp2, 1) != 0)
		return -1;

	dev->pressure_unscaled = (int64_t)(tmp2 << 16 | tmp1 << 8 | tmp);

	// read temperature data LSB
	if (PIOS_LPS25H_Read(LPS25H_TEMP_OUT_L_ADDR, &tmp, 1) != 0)
		return -1;

	// read temperature data MSB
	if (PIOS_LPS25H_Read(LPS25H_TEMP_OUT_H_ADDR, &tmp1, 1) != 0)
		return -1;

	dev->temperature_unscaled = (int32_t)(tmp1 << 8 | tmp);
	//printf("temp: %d ", (int)dev->temperature_unscaled);

	return 0;
}

///**
//* Reads one or more bytes into a buffer
//* \param[in] the command indicating the address to read
//* \param[out] buffer destination buffer
//* \param[in] len number of bytes which should be read
//* \return 0 if operation was successful
//* \return -1 if dev is invalid
//* \return -2 if error during I2C transfer
//*/
static int32_t PIOS_LPS25H_Read(uint8_t address, uint8_t *buffer, uint8_t len)
{
	if (PIOS_LPS25H_Validate(dev) != 0)
		return -1;

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = dev->i2c_addr,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = 1,
			.buf = &address,
		},
		{
			.info = __func__,
			.addr = dev->i2c_addr,
			.rw = PIOS_I2C_TXN_READ,
			.len = len,
			.buf = buffer,
		 }
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
 * @brief Write register
 * @param address
 * @param value
 * @return 0 if no error
 */
static int32_t PIOS_LPS25H_Write(uint8_t address, uint8_t value)
{
	uint8_t data[] = {
		address,
		value,
	};

	if (PIOS_LPS25H_Validate(dev) != 0)
		return -1;

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = dev->i2c_addr,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(data),
			.buf = data,
		 }
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

static void PIOS_LPS25H_Task(void *parameters)
{

	while (1) {
		// Update the pressure data
		PIOS_LPS25H_ClaimDevice();
		vTaskDelay(MS2TICKS(PIOS_LPS25H_GetDelay()));
		PIOS_LPS25H_ReadADC();
		PIOS_LPS25H_ReleaseDevice();

		// Compute the altitude from the pressure and temperature and send it out
		struct pios_sensor_baro_data data;
		data.temperature = 42.5 + (float)((100 * dev->temperature_unscaled) / ST_PRESS_LSB_PER_CELSIUS) / 100.0f;
		data.pressure = (float)((1000 * dev->pressure_unscaled) / (ST_PRESS_LSB_PER_KPA)) / 1000.0f;
		data.altitude = 44330.0f * (1.0f - powf(data.pressure / 101.3250f, (1.0f / 5.255f)));
		xQueueSend(dev->queue, (void*)&data, 0);
	}
}


#endif /** defined(PIOS_INCLUDE_LPS25H) **/

/**
 * @}
 * @}
 */
