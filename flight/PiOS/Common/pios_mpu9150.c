/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MPU9150 MPU9150 Functions
 * @brief Deals with the hardware interface to the 3-axis gyro
 * @{
 *
 * @file       pios_mpu9150.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      MPU9150 9-axis gyro accel and mag chip
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
#include "physical_constants.h"

#if defined(PIOS_INCLUDE_MPU9150)

#include "pios_mpu60x0.h"

/* Private constants */
#define MPU9150_TASK_PRIORITY	(tskIDLE_PRIORITY + configMAX_PRIORITIES - 1)	// max priority
#define MPU9150_TASK_STACK		(512 / 4)

#define MPU9150_WHOAMI_ID        0x68
#define MPU9150_MAG_ADDR         0x0c
#define MPU9150_MAG_STATUS       0x02
#define MPU9150_MAG_XH           0x03
#define MPU9150_MAG_XL           0x04
#define MPU9150_MAG_YH           0x05
#define MPU9150_MAG_YL           0x06
#define MPU9150_MAG_ZH           0x07
#define MPU9150_MAG_ZL           0x08
#define MPU9150_MAG_STATUS2      0x09
#define MPU9150_MAG_CNTR         0x0a

/* Global Variables */

enum pios_mpu9150_dev_magic {
	PIOS_MPU9150_DEV_MAGIC = 0xf212da62,
};

#define PIOS_MPU9150_MAX_DOWNSAMPLE 2
struct mpu9150_dev {
	uint32_t i2c_id;
	uint8_t i2c_addr;
	enum pios_mpu60x0_accel_range accel_range;
	enum pios_mpu60x0_range gyro_range;
	xQueueHandle gyro_queue;
	xQueueHandle accel_queue;
	xQueueHandle mag_queue;
	xTaskHandle TaskHandle;
	xSemaphoreHandle data_ready_sema;
	const struct pios_mpu60x0_cfg * cfg;
	enum pios_mpu60x0_filter filter;
	enum pios_mpu9150_dev_magic magic;
};

//! Global structure for this device device
static struct mpu9150_dev * dev;

//! Private functions
static struct mpu9150_dev * PIOS_MPU9150_alloc(void);
static int32_t PIOS_MPU9150_Validate(struct mpu9150_dev * dev);
static int32_t PIOS_MPU9150_Config(struct pios_mpu60x0_cfg const * cfg);
static int32_t PIOS_MPU9150_SetReg(uint8_t address, uint8_t buffer);
static int32_t PIOS_MPU9150_GetReg(uint8_t address);
static int32_t PIOS_MPU9150_ReadID();
static int32_t PIOS_MPU9150_Mag_SetReg(uint8_t reg, uint8_t buffer);
static int32_t PIOS_MPU9150_Mag_GetReg(uint8_t reg);
static void PIOS_MPU9150_Task(void *parameters);

/**
 * @brief Allocate a new device
 */
static struct mpu9150_dev * PIOS_MPU9150_alloc(void)
{
	struct mpu9150_dev * mpu9150_dev;
	
	mpu9150_dev = (struct mpu9150_dev *)PIOS_malloc(sizeof(*mpu9150_dev));
	if (!mpu9150_dev) return (NULL);
	
	mpu9150_dev->magic = PIOS_MPU9150_DEV_MAGIC;
	
	mpu9150_dev->accel_queue = xQueueCreate(PIOS_MPU9150_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_gyro_data));
	if (mpu9150_dev->accel_queue == NULL) {
		vPortFree(mpu9150_dev);
		return NULL;
	}

	mpu9150_dev->gyro_queue = xQueueCreate(PIOS_MPU9150_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_gyro_data));
	if (mpu9150_dev->gyro_queue == NULL) {
		vPortFree(mpu9150_dev);
		return NULL;
	}

	mpu9150_dev->mag_queue = xQueueCreate(PIOS_MPU9150_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_mag_data));
	if (mpu9150_dev->mag_queue == NULL) {
		vPortFree(mpu9150_dev);
		return NULL;
	}

	mpu9150_dev->data_ready_sema = xSemaphoreCreateMutex();
	if (mpu9150_dev->data_ready_sema == NULL) {
		vPortFree(mpu9150_dev);
		return NULL;
	}

	return(mpu9150_dev);
}

/**
 * @brief Validate the handle to the i2c device
 * @returns 0 for valid device or -1 otherwise
 */
static int32_t PIOS_MPU9150_Validate(struct mpu9150_dev * dev)
{
	if (dev == NULL) 
		return -1;
	if (dev->magic != PIOS_MPU9150_DEV_MAGIC)
		return -2;
	if (dev->i2c_id == 0)
		return -3;
	return 0;
}

/**
 * @brief Initialize the MPU9150 3-axis gyro sensor.
 * @return 0 for success, -1 for failure to allocate, -2 for failure to get irq
 */
int32_t PIOS_MPU9150_Init(uint32_t i2c_id, uint8_t i2c_addr, const struct pios_mpu60x0_cfg * cfg)
{
	dev = PIOS_MPU9150_alloc();
	if (dev == NULL)
		return -1;
	
	dev->i2c_id = i2c_id;
	dev->i2c_addr = i2c_addr;
	dev->cfg = cfg;

	/* Configure the MPU9150 Sensor */
	if (PIOS_MPU9150_Config(cfg) != 0)
		return -2;

	/* Set up EXTI line */
	PIOS_EXTI_Init(cfg->exti_cfg);

	// Wait 5 ms for data ready interrupt and make sure it happens
	// twice
	if ((xSemaphoreTake(dev->data_ready_sema, 5) != pdTRUE) ||
		(xSemaphoreTake(dev->data_ready_sema, 5) != pdTRUE)) {
		return -10;
	}

	int result = xTaskCreate(PIOS_MPU9150_Task, (const signed char *)"PIOS_MPU9150_Task",
						 MPU9150_TASK_STACK, NULL, MPU9150_TASK_PRIORITY,
						 &dev->TaskHandle);
	PIOS_Assert(result == pdPASS);

	PIOS_SENSORS_Register(PIOS_SENSOR_ACCEL, dev->accel_queue);
	PIOS_SENSORS_Register(PIOS_SENSOR_GYRO, dev->gyro_queue);
	PIOS_SENSORS_Register(PIOS_SENSOR_MAG, dev->mag_queue);

	return 0;
}

/**
 * @brief Initialize the MPU9150 3-axis gyro sensor
 * \return none
 * \param[in] PIOS_MPU9150_ConfigTypeDef struct to be used to configure sensor.
*
*/
static int32_t PIOS_MPU9150_Config(struct pios_mpu60x0_cfg const * cfg)
{
	// Reset chip
	if (PIOS_MPU9150_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, PIOS_MPU60X0_PWRMGMT_IMU_RST) != 0)
		return -1;

	// Give chip some time to initialize
	PIOS_DELAY_WaitmS(50);
	PIOS_WDG_Clear();

	//Power management configuration
	PIOS_MPU9150_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, cfg->Pwr_mgmt_clk);

	// User control
	PIOS_MPU9150_SetReg(PIOS_MPU60X0_USER_CTRL_REG, cfg->User_ctl & ~0x40);

	// Digital low-pass filter and scale
	// set this before sample rate else sample rate calculation will fail
	PIOS_MPU9150_SetLPF(cfg->default_filter);

	// Sample rate
	PIOS_MPU9150_SetSampleRate(cfg->default_samplerate);

	// Set the gyro scale
	PIOS_MPU9150_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);

	// Set the accel scale
	PIOS_MPU9150_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);

	// Interrupt configuration
	PIOS_MPU9150_SetReg(PIOS_MPU60X0_INT_CFG_REG, cfg->interrupt_cfg | 0x02);

	// To enable access to the mag on auxillary i2c we must set bit 0x02 in register 0x37
	// and clear bit 0x40 in register 0x6a (PIOS_MPU60X0_USER_CTRL_REG, default condition)
	
	// Disable mag first
	if (PIOS_MPU9150_Mag_SetReg(MPU9150_MAG_CNTR, 0x00) != 0)
		return -1;
	PIOS_DELAY_WaitmS(20);
	PIOS_WDG_Clear();
	// Clear status registers
	PIOS_MPU9150_Mag_GetReg(MPU9150_MAG_STATUS);
	PIOS_MPU9150_Mag_GetReg(MPU9150_MAG_STATUS2);
	PIOS_MPU9150_Mag_GetReg(MPU9150_MAG_XH);

	// Trigger first measurement
	if (PIOS_MPU9150_Mag_SetReg(MPU9150_MAG_CNTR, 0x01) != 0)
		return -1;

	// Interrupt enable
	PIOS_MPU9150_SetReg(PIOS_MPU60X0_INT_EN_REG, cfg->interrupt_en);

	return 0;
}

/**
 * Set the gyro range and store it locally for scaling
 */
int32_t PIOS_MPU9150_SetGyroRange(enum pios_mpu60x0_range gyro_range)
{
	if (PIOS_MPU9150_SetReg(PIOS_MPU60X0_GYRO_CFG_REG, gyro_range) != 0)
		return -1;

	switch(gyro_range) {
	case PIOS_MPU60X0_SCALE_250_DEG:
		PIOS_SENSORS_SetMaxGyro(250);
		break;
	case PIOS_MPU60X0_SCALE_500_DEG:
		PIOS_SENSORS_SetMaxGyro(500);
		break;
	case PIOS_MPU60X0_SCALE_1000_DEG:
		PIOS_SENSORS_SetMaxGyro(1000);
		break;
	case PIOS_MPU60X0_SCALE_2000_DEG:
		PIOS_SENSORS_SetMaxGyro(2000);
		break;
	}

	dev->gyro_range = gyro_range;
	return 0;
}

/**
 * Set the accel range and store it locally for scaling
 */
int32_t PIOS_MPU9150_SetAccelRange(enum pios_mpu60x0_accel_range accel_range)
{
	if (PIOS_MPU9150_SetReg(PIOS_MPU60X0_ACCEL_CFG_REG, accel_range) != 0)
		return -1;
	dev->accel_range = accel_range;
	return 0;
}

/**
 * Set the sample rate in Hz by determining the nearest divisor
 * @param[in] sample rate in Hz
 */
int32_t PIOS_MPU9150_SetSampleRate(uint16_t samplerate_hz)
{
	uint16_t filter_frequency = 8000;

	if (dev->filter != PIOS_MPU60X0_LOWPASS_256_HZ)
		filter_frequency = 1000;

	// limit samplerate to filter frequency
	if (samplerate_hz > filter_frequency)
		samplerate_hz = filter_frequency;

	// calculate divisor, round to nearest integeter
	int32_t divisor = (int32_t)(((float)filter_frequency / samplerate_hz) + 0.5f) - 1;

	// limit resulting divisor to register value range
	if (divisor < 0)
		divisor = 0;

	if (divisor > 0xff)
		divisor = 0xff;

	return PIOS_MPU9150_SetReg(PIOS_MPU60X0_SMPLRT_DIV_REG, (uint8_t)divisor);
}

/**
 * Configure the digital low-pass filter
 */
void PIOS_MPU9150_SetLPF(enum pios_mpu60x0_filter filter)
{
	PIOS_MPU9150_SetReg(PIOS_MPU60X0_DLPF_CFG_REG, filter);

	dev->filter = filter;
}

/**
 * Check if an MPU9150 is detected at the requested address
 * @return 0 if detected, -1 if successfully probed but wrong id
 *  -2 no device at address
 */
int32_t PIOS_MPU9150_Probe(uint32_t i2c_id, uint8_t i2c_addr)
{
	// This function needs to set up the full transactions because
	// it should not assume anything is configured

	uint8_t mag_addr_buffer[] = {
		0,
	};
	uint8_t mag_read_buffer[] = {
		0
	};

	const struct pios_i2c_txn mag_txn_list[] = {
		{
			.info = __func__,
			.addr = MPU9150_MAG_ADDR,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(mag_addr_buffer),
			.buf = mag_addr_buffer,
		},
		{
			.info = __func__,
			.addr = MPU9150_MAG_ADDR,
			.rw = PIOS_I2C_TXN_READ,
			.len = sizeof(mag_read_buffer),
			.buf = mag_read_buffer,
		}
	};

	int32_t retval = PIOS_I2C_Transfer(i2c_id, mag_txn_list, NELEMENTS(mag_txn_list));
	if (retval < 0)
		return -1;

	return 0;
}

/**
 * @brief Reads one or more bytes into a buffer
 * \param[in] address MPU9150 register address (depends on size)
 * \param[out] buffer destination buffer
 * \param[in] len number of bytes which should be read
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_MPU9150_Read(uint8_t address, uint8_t * buffer, uint8_t len)
{
	uint8_t addr_buffer[] = {
		address,
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = dev->i2c_addr,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(addr_buffer),
			.buf = addr_buffer,
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
 * @brief Writes one or more bytes to the MPU9150
 * \param[in] address Register address
 * \param[in] buffer source buffer
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_MPU9150_Write(uint8_t address, uint8_t buffer)
{
	uint8_t data[] = {
		address,
		buffer,
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = dev->i2c_addr,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(data),
			.buf = data,
		},
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
 * @brief Read a register from MPU9150
 * @returns The register value or -1 if failure to get bus
 * @param reg[in] Register address to be read
 */
static int32_t PIOS_MPU9150_GetReg(uint8_t reg)
{
	uint8_t data;

	int32_t retval = PIOS_MPU9150_Read(reg, &data, sizeof(data));

	if (retval != 0)
		return retval;
	else
		return data;
}

/**
 * @brief Writes one byte to the MPU9150
 * \param[in] reg Register address
 * \param[in] data Byte to write
 * \return 0 if operation was successful
 * \return -1 if unable to claim SPI bus
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_MPU9150_SetReg(uint8_t reg, uint8_t data)
{
	return PIOS_MPU9150_Write(reg, data);
}

/**
 * @brief Reads one or more bytes into a buffer
 * \param[in] address MPU9150 register address (depends on size)
 * \param[out] buffer destination buffer
 * \param[in] len number of bytes which should be read
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_MPU9150_Mag_Read(uint8_t address, uint8_t * buffer, uint8_t len)
{
	uint8_t addr_buffer[] = {
		address,
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = MPU9150_MAG_ADDR,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(addr_buffer),
			.buf = addr_buffer,
		},
		{
			.info = __func__,
			.addr = MPU9150_MAG_ADDR,
			.rw = PIOS_I2C_TXN_READ,
			.len = len,
			.buf = buffer,
		}
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
 * @brief Writes one or more bytes to the MPU9150
 * \param[in] address Register address
 * \param[in] buffer source buffer
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_MPU9150_Mag_Write(uint8_t address, uint8_t buffer)
{
	uint8_t data[] = {
		address,
		buffer,
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = MPU9150_MAG_ADDR,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(data),
			.buf = data,
		},
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
 * @brief Read a register from MPU9150
 * @returns The register value or -1 if failure to get bus
 * @param reg[in] Register address to be read
 */
static int32_t PIOS_MPU9150_Mag_GetReg(uint8_t reg)
{
	uint8_t data;

	int32_t retval = PIOS_MPU9150_Mag_Read(reg, &data, sizeof(data));

	if (retval != 0)
		return retval;
	else
		return data;
}

/**
 * @brief Writes one byte to the MPU9150
 * \param[in] reg Register address
 * \param[in] data Byte to write
 * \return 0 if operation was successful
 * \return -1 if unable to claim SPI bus
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_MPU9150_Mag_SetReg(uint8_t reg, uint8_t data)
{
	return PIOS_MPU9150_Mag_Write(reg, data);
}

/*
 * @brief Read the identification bytes from the MPU9150 sensor
 * \return ID read from MPU9150 or -1 if failure
*/
static int32_t PIOS_MPU9150_ReadID()
{
	int32_t mpu9150_id = PIOS_MPU9150_GetReg(PIOS_MPU60X0_WHOAMI);
	if (mpu9150_id < 0)
		return -1;
	return mpu9150_id;
}


static float PIOS_MPU9150_GetGyroScale()
{
	switch (dev->gyro_range) {
		case PIOS_MPU60X0_SCALE_250_DEG:
			return 1.0f / 131.0f;
		case PIOS_MPU60X0_SCALE_500_DEG:
			return 1.0f / 65.5f;
		case PIOS_MPU60X0_SCALE_1000_DEG:
			return 1.0f / 32.8f;
		case PIOS_MPU60X0_SCALE_2000_DEG:
			return 1.0f / 16.4f;
	}
	return 0;
}

#if defined(PIOS_MPU6050_ACCEL)
static float PIOS_MPU9150_GetAccelScale()
{
	switch (dev->accel_range) {
		case PIOS_MPU60X0_ACCEL_2G:
			return GRAVITY / 16384.0f;
		case PIOS_MPU60X0_ACCEL_4G:
			return GRAVITY / 8192.0f;
		case PIOS_MPU60X0_ACCEL_8G:
			return GRAVITY / 4096.0f;
		case PIOS_MPU60X0_ACCEL_16G:
			return GRAVITY / 2048.0f;
	}
	return 0;
}
#endif /* PIOS_MPU6050_ACCEL */

//! Return mGa / LSB
static float PIOS_MPU9150_GetMagScale()
{
	return 3.0f; //(1229.0*10.0/4096.0)
}

/**
 * @brief Run self-test operation.
 * \return 0 if test succeeded
 * \return non-zero value if test succeeded
 */
uint8_t PIOS_MPU9150_Test(void)
{
	/* Verify that ID matches (MPU9150 ID is MPU9150_WHOAMI_ID) */
	int32_t mpu9150_id = PIOS_MPU9150_ReadID();
	if (mpu9150_id < 0)
		return -1;
	
	if (mpu9150_id != MPU9150_WHOAMI_ID)
		return -2;
	
	return 0;
}

/**
* @brief IRQ Handler.  Read all the data from onboard buffer
*/
bool PIOS_MPU9150_IRQHandler(void)
{
	if (PIOS_MPU9150_Validate(dev) != 0)
		return false;

    portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;

    xSemaphoreGiveFromISR(dev->data_ready_sema, &xHigherPriorityTaskWoken);

    return xHigherPriorityTaskWoken == pdTRUE;
}

static void PIOS_MPU9150_Task(void *parameters)
{
	while (1) {
		//Wait for data ready interrupt
		if (xSemaphoreTake(dev->data_ready_sema, portMAX_DELAY) != pdTRUE)
			continue;

		enum {
		    IDX_ACCEL_XOUT_H = 0,
		    IDX_ACCEL_XOUT_L,
		    IDX_ACCEL_YOUT_H,
		    IDX_ACCEL_YOUT_L,
		    IDX_ACCEL_ZOUT_H,
		    IDX_ACCEL_ZOUT_L,
		    IDX_TEMP_OUT_H,
		    IDX_TEMP_OUT_L,
		    IDX_GYRO_XOUT_H,
		    IDX_GYRO_XOUT_L,
		    IDX_GYRO_YOUT_H,
		    IDX_GYRO_YOUT_L,
		    IDX_GYRO_ZOUT_H,
		    IDX_GYRO_ZOUT_L,
		    BUFFER_SIZE,
		};


		uint8_t mpu9150_rec_buf[BUFFER_SIZE];

		if (PIOS_MPU9150_Read(PIOS_MPU60X0_ACCEL_X_OUT_MSB, mpu9150_rec_buf, sizeof(mpu9150_rec_buf)) < 0) {
			continue;
		}

		// Rotate the sensor to OP convention.  The datasheet defines X as towards the right
		// and Y as forward.  OP convention transposes this.  Also the Z is defined negatively
		// to our convention

		// Currently we only support rotations on top so switch X/Y accordingly
		struct pios_sensor_accel_data accel_data;
		struct pios_sensor_gyro_data gyro_data;

		switch (dev->cfg->orientation) {
		case PIOS_MPU60X0_TOP_0DEG:
			accel_data.y = (int16_t)(mpu9150_rec_buf[IDX_ACCEL_XOUT_H] << 8 | mpu9150_rec_buf[IDX_ACCEL_XOUT_L]);
			accel_data.x = (int16_t)(mpu9150_rec_buf[IDX_ACCEL_YOUT_H] << 8 | mpu9150_rec_buf[IDX_ACCEL_YOUT_L]);
			gyro_data.y  = (int16_t)(mpu9150_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu9150_rec_buf[IDX_GYRO_XOUT_L]);
			gyro_data.x  = (int16_t)(mpu9150_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu9150_rec_buf[IDX_GYRO_YOUT_L]);
			break;
		case PIOS_MPU60X0_TOP_90DEG:
			accel_data.y = - (int16_t)(mpu9150_rec_buf[IDX_ACCEL_YOUT_H] << 8 | mpu9150_rec_buf[IDX_ACCEL_YOUT_L]);
			accel_data.x = (int16_t)(mpu9150_rec_buf[IDX_ACCEL_XOUT_H] << 8 | mpu9150_rec_buf[IDX_ACCEL_XOUT_L]);
			gyro_data.y  = - (int16_t)(mpu9150_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu9150_rec_buf[IDX_GYRO_YOUT_L]);
			gyro_data.x  = (int16_t)(mpu9150_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu9150_rec_buf[IDX_GYRO_XOUT_L]);
			break;
		case PIOS_MPU60X0_TOP_180DEG:
			accel_data.y = - (int16_t)(mpu9150_rec_buf[IDX_ACCEL_XOUT_H] << 8 | mpu9150_rec_buf[IDX_ACCEL_XOUT_L]);
			accel_data.x = - (int16_t)(mpu9150_rec_buf[IDX_ACCEL_YOUT_H] << 8 | mpu9150_rec_buf[IDX_ACCEL_YOUT_L]);
			gyro_data.y  = - (int16_t)(mpu9150_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu9150_rec_buf[IDX_GYRO_XOUT_L]);
			gyro_data.x  = - (int16_t)(mpu9150_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu9150_rec_buf[IDX_GYRO_YOUT_L]);
			break;
		case PIOS_MPU60X0_TOP_270DEG:
			accel_data.y = (int16_t)(mpu9150_rec_buf[IDX_ACCEL_YOUT_H] << 8 | mpu9150_rec_buf[IDX_ACCEL_YOUT_L]);
			accel_data.x = - (int16_t)(mpu9150_rec_buf[IDX_ACCEL_XOUT_H] << 8 | mpu9150_rec_buf[IDX_ACCEL_XOUT_L]);
			gyro_data.y  = (int16_t)(mpu9150_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu9150_rec_buf[IDX_GYRO_YOUT_L]);
			gyro_data.x  = - (int16_t)(mpu9150_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu9150_rec_buf[IDX_GYRO_XOUT_L]);
			break;
		}

		gyro_data.z  = - (int16_t)(mpu9150_rec_buf[IDX_GYRO_ZOUT_H] << 8 | mpu9150_rec_buf[IDX_GYRO_ZOUT_L]);
		accel_data.z = - (int16_t)(mpu9150_rec_buf[IDX_ACCEL_ZOUT_H] << 8 | mpu9150_rec_buf[IDX_ACCEL_ZOUT_L]);

		int16_t raw_temp = (int16_t)(mpu9150_rec_buf[IDX_TEMP_OUT_H] << 8 | mpu9150_rec_buf[IDX_TEMP_OUT_L]);
		float temperature = 35.0f + ((float)raw_temp + 512.0f) / 340.0f;

		// Apply sensor scaling
		float accel_scale = PIOS_MPU9150_GetAccelScale();
		accel_data.x *= accel_scale;
		accel_data.y *= accel_scale;
		accel_data.z *= accel_scale;
		accel_data.temperature = temperature;

		float gyro_scale = PIOS_MPU9150_GetGyroScale();
		gyro_data.x *= gyro_scale;
		gyro_data.y *= gyro_scale;
		gyro_data.z *= gyro_scale;
		gyro_data.temperature = temperature;

		xQueueSendToBack(dev->accel_queue, (void *)&accel_data, 0);
		xQueueSendToBack(dev->gyro_queue, (void *)&gyro_data, 0);

		// Check for mag data ready.  Reading it clears this flag.
		if (PIOS_MPU9150_Mag_GetReg(MPU9150_MAG_STATUS) > 0) {
			struct pios_sensor_mag_data mag_data;
			uint8_t mpu9150_mag_buffer[6];
			if (PIOS_MPU9150_Mag_Read(MPU9150_MAG_XH, mpu9150_mag_buffer, sizeof(mpu9150_mag_buffer)) == 0) {
				switch(dev->cfg->orientation) {
				case PIOS_MPU60X0_TOP_0DEG:
					mag_data.x = (int16_t) (mpu9150_mag_buffer[1] << 0x08 | mpu9150_mag_buffer[0]);
					mag_data.y = (int16_t) (mpu9150_mag_buffer[3] << 0x08 | mpu9150_mag_buffer[2]);
					break;
				case PIOS_MPU60X0_TOP_90DEG:
					mag_data.y = (int16_t)  (mpu9150_mag_buffer[1] << 0x08 | mpu9150_mag_buffer[0]);
					mag_data.x = (int16_t) -(mpu9150_mag_buffer[3] << 0x08 | mpu9150_mag_buffer[2]);
					break;
				case PIOS_MPU60X0_TOP_180DEG:
					mag_data.x = (int16_t) -(mpu9150_mag_buffer[1] << 0x08 | mpu9150_mag_buffer[0]);
					mag_data.y = (int16_t) -(mpu9150_mag_buffer[3] << 0x08 | mpu9150_mag_buffer[2]);
					break;
				case PIOS_MPU60X0_TOP_270DEG:
					mag_data.y = (int16_t) -(mpu9150_mag_buffer[1] << 0x08 | mpu9150_mag_buffer[0]);
					mag_data.x = (int16_t)  (mpu9150_mag_buffer[3] << 0x08 | mpu9150_mag_buffer[2]);
					break;
				}
				mag_data.z = (int16_t) (mpu9150_mag_buffer[5] << 0x08 | mpu9150_mag_buffer[4]);

				float mag_scale = PIOS_MPU9150_GetMagScale();
				mag_data.x *= mag_scale;
				mag_data.y *= mag_scale;
				mag_data.z *= mag_scale;

				// Trigger another measurement
				PIOS_MPU9150_Mag_SetReg(MPU9150_MAG_CNTR, 0x01);

				xQueueSendToBack(dev->mag_queue, (void *) &mag_data, 0);
			}
		}

	}
}

#endif

/**
 * @}
 * @}
 */
