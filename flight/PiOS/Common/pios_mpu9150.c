/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MPU9150 MPU9150 Functions
 * @brief Deals with the hardware interface to the 3-axis gyro
 * @{
 *
 * @file       pios_mpu9150.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
 * @brief      MPU9150 6-axis gyro and accel chip
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
#define MPU9150_TASK_STACK		(484 / 4)

#define MPU9150_MAG_ADDR         0x0c
#define MPU9150_MAG_STATUS       0x02
#define MPU9150_MAG_XH           0x03
#define MPU9150_MAG_XL           0x04
#define MPU9150_MAG_YH           0x05
#define MPU9150_MAG_YL           0x06
#define MPU9150_MAG_ZH           0x07
#define MPU9150_MAG_ZL           0x08
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
	bool configured;
	enum pios_mpu9150_dev_magic magic;
};

//! Global structure for this device device
static struct mpu9150_dev * dev;

//! Private functions
static struct mpu9150_dev * PIOS_MPU9150_alloc(void);
static int32_t PIOS_MPU9150_Validate(struct mpu9150_dev * dev);
static void PIOS_MPU9150_Config(struct pios_mpu60x0_cfg const * cfg);
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
	
	mpu9150_dev = (struct mpu9150_dev *)pvPortMalloc(sizeof(*mpu9150_dev));
	if (!mpu9150_dev) return (NULL);
	
	mpu9150_dev->magic = PIOS_MPU9150_DEV_MAGIC;
	
	mpu9150_dev->accel_queue = xQueueCreate(PIOS_MPU9150_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_gyro_data));
	if(mpu9150_dev->accel_queue == NULL) {
		vPortFree(mpu9150_dev);
		return NULL;
	}

	mpu9150_dev->gyro_queue = xQueueCreate(PIOS_MPU9150_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_gyro_data));
	if(mpu9150_dev->gyro_queue == NULL) {
		vPortFree(mpu9150_dev);
		return NULL;
	}

	mpu9150_dev->mag_queue = xQueueCreate(PIOS_MPU9150_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_mag_data));
	if(mpu9150_dev->mag_queue == NULL) {
		vPortFree(mpu9150_dev);
		return NULL;
	}

	mpu9150_dev->data_ready_sema = xSemaphoreCreateMutex();
	if(mpu9150_dev->data_ready_sema == NULL) {
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
 * @return 0 for success, -1 for failure
 */
int32_t PIOS_MPU9150_Init(uint32_t i2c_id, uint8_t i2c_addr, const struct pios_mpu60x0_cfg * cfg)
{
	dev = PIOS_MPU9150_alloc();
	if(dev == NULL)
		return -1;
	
	dev->i2c_id = i2c_id;
	dev->i2c_addr = i2c_addr;
	dev->cfg = cfg;

	/* Configure the MPU9150 Sensor */
	PIOS_MPU9150_Config(cfg);

	int result = xTaskCreate(PIOS_MPU9150_Task, (const signed char *)"PIOS_MPU9150_Task",
						 MPU9150_TASK_STACK, NULL, MPU9150_TASK_PRIORITY,
						 &dev->TaskHandle);
	PIOS_Assert(result == pdPASS);

	/* Set up EXTI line */
	PIOS_EXTI_Init(cfg->exti_cfg);

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
static void PIOS_MPU9150_Config(struct pios_mpu60x0_cfg const * cfg)
{
	// Reset chip
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, 0x80) != 0);
	PIOS_DELAY_WaitmS(100);
	
	// Reset chip and fifo
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_USER_CTRL_REG, 0x01 | 0x02 | 0x04) != 0);
	// Wait for reset to finish
	while (PIOS_MPU9150_GetReg(PIOS_MPU60X0_USER_CTRL_REG) & 0x07);
	
	//Power management configuration
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, cfg->Pwr_mgmt_clk) != 0) ;

	// Interrupt configuration
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_INT_CFG_REG, cfg->interrupt_cfg) != 0) ;

	// Interrupt configuration
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_INT_EN_REG, cfg->interrupt_en) != 0) ;

	// FIFO storage
#if defined(PIOS_MPU6050_ACCEL)
	// Set the accel scale
	PIOS_MPU9150_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
	
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_FIFO_EN_REG, cfg->Fifo_store | PIOS_MPU60X0_ACCEL_OUT) != 0);
#else
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_FIFO_EN_REG, cfg->Fifo_store) != 0);
#endif
	
	// Sample rate divider
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_SMPLRT_DIV_REG, cfg->Smpl_rate_div) != 0) ;
	
	// Digital low-pass filter and scale
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_DLPF_CFG_REG, cfg->filter) != 0) ;
	
	// Digital low-pass filter and scale
	PIOS_MPU9150_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);

	// Interrupt configuration
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_USER_CTRL_REG, cfg->User_ctl) != 0) ;
	
	// Interrupt configuration
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, cfg->Pwr_mgmt_clk) != 0) ;
	
	// To enable access to the mag on auxillary i2c we must set bit 0x02 in register 0x37
	// and clear bit 0x40 in register 0x6a (default condition)

	// Interrupt configuration and enable the I2C bypass mode
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_INT_CFG_REG, cfg->interrupt_cfg | 0x02) != 0) ;
	
	while (PIOS_MPU9150_Mag_SetReg(MPU9150_MAG_CNTR, 0x01) != 0) ;

	// Interrupt configuration
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_INT_EN_REG, cfg->interrupt_en) != 0) ;
	if((PIOS_MPU9150_GetReg(PIOS_MPU60X0_INT_EN_REG)) != cfg->interrupt_en)
		return;
	
	dev->configured = true;
}

/**
 * Set the gyro range and store it locally for scaling
 */
void PIOS_MPU9150_SetGyroRange(enum pios_mpu60x0_range gyro_range)
{
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_GYRO_CFG_REG, gyro_range) != 0);
	dev->gyro_range = gyro_range;
}

/**
 * Set the accel range and store it locally for scaling
 */
void PIOS_MPU9150_SetAccelRange(enum pios_mpu60x0_accel_range accel_range)
{
	while (PIOS_MPU9150_SetReg(PIOS_MPU60X0_ACCEL_CFG_REG, accel_range) != 0);
	dev->accel_range = accel_range;
}

/**
 * Check if an MPU9150 is detected at the requested address
 * @return 0 if detected, -1 if successfully probed and not there
 *  -2 if another device is there.
 */
int32_t PIOS_MPU9150_Probe(uint32_t i2c_id, uint8_t i2c_addr)
{
	uint8_t addr_buffer[] = {
		PIOS_MPU60X0_WHOAMI,
	};
	uint8_t read_buffer[1] = {
		0
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = i2c_addr,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(addr_buffer),
			.buf = addr_buffer,
		},
		{
			.info = __func__,
			.addr = i2c_addr,
			.rw = PIOS_I2C_TXN_READ,
			.len = sizeof(read_buffer),
			.buf = read_buffer,
		}
	};

	int32_t retval = PIOS_I2C_Transfer(i2c_id, txn_list, NELEMENTS(txn_list));
	if (retval < 0)
		return -2;

	if (read_buffer[0] == 0x68)
		return 0;

	return -1;
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
	if(mpu9150_id < 0)
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

/**
 * @brief Run self-test operation.
 * \return 0 if test succeeded
 * \return non-zero value if test succeeded
 */
uint8_t PIOS_MPU9150_Test(void)
{
	/* Verify that ID matches (MPU9150 ID is 0x68) */
	int32_t mpu9150_id = PIOS_MPU9150_ReadID();
	if(mpu9150_id < 0)
		return -1;
	
	if(mpu9150_id != 0x68)
		return -2;
	
	return 0;
}

/**
 * @brief Run self-test operation.
 * \return 0 if test succeeded
 * \return non-zero value if test succeeded
 */
static int32_t PIOS_MPU9150_FifoDepth(void)
{
	uint8_t mpu9150_rec_buf[2];

	if (PIOS_MPU9150_Read(PIOS_MPU60X0_FIFO_CNT_MSB, mpu9150_rec_buf, sizeof(mpu9150_rec_buf)) < 0) {
		return -1;
	}
	
	return (mpu9150_rec_buf[0] << 8) | mpu9150_rec_buf[1];
}

/**
 * @brief Get the status code.
 * \return the status code if successful
 * \return negative value if failed
 */
static int32_t PIOS_MPU9150_GetStatus(void)
{
	uint8_t mpu9150_rec_buf[1];

	if (PIOS_MPU9150_Read(PIOS_MPU60X0_INT_STATUS_REG, mpu9150_rec_buf, sizeof(mpu9150_rec_buf)) < 0) {
		return -1;
	}

	return mpu9150_rec_buf[0];
}

/**
* @brief IRQ Handler.  Read all the data from onboard buffer
*/
bool PIOS_MPU9150_IRQHandler(void)
{
    portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;

    xSemaphoreGiveFromISR(dev->data_ready_sema, &xHigherPriorityTaskWoken);

    return xHigherPriorityTaskWoken == pdTRUE;
}

static void PIOS_MPU9150_Task(void *parameters)
{
	while (1)
	{
		if (PIOS_MPU9150_Validate(dev) != 0) {
			vTaskDelay(100 * portTICK_RATE_MS);
			continue;
		}

		//Wait for data ready interrupt
		if (xSemaphoreTake(dev->data_ready_sema, portMAX_DELAY) != pdTRUE)
			continue;

		if(!dev->configured)
			continue;

		int32_t status = PIOS_MPU9150_GetStatus();
		if (status & PIOS_MPU60X0_INT_STATUS_OVERFLOW) {
			dev->configured = false;
			continue;
		}

		int32_t mpu9150_count = PIOS_MPU9150_FifoDepth();
		if(mpu9150_count < sizeof(struct pios_mpu60x0_data))
			continue;
	
		static uint8_t mpu9150_rec_buf[sizeof(struct pios_mpu60x0_data)];
		
		if (PIOS_MPU9150_Read(PIOS_MPU60X0_FIFO_REG, mpu9150_rec_buf, sizeof(mpu9150_rec_buf)) < 0) {
			continue;
		}
	
		// In the case where extras samples backed up grabbed an extra
		if (mpu9150_count >= (sizeof(mpu9150_rec_buf) * 2)) {

			//overwrite until the newest entry is present in mpu9150_rec_buf
			if (PIOS_MPU9150_Read(PIOS_MPU60X0_FIFO_REG, mpu9150_rec_buf, sizeof(mpu9150_rec_buf)) < 0) {
				continue;
			}
		}

		// Rotate the sensor to OP convention.  The datasheet defines X as towards the right
		// and Y as forward.  OP convention transposes this.  Also the Z is defined negatively
		// to our convention

#if defined(PIOS_MPU6050_ACCEL)

		// Currently we only support rotations on top so switch X/Y accordingly
		struct pios_sensor_accel_data accel_data;
		struct pios_sensor_gyro_data gyro_data;

		switch(dev->cfg->orientation) {
		case PIOS_MPU60X0_TOP_0DEG:
			accel_data.y = (int16_t) (mpu9150_rec_buf[0] << 8 | mpu9150_rec_buf[1]);    // chip X
			accel_data.x = (int16_t) (mpu9150_rec_buf[2] << 8 | mpu9150_rec_buf[3]);    // chip Y
			gyro_data.y  = (int16_t) (mpu9150_rec_buf[8] << 8  | mpu9150_rec_buf[9]);   // chip X
			gyro_data.x  = (int16_t) (mpu9150_rec_buf[10] << 8 | mpu9150_rec_buf[11]);  // chip Y
			break;
		case PIOS_MPU60X0_TOP_90DEG:
			accel_data.y = (int16_t) -(mpu9150_rec_buf[2] << 8 | mpu9150_rec_buf[3]);   // chip Y
			accel_data.x = (int16_t)  (mpu9150_rec_buf[0] << 8 | mpu9150_rec_buf[1]);   // chip X
			gyro_data.y  = (int16_t) -(mpu9150_rec_buf[10] << 8 | mpu9150_rec_buf[11]); // chip Y
			gyro_data.x  = (int16_t)  (mpu9150_rec_buf[8] << 8  | mpu9150_rec_buf[9]);  // chip X
			break;
		case PIOS_MPU60X0_TOP_180DEG:
			accel_data.y = (int16_t) -(mpu9150_rec_buf[0] << 8 | mpu9150_rec_buf[1]);   // chip X
			accel_data.x = (int16_t) -(mpu9150_rec_buf[2] << 8 | mpu9150_rec_buf[3]);   // chip Y
			gyro_data.y  = (int16_t) -(mpu9150_rec_buf[8] << 8  | mpu9150_rec_buf[9]); // chip X
			gyro_data.x  = (int16_t) -(mpu9150_rec_buf[10] << 8 | mpu9150_rec_buf[11]); // chip Y
			break;
		case PIOS_MPU60X0_TOP_270DEG:
			accel_data.y = (int16_t)  (mpu9150_rec_buf[2] << 8 | mpu9150_rec_buf[3]);   // chip Y
			accel_data.x = (int16_t) -(mpu9150_rec_buf[0] << 8 | mpu9150_rec_buf[1]);   // chip X
			gyro_data.y  = (int16_t)  (mpu9150_rec_buf[10] << 8 | mpu9150_rec_buf[11]); // chip Y
			gyro_data.x  = (int16_t) -(mpu9150_rec_buf[8] << 8  | mpu9150_rec_buf[9]); // chip X
			break;
		}
		gyro_data.z  = (int16_t) -(mpu9150_rec_buf[12] << 8 | mpu9150_rec_buf[13]);
		accel_data.z = (int16_t) -(mpu9150_rec_buf[4] << 8 | mpu9150_rec_buf[5]);

		int16_t raw_temp = mpu9150_rec_buf[6] << 8 | mpu9150_rec_buf[7];
		float temperature = 35.0f + ((float) raw_temp + 512.0f) / 340.0f;

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

		portBASE_TYPE xHigherPriorityTaskWoken_accel;
		xQueueSendToBackFromISR(dev->accel_queue, (void *) &accel_data, &xHigherPriorityTaskWoken_accel);

		portBASE_TYPE xHigherPriorityTaskWoken_gyro;
		xQueueSendToBackFromISR(dev->gyro_queue, (void *) &gyro_data, &xHigherPriorityTaskWoken_gyro);

#else

		struct pios_sensor_gyro_data gyro_data;
		switch(dev->cfg->orientation) {
		case PIOS_MPU60X0_TOP_0DEG:
			gyro_data.y  = (int16_t) (mpu9150_rec_buf[2] << 8 | mpu9150_rec_buf[3]);
			gyro_data.x  = (int16_t) (mpu9150_rec_buf[4] << 8 | mpu9150_rec_buf[5]);
			break;
		case PIOS_MPU60X0_TOP_90DEG:
			gyro_data.y  = (int16_t) -(mpu9150_rec_buf[4] << 8 | mpu9150_rec_buf[5]); // chip Y
			gyro_data.x  = (int16_t)  (mpu9150_rec_buf[2] << 8 | mpu9150_rec_buf[3]); // chip X
			break;
		case PIOS_MPU60X0_TOP_180DEG:
			gyro_data.y  = (int16_t) -(mpu9150_rec_buf[2] << 8 | mpu9150_rec_buf[3]);
			gyro_data.x  = (int16_t) -(mpu9150_rec_buf[4] << 8 | mpu9150_rec_buf[5]);
			break;
		case PIOS_MPU60X0_TOP_270DEG:
			gyro_data.y  = (int16_t)  (mpu9150_rec_buf[4] << 8 | mpu9150_rec_buf[5]); // chip Y
			gyro_data.x  = (int16_t) -(mpu9150_rec_buf[2] << 8 | mpu9150_rec_buf[3]); // chip X
			break;
		}
		gyro_data.z = (int16_t) -(mpu9150_rec_buf[6] << 8 | mpu9150_rec_buf[7]);

		int32_t raw_temp = mpu9150_rec_buf[0] << 8 | mpu9150_rec_buf[1];
		float temperature = 35.0f + ((float) raw_temp + 512.0f) / 340.0f;

		// Apply sensor scaling
		float gyro_scale = PIOS_MPU9150_GetGyroScale();
		gyro_data.x *= gyro_scale;
		gyro_data.y *= gyro_scale;
		gyro_data.z *= gyro_scale;
		gyro_data.temperature = temperature;

		portBASE_TYPE xHigherPriorityTaskWoken_gyro;
		xQueueSendToBackFromISR(dev->gyro_queue, (void *) &gyro_data, &xHigherPriorityTaskWoken_gyro);

#endif

		if (PIOS_MPU9150_Mag_GetReg(MPU9150_MAG_STATUS) > 0) {
			struct pios_sensor_mag_data mag_data;
			mag_data.x = (int16_t) (PIOS_MPU9150_Mag_GetReg(MPU9150_MAG_XH) << 0x08 | PIOS_MPU9150_Mag_GetReg(MPU9150_MAG_XL));
			mag_data.y = (int16_t) (PIOS_MPU9150_Mag_GetReg(MPU9150_MAG_YH) << 0x08 | PIOS_MPU9150_Mag_GetReg(MPU9150_MAG_YL));
			mag_data.z = (int16_t) (PIOS_MPU9150_Mag_GetReg(MPU9150_MAG_ZH) << 0x08 | PIOS_MPU9150_Mag_GetReg(MPU9150_MAG_ZL));
			PIOS_MPU9150_Mag_SetReg(MPU9150_MAG_CNTR, 0x01); // Trigger another measurement

			portBASE_TYPE xHigherPriorityTaskWoken_mag;
			xQueueSendToBackFromISR(dev->mag_queue, (void *) &mag_data, &xHigherPriorityTaskWoken_mag);
		}

	}
}

#endif

/**
 * @}
 * @}
 */
