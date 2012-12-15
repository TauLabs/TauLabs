/**
 ******************************************************************************
 * @file       pios_lsm303.c
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_LSM303 LSM303 Functions
 * @{
 * @brief LSM303 3-axis accelerometer and 3-axis magnetometer driver
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

/* Project Includes */
#include "pios.h"

#if defined(PIOS_INCLUDE_LSM303)

/* Private constants */
#define LSM303_TASK_PRIORITY	(tskIDLE_PRIORITY + configMAX_PRIORITIES - 1)	// max priority
#define LSM303_TASK_STACK		(288 / 4)

#include "fifo_buffer.h"

/* Global Variables */

enum pios_lsm303_dev_magic {
	PIOS_LSM303_DEV_MAGIC = 0xef8e9e1d,
};

#define PIOS_LSM303_MAX_DOWNSAMPLE 2
struct lsm303_dev {
	uint32_t i2c_id;
	uint8_t i2c_addr_accel;
	uint8_t i2c_addr_mag;
	xQueueHandle queue_accel;
	xTaskHandle TaskHandle;
	xSemaphoreHandle data_ready_sema;
	const struct pios_lsm303_cfg * cfg;
	enum pios_lsm303_dev_magic magic;
};

//! Global structure for this device device
static struct lsm303_dev * dev;
volatile bool lsm303_configured = false;

//! Private functions
static struct lsm303_dev * PIOS_LSM303_alloc(void);
static int32_t PIOS_LSM303_Validate(struct lsm303_dev * dev);
static void PIOS_LSM303_Config(struct pios_lsm303_cfg const * cfg);
static int32_t PIOS_LSM303_SetReg_Accel(uint8_t address, uint8_t buffer);
static int32_t PIOS_LSM303_GetReg_Accel(uint8_t address);
static int32_t PIOS_LSM303_SetReg_Mag(uint8_t address, uint8_t buffer);
static int32_t PIOS_LSM303_GetReg_Mag(uint8_t address);
static void PIOS_LSM303_Task(void *parameters);

#define GRAV 9.81f

/**
 * @brief Allocate a new device
 */
static struct lsm303_dev * PIOS_LSM303_alloc(void)
{
	struct lsm303_dev * lsm303_dev;
	
	lsm303_dev = (struct lsm303_dev *)pvPortMalloc(sizeof(*lsm303_dev));
	if (!lsm303_dev) return (NULL);
	
	lsm303_dev->magic = PIOS_LSM303_DEV_MAGIC;
	
	lsm303_dev->queue_accel = xQueueCreate(PIOS_LSM303_MAX_DOWNSAMPLE, sizeof(struct pios_lsm303_accel_data));
	vQueueAddToRegistry(lsm303_dev->queue_accel, (signed char*)"pios_lsm303_queue_accel");
	if (lsm303_dev->queue_accel == NULL) {
		vPortFree(lsm303_dev);
		return NULL;
	}
	
	lsm303_dev->data_ready_sema = xSemaphoreCreateMutex();
	vQueueAddToRegistry(lsm303_dev->data_ready_sema, (signed char*)"pios_lsm303_data_ready_sema");
	if (lsm303_dev->data_ready_sema == NULL) {
		vPortFree(lsm303_dev);
		return NULL;
	}

	return(lsm303_dev);
}

/**
 * @brief Validate the handle to the i2c device
 * @returns 0 for valid device or -1 otherwise
 */
static int32_t PIOS_LSM303_Validate(struct lsm303_dev * dev)
{
	if (dev == NULL) 
		return -1;
	if (dev->magic != PIOS_LSM303_DEV_MAGIC)
		return -2;
	if (dev->i2c_id == 0)
		return -3;
	return 0;
}

/**
 * @brief Initialize the LSM303 3-axis gyro sensor.
 * @return 0 for success, -1 for failure
 */
int32_t PIOS_LSM303_Init(uint32_t i2c_id, const struct pios_lsm303_cfg * cfg)
{
	dev = PIOS_LSM303_alloc();
	if(dev == NULL)
		return -1;
	
	dev->i2c_id = i2c_id;
	switch (cfg->devicetype)
	{
	case PIOS_LSM303DLHC_DEVICE:
		dev->i2c_addr_accel = 0x19;
		dev->i2c_addr_mag = 0x1e;
		break;
	default:
		//not implemented
		PIOS_Assert(false);
	}

	dev->cfg = cfg;

	/* Configure the LSM303 Sensor */
	PIOS_LSM303_Config(cfg);

	int result = xTaskCreate(PIOS_LSM303_Task, (const signed char *)"pios_lsm303",
						 LSM303_TASK_STACK, NULL, LSM303_TASK_PRIORITY,
						 &dev->TaskHandle);
	PIOS_Assert(result == pdPASS);

	/* Set up EXTI line */
	PIOS_EXTI_Init(cfg->exti_cfg);

	// An initial read is needed to get it running
	struct pios_lsm303_accel_data data;
	PIOS_LSM303_ReadData_Accel(&data);

	return 0;
}

/**
 * @brief Initialize the LSM303 3-axis gyro sensor
 * \return none
 * \param[in] PIOS_LSM303_ConfigTypeDef struct to be used to configure sensor.
*
*/
static void PIOS_LSM303_Config(struct pios_lsm303_cfg const * cfg)
{
	// Reset
	while (PIOS_LSM303_SetReg_Accel(PIOS_LSM303_CTRL_REG5_A, PIOS_LSM303_CTRL5_BOOT) != 0);

	// This register enables the channels and sets the bandwidth
	while (PIOS_LSM303_SetReg_Accel(PIOS_LSM303_CTRL_REG1_A,
		PIOS_LSM303_CTRL1_400HZ |
		PIOS_LSM303_CTRL1_ZEN |
		PIOS_LSM303_CTRL1_YEN |
		PIOS_LSM303_CTRL1_XEN) != 0);

	// Disable the high pass filters
	while (PIOS_LSM303_SetReg_Accel(PIOS_LSM303_CTRL_REG2_A, 0) != 0);

	// Set INT1 to go high on data ready
	while (PIOS_LSM303_SetReg_Accel(PIOS_LSM303_CTRL_REG3_A, PIOS_LSM303_CTRL3_I1_DRDY1) != 0);

	// set range
	while (PIOS_LSM303_SetReg_Accel(PIOS_LSM303_CTRL_REG4_A, cfg->accel_range) != 0);

	// Enable FIFO
	while (PIOS_LSM303_SetReg_Accel(PIOS_LSM303_CTRL_REG5_A, PIOS_LSM303_CTRL5_FIFO_EN) != 0);

	// Fifo stream mode
	while (PIOS_LSM303_SetReg_Accel(PIOS_LSM303_FIFO_CTRL_REG_A, PIOS_LSM303_FIFO_MODE_STREAM) != 0);

	lsm303_configured = true;
}

/**
 * @brief Reads one or more bytes into a buffer
 * \param[in] address HMC5883 register address (depends on size)
 * \param[out] buffer destination buffer
 * \param[in] len number of bytes which should be read
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_LSM303_Read_Accel(uint8_t address, uint8_t * buffer, uint8_t len)
{
	uint8_t addr_buffer[] = {
		len <= 1 ? address : (address | 0x80),
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = dev->i2c_addr_accel,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(addr_buffer),
			.buf = addr_buffer,
		},
		{
			.info = __func__,
			.addr = dev->i2c_addr_accel,
			.rw = PIOS_I2C_TXN_READ,
			.len = len,
			.buf = buffer,
		}
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
 * @brief Writes one or more bytes to the HMC5883
 * \param[in] address Register address
 * \param[in] buffer source buffer
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_LSM303_Write_Accel(uint8_t address, uint8_t buffer)
{
	uint8_t data[] = {
		address,
		buffer,
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = dev->i2c_addr_accel,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(data),
			.buf = data,
		},
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
 * @brief Reads one or more bytes into a buffer
 * \param[in] address HMC5883 register address (depends on size)
 * \param[out] buffer destination buffer
 * \param[in] len number of bytes which should be read
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_LSM303_Read_Mag(uint8_t address, uint8_t * buffer, uint8_t len)
{
	uint8_t addr_buffer[] = {
		len <= 1 ? address : (address | 0x80),
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = dev->i2c_addr_mag,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(addr_buffer),
			.buf = addr_buffer,
		},
		{
			.info = __func__,
			.addr = dev->i2c_addr_mag,
			.rw = PIOS_I2C_TXN_READ,
			.len = len,
			.buf = buffer,
		}
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
 * @brief Writes one or more bytes to the HMC5883
 * \param[in] address Register address
 * \param[in] buffer source buffer
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_LSM303_Write_Mag(uint8_t address, uint8_t buffer)
{
	uint8_t data[] = {
		address,
		buffer,
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = dev->i2c_addr_mag,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(data),
			.buf = data,
		},
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
 * @brief Read a register from LSM303
 * @returns The register value or -1 if failure to get bus
 * @param reg[in] Register address to be read
 */
static int32_t PIOS_LSM303_GetReg_Accel(uint8_t reg)
{
	uint8_t data;

	int32_t retval = PIOS_LSM303_Read_Accel(reg, &data, sizeof(data));

	if (retval != 0)
		return retval;
	else
		return data;
}

/**
 * @brief Writes one byte to the LSM303
 * \param[in] reg Register address
 * \param[in] data Byte to write
 * \return 0 if operation was successful
 * \return -1 if unable to claim SPI bus
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_LSM303_SetReg_Accel(uint8_t reg, uint8_t data)
{
	return PIOS_LSM303_Write_Accel(reg, data);
}

/**
 * @brief Read a register from LSM303
 * @returns The register value or -1 if failure to get bus
 * @param reg[in] Register address to be read
 */
static int32_t PIOS_LSM303_GetReg_Mag(uint8_t reg)
{
	uint8_t data;

	int32_t retval = PIOS_LSM303_Read_Mag(reg, &data, sizeof(data));

	if (retval != 0)
		return retval;
	else
		return data;
}

/**
 * @brief Writes one byte to the LSM303
 * \param[in] reg Register address
 * \param[in] data Byte to write
 * \return 0 if operation was successful
 * \return -1 if unable to claim SPI bus
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_LSM303_SetReg_Mag(uint8_t reg, uint8_t data)
{
	return PIOS_LSM303_Write_Mag(reg, data);
}

/**
 * @brief Read current X, Z, Y values (in that order)
 * \param[out] int16_t array of size 3 to store X, Z, and Y accelerometer readings
 * \returns The number of samples remaining in the fifo
 */
int32_t PIOS_LSM303_ReadData_Accel(struct pios_lsm303_accel_data * data)
{
	if (PIOS_LSM303_Read_Accel(PIOS_LSM303_OUT_X_L_A, (uint8_t*)data, sizeof(*data)) < 0) {
		return -2;
	}
	return 0;
}

/**
 * @brief Read current X, Z, Y values (in that order)
 * \param[out] int16_t array of size 3 to store X, Y, Z and temperature magnetometer readings
 * \returns The number of samples remaining in the fifo
 */
int32_t PIOS_LSM303_ReadData_Mag(struct pios_lsm303_mag_data * data)
{
	if (PIOS_LSM303_Read_Mag(PIOS_LSM303_OUT_X_H_M, (uint8_t*)data, sizeof(*data)) < 0) {
		return -2;
	}
	return 0;
}

/*
 * @brief Read the identification bytes from the LSM303 sensor
 * \return ID read from LSM303 or -1 if failure
*/
int32_t PIOS_LSM303_ReadID()
{
	return 0;
}

/**
 * \brief Reads the queue handle
 * \return Handle to the queue or null if invalid device
 */
xQueueHandle PIOS_LSM303_GetQueue_Accel()
{
	if(PIOS_LSM303_Validate(dev) != 0)
		return (xQueueHandle) NULL;
	
	return dev->queue_accel;
}


float PIOS_LSM303_GetScale_Mag()
{
/* fixme:
	switch (dev->cfg->mag_range) {
		case PIOS_LSM303_SCALE_250_DEG:
			return 1.0f / 131.0f;
		case PIOS_LSM303_SCALE_500_DEG:
			return 1.0f / 65.5f;
		case PIOS_LSM303_SCALE_1000_DEG:
			return 1.0f / 32.8f;
		case PIOS_LSM303_SCALE_2000_DEG:
			return 1.0f / 16.4f;
	}
*/
	return 0;
}

float PIOS_LSM303_GetScale_Accel()
{
	switch (dev->cfg->accel_range) {
		case PIOS_LSM303_ACCEL_2_G:
			return GRAV / 16384.0f;
		case PIOS_LSM303_ACCEL_4_G:
			return GRAV / 8192.0f;
		case PIOS_LSM303_ACCEL_8_G:
			return GRAV / 4096.0f;
		case PIOS_LSM303_ACCEL_16_G:
			return GRAV / 2048.0f;
	}
	return 0;
}

/**
 * @brief Run self-test operation.
 * \return 0 if test succeeded
 * \return non-zero value if test succeeded
 */
uint8_t PIOS_LSM303_Test_Accel(void)
{
	struct pios_lsm303_accel_data data;
	return PIOS_LSM303_ReadData_Accel(&data);
}

/**
 * @brief Run self-test operation.
 * \return 0 if test succeeded
 * \return non-zero value if test succeeded
 */
uint8_t PIOS_LSM303_Test_Mag(void)
{
	return 0;
}

/**
* @brief IRQ Handler.  Read all the data from onboard buffer
*/
bool PIOS_LSM303_IRQHandler(void)
{
    portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;

    xSemaphoreGiveFromISR(dev->data_ready_sema, &xHigherPriorityTaskWoken);

    return xHigherPriorityTaskWoken == pdTRUE;
}

void PIOS_LSM303_Task(void *parameters)
{
	while (1)
	{
		//Wait for data ready interrupt
		if (xSemaphoreTake(dev->data_ready_sema, portMAX_DELAY) != pdTRUE)
			continue;

		if (!lsm303_configured)
			continue;
	
		struct pios_lsm303_accel_data data;

		if (PIOS_LSM303_ReadData_Accel(&data) < 0) {
			continue;
		}
	
		xQueueSend(dev->queue_accel, (void *) &data, 0);
	}
}

#endif

/**
 * @}
 * @}
 */
