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
#define LSM303_TASK_STACK		(512 / 4)

#include "fifo_buffer.h"

/* Global Variables */

enum pios_lsm303_dev_magic {
	PIOS_LSM303_DEV_MAGIC = 0xef8e9e1d,
};

#define PIOS_LSM303_MAX_DOWNSAMPLE 1
struct lsm303_dev {
	uint32_t i2c_id;
	uint8_t i2c_addr_accel;
	uint8_t i2c_addr_mag;
	enum pios_lsm303_accel_range accel_range;
	enum pios_lsm303_mag_range mag_range;
	xQueueHandle queue_accel;
	xQueueHandle queue_mag;
	xTaskHandle TaskHandle;
	xSemaphoreHandle data_ready_sema;
	volatile bool configured;
	const struct pios_lsm303_cfg * cfg;
	enum pios_lsm303_dev_magic magic;
};

//! Internal representation of unscaled accelerometer data
struct pios_lsm303_accel_data {
	int16_t accel_x;
	int16_t accel_y;
	int16_t accel_z;
};

//! Internal representation of unscaled magnetometer data
struct pios_lsm303_mag_data {
	int16_t mag_x;
	int16_t mag_y;
	int16_t mag_z;
};

//! Global structure for this device device
static struct lsm303_dev * dev;

//! Private functions
static struct lsm303_dev * PIOS_LSM303_alloc(void);
static int32_t PIOS_LSM303_Validate(struct lsm303_dev * dev);
static void PIOS_LSM303_Config(struct pios_lsm303_cfg const * cfg);
static int32_t PIOS_LSM303_Accel_SetReg(uint8_t address, uint8_t buffer);
static int32_t PIOS_LSM303_Mag_SetReg(uint8_t address, uint8_t buffer);
static int32_t PIOS_LSM303_Mag_GetReg(uint8_t address);
static int32_t PIOS_LSM303_Accel_ReadData(struct pios_lsm303_accel_data * data);
static int32_t PIOS_LSM303_Mag_ReadData(struct pios_lsm303_mag_data * data);
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
	
	lsm303_dev->queue_accel = xQueueCreate(PIOS_LSM303_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_accel_data));
	if (lsm303_dev->queue_accel == NULL) {
		vPortFree(lsm303_dev);
		return NULL;
	}

	lsm303_dev->queue_mag = xQueueCreate(PIOS_LSM303_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_mag_data));
	vQueueAddToRegistry(lsm303_dev->queue_mag, (signed char*)"pios_lsm303_queue_mag");
	if (lsm303_dev->queue_mag == NULL) {
		vPortFree(lsm303_dev);
		return NULL;
	}
	
	lsm303_dev->data_ready_sema = xSemaphoreCreateMutex();
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

	PIOS_SENSORS_Register(PIOS_SENSOR_ACCEL, dev->queue_accel);
	PIOS_SENSORS_Register(PIOS_SENSOR_MAG, dev->queue_mag);

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
	/*
	 * accel
	 */

	// Reset
	while (PIOS_LSM303_Accel_SetReg(PIOS_LSM303_CTRL_REG5_A, PIOS_LSM303_CTRL5_BOOT) != 0);

	// This register enables the channels and sets the bandwidth
	while (PIOS_LSM303_Accel_SetReg(PIOS_LSM303_CTRL_REG1_A,
		PIOS_LSM303_CTRL1_400HZ |
		PIOS_LSM303_CTRL1_ZEN |
		PIOS_LSM303_CTRL1_YEN |
		PIOS_LSM303_CTRL1_XEN) != 0);

	// Disable the high pass filters
	while (PIOS_LSM303_Accel_SetReg(PIOS_LSM303_CTRL_REG2_A, 0) != 0);

	// Set INT1 to go high on data ready
	while (PIOS_LSM303_Accel_SetReg(PIOS_LSM303_CTRL_REG3_A, PIOS_LSM303_CTRL3_I1_DRDY1) != 0);

	// Set the accel scale
	PIOS_LSM303_Accel_SetRange(PIOS_LSM303_ACCEL_8G);

	// Enable FIFO
	while (PIOS_LSM303_Accel_SetReg(PIOS_LSM303_CTRL_REG5_A, PIOS_LSM303_CTRL5_FIFO_EN) != 0);

	// Fifo stream mode
	while (PIOS_LSM303_Accel_SetReg(PIOS_LSM303_FIFO_CTRL_REG_A, PIOS_LSM303_FIFO_MODE_STREAM) != 0);

	/*
	 * mag
	 */
	// set the mag bandwidth
	while (PIOS_LSM303_Mag_SetReg(PIOS_LSM303_CRA_REG_M, PIOS_LSM303_CRA_220HZ) != 0);

	// set mag to continuous operation
	while (PIOS_LSM303_Mag_SetReg(PIOS_LSM303_MR_REG_M, PIOS_LSM303_MR_CONTINUOUS) != 0);

	// Set the mag range
	PIOS_LSM303_Mag_SetRange(PIOS_LSM303_MAG_1_9GA);

	dev->configured = true;
}

/**
 * @brief Reads one or more bytes into a buffer
 * \param[in] address LSM303 register address (depends on size)
 * \param[out] buffer destination buffer
 * \param[in] len number of bytes which should be read
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_LSM303_Accel_Read(uint8_t address, uint8_t * buffer, uint8_t len)
{
	if (PIOS_LSM303_Validate(dev) != 0)
		return -1;

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
 * @brief Writes one or more bytes to the LSM303
 * \param[in] address Register address
 * \param[in] buffer source buffer
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_LSM303_Accel_Write(uint8_t address, uint8_t buffer)
{
	if (PIOS_LSM303_Validate(dev) != 0)
		return -1;

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
 * \param[in] address LSM303 register address (depends on size)
 * \param[out] buffer destination buffer
 * \param[in] len number of bytes which should be read
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_LSM303_Mag_Read(uint8_t address, uint8_t * buffer, uint8_t len)
{
	if (PIOS_LSM303_Validate(dev) != 0)
		return -1;

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
 * @brief Writes one or more bytes to the LSM303
 * \param[in] address Register address
 * \param[in] buffer source buffer
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_LSM303_Mag_Write(uint8_t address, uint8_t buffer)
{
	if (PIOS_LSM303_Validate(dev) != 0)
		return -1;

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
 * @brief Writes one byte to the LSM303
 * \param[in] reg Register address
 * \param[in] data Byte to write
 * \return 0 if operation was successful
 * \return -1 if unable to claim SPI bus
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_LSM303_Accel_SetReg(uint8_t reg, uint8_t data)
{
	return PIOS_LSM303_Accel_Write(reg, data);
}

/**
 * @brief Read a register from LSM303
 * @returns The register value or -1 if failure to get bus
 * @param reg[in] Register address to be read
 */
static int32_t PIOS_LSM303_Mag_GetReg(uint8_t reg)
{
	uint8_t data;

	int32_t retval = PIOS_LSM303_Mag_Read(reg, &data, sizeof(data));

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
static int32_t PIOS_LSM303_Mag_SetReg(uint8_t reg, uint8_t data)
{
	return PIOS_LSM303_Mag_Write(reg, data);
}

/**
 * @brief Read current X, Z, Y values (in that order)
 * \param[out] int16_t array of size 3 to store X, Z, and Y accelerometer readings
 * \returns The number of samples remaining in the fifo
 */
static int32_t PIOS_LSM303_Accel_ReadData(struct pios_lsm303_accel_data * data)
{
	if (PIOS_LSM303_Accel_Read(PIOS_LSM303_OUT_X_L_A, (uint8_t*)data, sizeof(*data)) < 0) {
		return -2;
	}
	return 0;
}

/**
 * @brief Read current X, Z, Y values (in that order)
 * \param[out] int16_t array of size 3 to store X, Y, Z and temperature magnetometer readings
 * \returns The number of samples remaining in the fifo
 */
static int32_t PIOS_LSM303_Mag_ReadData(struct pios_lsm303_mag_data * data)
{
	uint8_t temp[6];
	if (PIOS_LSM303_Mag_Read(PIOS_LSM303_OUT_X_H_M, temp, sizeof(temp)) < 0) {
		return -2;
	}

	data->mag_x = temp[0] << 8 | temp[1];
	data->mag_z = temp[2] << 8 | temp[3];
	data->mag_y = temp[4] << 8 | temp[5];

	return 0;
}

/*
 * @brief Read the identification bytes from the LSM303 sensor
 * \return ID read from LSM303 or -1 if failure
*/
int32_t PIOS_LSM303_Mag_ReadID()
{
	uint8_t id[3];
	int32_t retval = PIOS_LSM303_Mag_Read(PIOS_LSM303_IRA_REG_M, id, sizeof(id));

	if (retval != 0)
		return retval;
	else
		return (id[0] << 16 | id[1] << 8 | id[2]);
}

/**
 * Set the accel range and store it locally for scaling
 */
void PIOS_LSM303_Accel_SetRange(enum pios_lsm303_accel_range accel_range)
{
	if (PIOS_LSM303_Validate(dev) != 0)
		return;

	while (PIOS_LSM303_Accel_SetReg(PIOS_LSM303_CTRL_REG4_A, accel_range) != 0);
	dev->accel_range = accel_range;
}

/**
 * Set the mag range and store it locally for scaling
 */
void PIOS_LSM303_Mag_SetRange(enum pios_lsm303_mag_range mag_range)
{
	if (PIOS_LSM303_Validate(dev) != 0)
		return;

	while (PIOS_LSM303_Mag_SetReg(PIOS_LSM303_CRB_REG_M, mag_range) != 0);
	dev->mag_range = mag_range;
}

static float PIOS_LSM303_Accel_GetScale()
{
	// Not validating device here for efficiency

	switch (dev->accel_range) {
		case PIOS_LSM303_ACCEL_2G:
			return GRAV / (16 * 1000.0f);    //1mg/LSB, left shifted by four bits
		case PIOS_LSM303_ACCEL_4G:
			return GRAV / (16 * 500.0f);     //2mg/LSB, left shifted by four bits
		case PIOS_LSM303_ACCEL_8G:
			return GRAV / (16 * 250.0f);     //4mg/LSB, left shifted by four bits
		case PIOS_LSM303_ACCEL_16G:
			return GRAV / (16 * 250.0f / 3); //12mg/LSB, left shifted by four bits
	}
	return 0;
}

static float PIOS_LSM303_Mag_GetScaleXY()
{
	// Not validating device here for efficiency

	switch (dev->mag_range) {
		case PIOS_LSM303_MAG_1_3GA:
			return 1000.0f / 1100.0f; //[mg/LSB]
		case PIOS_LSM303_MAG_1_9GA:
			return 1000.0f / 855.0f;  //[mg/LSB]
		case PIOS_LSM303_MAG_2_5GA:
			return 1000.0f / 670.0f;  //[mg/LSB]
		case PIOS_LSM303_MAG_4_0GA:
			return 1000.0f / 450.0f;  //[mg/LSB]
		case PIOS_LSM303_MAG_4_7GA:
			return 1000.0f / 400.0f;  //[mg/LSB]
		case PIOS_LSM303_MAG_5_6GA:
			return 1000.0f / 330.0f;  //[mg/LSB]
		case PIOS_LSM303_MAG_8_1GA:
			return 1000.0f / 230.0f;  //[mg/LSB]
	}
	return 0;
}

static float PIOS_LSM303_Mag_GetScaleZ()
{
	// Not validating device here for efficiency

	switch (dev->mag_range) {
		case PIOS_LSM303_MAG_1_3GA:
			return 1000.0f / 980.0f;  //[mg/LSB]
		case PIOS_LSM303_MAG_1_9GA:
			return 1000.0f / 760.0f;  //[mg/LSB]
		case PIOS_LSM303_MAG_2_5GA:
			return 1000.0f / 600.0f;  //[mg/LSB]
		case PIOS_LSM303_MAG_4_0GA:
			return 1000.0f / 400.0f;  //[mg/LSB]
		case PIOS_LSM303_MAG_4_7GA:
			return 1000.0f / 355.0f;  //[mg/LSB]
		case PIOS_LSM303_MAG_5_6GA:
			return 1000.0f / 295.0f;  //[mg/LSB]
		case PIOS_LSM303_MAG_8_1GA:
			return 1000.0f / 205.0f;  //[mg/LSB]
	}
	return 0;
}

/**
 * @brief Run self-test operation.
 * \return 0 if test succeeded
 * \return non-zero value if test succeeded
 */
int32_t PIOS_LSM303_Accel_Test(void)
{
	struct pios_lsm303_accel_data data;
	return PIOS_LSM303_Accel_ReadData(&data);
}

/**
 * @brief Run self-test operation.
 * \return 0 if test succeeded
 * \return non-zero value if test succeeded
 */
int32_t PIOS_LSM303_Mag_Test(void)
{
	int32_t id = PIOS_LSM303_Mag_ReadID();
	if (id != 0x483433)	// "H43"
		return -1;

	struct pios_lsm303_mag_data data;
	return PIOS_LSM303_Mag_ReadData(&data);
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

static void PIOS_LSM303_Task(void *parameters)
{
	// Do not try and process sensor until the device is valid
	while (PIOS_LSM303_Validate(dev) != 0) {
		vTaskDelay(100 * portTICK_RATE_MS);
	}

	while (1)
	{
		//Wait for data ready interrupt
		if (xSemaphoreTake(dev->data_ready_sema, 5 * portTICK_RATE_MS) != pdTRUE) {
			// If this expires kick start the sensor
			struct pios_lsm303_accel_data data;
			PIOS_LSM303_Accel_ReadData(&data);
			continue;
		}

		if (!dev->configured)
			continue;
	
		/*
		 * Process accel data
		 */
		{
			struct pios_lsm303_accel_data data;
			if (PIOS_LSM303_Accel_ReadData(&data) < 0) {
				continue;
			}

			float accel_scale = PIOS_LSM303_Accel_GetScale();

			struct pios_sensor_accel_data normalized_data;
			switch (dev->cfg->orientation)
			{
				default:
				case PIOS_LSM303_TOP_0DEG:
					normalized_data.x = +data.accel_x * accel_scale;
					normalized_data.y = -data.accel_y * accel_scale;
					break;
				case PIOS_LSM303_TOP_90DEG:
					normalized_data.x = +data.accel_y * accel_scale;
					normalized_data.y = +data.accel_x * accel_scale;
					break;
				case PIOS_LSM303_TOP_180DEG:
					normalized_data.x = -data.accel_x * accel_scale;
					normalized_data.y = +data.accel_y * accel_scale;
					break;
				case PIOS_LSM303_TOP_270DEG:
					normalized_data.x = -data.accel_y * accel_scale;
					normalized_data.y = -data.accel_x * accel_scale;
					break;
			}
			normalized_data.z = -data.accel_z * accel_scale;
			normalized_data.temperature = 0;

			xQueueSend(dev->queue_accel, (void*)&normalized_data, 0);
		}

		/*
		 * Process mag data
		 */
		{
			if ((PIOS_LSM303_Mag_GetReg(PIOS_LSM303_SR_REG_M) & PIOS_LSM303_SR_DRDY) != 0)
			{
				struct pios_lsm303_mag_data data;
				if (PIOS_LSM303_Mag_ReadData(&data) < 0) {
					continue;
				}

				float mag_scale_xy = PIOS_LSM303_Mag_GetScaleXY();
				float mag_scale_z = PIOS_LSM303_Mag_GetScaleZ();
				struct pios_sensor_mag_data normalized_data;
				switch (dev->cfg->orientation)
				{
					default:
					case PIOS_LSM303_TOP_0DEG:
						normalized_data.x = +data.mag_x * mag_scale_xy;
						normalized_data.y = -data.mag_y * mag_scale_xy;
						break;
					case PIOS_LSM303_TOP_90DEG:
						normalized_data.x = +data.mag_y * mag_scale_xy;
						normalized_data.y = +data.mag_x * mag_scale_xy;
						break;
					case PIOS_LSM303_TOP_180DEG:
						normalized_data.x = -data.mag_x * mag_scale_xy;
						normalized_data.y = +data.mag_y * mag_scale_xy;
						break;
					case PIOS_LSM303_TOP_270DEG:
						normalized_data.x = -data.mag_y * mag_scale_xy;
						normalized_data.y = -data.mag_x * mag_scale_xy;
						break;
				}
				normalized_data.z = -data.mag_z * mag_scale_z;

				xQueueSend(dev->queue_mag, (void*)&normalized_data, 0);
			}
		}
	}
}

#endif

/**
 * @}
 * @}
 */
