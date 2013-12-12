/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_HMC5883 HMC5883 Functions
 * @brief Deals with the hardware interface to the magnetometers
 * @{
 * @file       pios_hmc5883.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      HMC5883 Magnetic Sensor Functions from AHRS
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
#include "pios_hmc5883_priv.h"

#if defined(PIOS_INCLUDE_HMC5883)

#include "pios_semaphore.h"

/* Private constants */
#define HMC5883_TASK_PRIORITY        (tskIDLE_PRIORITY + configMAX_PRIORITIES - 1)  // max priority
#define HMC5883_TASK_STACK	         (512 / 4)
#define PIOS_HMC5883_MAX_DOWNSAMPLE  1

/* Global Variables */

/* Local Types */
enum pios_hmc5883_dev_magic {
	PIOS_HMC5883_DEV_MAGIC = 0x3d8e17ab,
};

struct hmc5883_dev {
	uint32_t i2c_id;
	const struct pios_hmc5883_cfg *cfg;
	xQueueHandle queue;
	xTaskHandle task;
	struct pios_semaphore *data_ready_sema;
	enum pios_hmc5883_dev_magic magic;
	enum pios_hmc5883_orientation orientation;
};

/* Local Variables */
static int32_t PIOS_HMC5883_Config(const struct pios_hmc5883_cfg * cfg);
static int32_t PIOS_HMC5883_Read(uint8_t address, uint8_t * buffer, uint8_t len);
static int32_t PIOS_HMC5883_Write(uint8_t address, uint8_t buffer);
static void PIOS_HMC5883_Task(void *parameters);

static struct hmc5883_dev *dev;

/**
 * @brief Allocate a new device
 */
static struct hmc5883_dev * PIOS_HMC5883_alloc(void)
{
	struct hmc5883_dev *hmc5883_dev;
	
	hmc5883_dev = (struct hmc5883_dev *)PIOS_malloc(sizeof(*hmc5883_dev));
	if (!hmc5883_dev) return (NULL);
	
	hmc5883_dev->magic = PIOS_HMC5883_DEV_MAGIC;
	
	hmc5883_dev->queue = xQueueCreate(PIOS_HMC5883_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_mag_data));
	if (hmc5883_dev->queue == NULL) {
		vPortFree(hmc5883_dev);
		return NULL;
	}

	return(hmc5883_dev);
}

/**
 * @brief Validate the handle to the i2c device
 * @returns 0 for valid device or -1 otherwise
 */
static int32_t PIOS_HMC5883_Validate(struct hmc5883_dev *dev)
{
	if (dev == NULL) 
		return -1;
	if (dev->magic != PIOS_HMC5883_DEV_MAGIC)
		return -2;
	if (dev->i2c_id == 0)
		return -3;
	return 0;
}

/**
 * @brief Initialize the HMC5883 magnetometer sensor.
 * @return 0 on success
 */
int32_t PIOS_HMC5883_Init(uint32_t i2c_id, const struct pios_hmc5883_cfg *cfg)
{
	dev = (struct hmc5883_dev *) PIOS_HMC5883_alloc();
	if (dev == NULL)
		return -1;

	dev->cfg = cfg;
	dev->i2c_id = i2c_id;
	dev->orientation = cfg->Default_Orientation;

	/* check if we are using an irq line */
	if (cfg->exti_cfg != NULL) {
		PIOS_EXTI_Init(cfg->exti_cfg);

		dev->data_ready_sema = PIOS_Semaphore_Create();
		PIOS_Assert(dev->data_ready_sema != NULL);
	}

	if (PIOS_HMC5883_Config(cfg) != 0)
		return -2;

	PIOS_SENSORS_Register(PIOS_SENSOR_MAG, dev->queue);

	int result = xTaskCreate(PIOS_HMC5883_Task, (const signed char *)"pios_hmc5883",
						 HMC5883_TASK_STACK, NULL, HMC5883_TASK_PRIORITY,
						 &dev->task);

	PIOS_Assert(result == pdPASS);

	return 0;
}

/**
 * @brief Updates the HMC5883 chip orientation.
 * @returns 0 for success or -1 for failure
 */
int32_t PIOS_HMC5883_SetOrientation(enum pios_hmc5883_orientation orientation)
{
	if (PIOS_HMC5883_Validate(dev) != 0)
		return -1;

	dev->orientation = orientation;

	return 0;
}

/**
 * @brief Initialize the HMC5883 magnetometer sensor
 * \return none
 * \param[in] PIOS_HMC5883_ConfigTypeDef struct to be used to configure sensor.
 *
 * CTRL_REGA: Control Register A
 * Read Write
 * Default value: 0x10
 * 7:5  0   These bits must be cleared for correct operation.
 * 4:2 DO2-DO0: Data Output Rate Bits
 *             DO2 |  DO1 |  DO0 |   Minimum Data Output Rate (Hz)
 *            ------------------------------------------------------
 *              0  |  0   |  0   |            0.75
 *              0  |  0   |  1   |            1.5
 *              0  |  1   |  0   |            3
 *              0  |  1   |  1   |            7.5
 *              1  |  0   |  0   |           15 (default)
 *              1  |  0   |  1   |           30
 *              1  |  1   |  0   |           75
 *              1  |  1   |  1   |           Not Used
 * 1:0 MS1-MS0: Measurement Configuration Bits
 *             MS1 | MS0 |   MODE
 *            ------------------------------
 *              0  |  0   |  Normal
 *              0  |  1   |  Positive Bias
 *              1  |  0   |  Negative Bias
 *              1  |  1   |  Not Used
 *
 * CTRL_REGB: Control RegisterB
 * Read Write
 * Default value: 0x20
 * 7:5 GN2-GN0: Gain Configuration Bits.
 *             GN2 |  GN1 |  GN0 |   Mag Input   | Gain       | Output Range
 *                 |      |      |  Range[Ga]    | [LSB/mGa]  |
 *            ------------------------------------------------------
 *              0  |  0   |  0   |  ±0.88Ga      |   1370     | 0xF8000x07FF (-2048:2047)
 *              0  |  0   |  1   |  ±1.3Ga (def) |   1090     | 0xF8000x07FF (-2048:2047)
 *              0  |  1   |  0   |  ±1.9Ga       |   820      | 0xF8000x07FF (-2048:2047)
 *              0  |  1   |  1   |  ±2.5Ga       |   660      | 0xF8000x07FF (-2048:2047)
 *              1  |  0   |  0   |  ±4.0Ga       |   440      | 0xF8000x07FF (-2048:2047)
 *              1  |  0   |  1   |  ±4.7Ga       |   390      | 0xF8000x07FF (-2048:2047)
 *              1  |  1   |  0   |  ±5.6Ga       |   330      | 0xF8000x07FF (-2048:2047)
 *              1  |  1   |  1   |  ±8.1Ga       |   230      | 0xF8000x07FF (-2048:2047)
 *                               |Not recommended|
 *
 * 4:0 CRB4-CRB: 0 This bit must be cleared for correct operation.
 *
 * _MODE_REG: Mode Register
 * Read Write
 * Default value: 0x02
 * 7:2  0   These bits must be cleared for correct operation.
 * 1:0 MD1-MD0: Mode Select Bits
 *             MS1 | MS0 |   MODE
 *            ------------------------------
 *              0  |  0   |  Continuous-Conversion Mode.
 *              0  |  1   |  Single-Conversion Mode
 *              1  |  0   |  Negative Bias
 *              1  |  1   |  Sleep Mode
 */
static int32_t PIOS_HMC5883_Config(const struct pios_hmc5883_cfg * cfg)
{
	// CRTL_REGA
	if (PIOS_HMC5883_Write(PIOS_HMC5883_CONFIG_REG_A, cfg->M_ODR | cfg->Meas_Conf) != 0)
		return -1;
	
	// CRTL_REGB
	if (PIOS_HMC5883_Write(PIOS_HMC5883_CONFIG_REG_B, cfg->Gain) != 0)
		return -1;
	
	// Mode register
	if (PIOS_HMC5883_Write(PIOS_HMC5883_MODE_REG, cfg->Mode) != 0)
		return -1;
	
	return 0;
}

/**
 * Get the mag sensitivity based on the active settings
 * @returns Sensitivity in LSB / Ga
 */
static uint16_t PIOS_HMC5883_Config_GetSensitivity()
{
	switch (dev->cfg->Gain) {
	case PIOS_HMC5883_GAIN_0_88:
		return PIOS_HMC5883_Sensitivity_0_88Ga;
	case PIOS_HMC5883_GAIN_1_3:
		return PIOS_HMC5883_Sensitivity_1_3Ga;
	case PIOS_HMC5883_GAIN_1_9:
		return PIOS_HMC5883_Sensitivity_1_9Ga;
	case PIOS_HMC5883_GAIN_2_5:
		return PIOS_HMC5883_Sensitivity_2_5Ga;
	case PIOS_HMC5883_GAIN_4_0:
		return PIOS_HMC5883_Sensitivity_4_0Ga;
	case PIOS_HMC5883_GAIN_4_7:
		return PIOS_HMC5883_Sensitivity_4_7Ga;
	case PIOS_HMC5883_GAIN_5_6:
		return PIOS_HMC5883_Sensitivity_5_6Ga;
	case PIOS_HMC5883_GAIN_8_1:
		return PIOS_HMC5883_Sensitivity_8_1Ga;
	}

	return 0;
}

/**
 * @brief Read current X, Z, Y values (in that order)
 * \param[out] int16_t array of size 3 to store X, Z, and Y magnetometer readings
 * \return 0 for success or -1 for failure
 */
static int32_t PIOS_HMC5883_ReadMag(struct pios_sensor_mag_data *mag_data)
{
	if (PIOS_HMC5883_Validate(dev) != 0)
		return -1;

	/* don't use PIOS_HMC5883_Read and PIOS_HMC5883_Write here because the task could be
	 * switched out of context in between which would give the sensor less time to capture
	 * the next sample.
	 */
	uint8_t addr_read = PIOS_HMC5883_DATAOUT_XMSB_REG;
	uint8_t buffer_read[6];

	// PIOS_HMC5883_MODE_CONTINUOUS: This should not be necessary but for some reason it is coming out of continuous conversion mode
	// PIOS_HMC5883_MODE_SINGLE: This triggers the next measurement
	uint8_t buffer_write[2] = {
		PIOS_HMC5883_MODE_REG,
		dev->cfg->Mode
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = PIOS_HMC5883_I2C_ADDR,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(addr_read),
			.buf = &addr_read,
		},
		{
			.info = __func__,
			.addr = PIOS_HMC5883_I2C_ADDR,
			.rw = PIOS_I2C_TXN_READ,
			.len = sizeof(buffer_read),
			.buf = buffer_read,
		},
		{
			.info = __func__,
			.addr = PIOS_HMC5883_I2C_ADDR,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(buffer_write),
			.buf = buffer_write,
		},
	};

	if (PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list)) != 0)
		return -1;

	int16_t mag_x, mag_y, mag_z;
	uint16_t sensitivity = PIOS_HMC5883_Config_GetSensitivity();
	mag_x = ((int16_t) ((uint16_t)buffer_read[0] << 8) + buffer_read[1]) * 1000 / sensitivity;
	mag_z = ((int16_t) ((uint16_t)buffer_read[2] << 8) + buffer_read[3]) * 1000 / sensitivity;
	mag_y = ((int16_t) ((uint16_t)buffer_read[4] << 8) + buffer_read[5]) * 1000 / sensitivity;

	// Define "0" when the fiducial is in the front left of the board
	switch (dev->orientation) {
		case PIOS_HMC5883_TOP_0DEG:
			mag_data->x = -mag_x;
			mag_data->y = mag_y;
			mag_data->z = -mag_z;
			break;
		case PIOS_HMC5883_TOP_90DEG:
			mag_data->x = -mag_y;
			mag_data->y = -mag_x;
			mag_data->z = -mag_z;
			break;
		case PIOS_HMC5883_TOP_180DEG:
			mag_data->x = mag_x;
			mag_data->y = -mag_y;
			mag_data->z = -mag_z;
			break;
		case PIOS_HMC5883_TOP_270DEG:
			mag_data->x = mag_y;
			mag_data->y = mag_x;
			mag_data->z = -mag_z;
			break;
		case PIOS_HMC5883_BOTTOM_0DEG:
			mag_data->x = -mag_x;
			mag_data->y = -mag_y;
			mag_data->z = mag_z;
			break;
		case PIOS_HMC5883_BOTTOM_90DEG:
			mag_data->x = -mag_y;
			mag_data->y = mag_x;
			mag_data->z = mag_z;
			break;
		case PIOS_HMC5883_BOTTOM_180DEG:
			mag_data->x = mag_x;
			mag_data->y = mag_y;
			mag_data->z = mag_z;
			break;
		case PIOS_HMC5883_BOTTOM_270DEG:
			mag_data->x = mag_y;
			mag_data->y = -mag_x;
			mag_data->z = mag_z;
			break;
	}
	
	return 0;
}


/**
 * @brief Read the identification bytes from the HMC5883 sensor
 * \param[out] uint8_t array of size 4 to store HMC5883 ID.
 * \return 0 if successful, -1 if not
 */
static uint8_t PIOS_HMC5883_ReadID(uint8_t out[4])
{
	uint8_t retval = PIOS_HMC5883_Read(PIOS_HMC5883_DATAOUT_IDA_REG, out, 3);
	out[3] = '\0';
	return retval;
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
static int32_t PIOS_HMC5883_Read(uint8_t address, uint8_t * buffer, uint8_t len)
{
	if(PIOS_HMC5883_Validate(dev) != 0)
		return -1;

	uint8_t addr_buffer[] = {
		address,
	};
	
	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = PIOS_HMC5883_I2C_ADDR,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(addr_buffer),
			.buf = addr_buffer,
		}
		,
		{
			.info = __func__,
			.addr = PIOS_HMC5883_I2C_ADDR,
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
static int32_t PIOS_HMC5883_Write(uint8_t address, uint8_t buffer)
{
	if(PIOS_HMC5883_Validate(dev) != 0)
		return -1;

	uint8_t data[] = {
		address,
		buffer,
	};
	
	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = PIOS_HMC5883_I2C_ADDR,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(data),
			.buf = data,
		}
		,
	};

	return PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
 * @brief Run self-test operation.  Do not call this during operational use!!
 * \return 0 if success, -1 if test failed
 */
int32_t PIOS_HMC5883_Test(void)
{
	/* Verify that ID matches (HMC5883 ID is null-terminated ASCII string "H43") */
	char id[4];
	PIOS_HMC5883_ReadID((uint8_t *)id);
	if((id[0] != 'H') || (id[1] != '4') || (id[2] != '3')) // Expect H43
		return -1;

	return 0;
}

/**
 * @brief IRQ Handler
 */
bool PIOS_HMC5883_IRQHandler(void)
{
	if (PIOS_HMC5883_Validate(dev) != 0)
		return false;

	bool woken = false;
	PIOS_Semaphore_Give_FromISR(dev->data_ready_sema, &woken);

	return woken;
}

/**
 * The HMC5883 task
 */
static void PIOS_HMC5883_Task(void *parameters)
{
	while (PIOS_HMC5883_Validate(dev) != 0) {
		vTaskDelay(MS2TICKS(100));
	}

	portTickType sample_delay;

	switch (dev->cfg->M_ODR) {
	case PIOS_HMC5883_ODR_0_75:
		sample_delay = MS2TICKS(1000 / 0.75f) + 0.99999f;
		break;
	case PIOS_HMC5883_ODR_1_5:
		sample_delay = MS2TICKS(1000 / 1.5f) + 0.99999f;
		break;
	case PIOS_HMC5883_ODR_3:
		sample_delay = MS2TICKS(1000 / 3.0f) + 0.99999f;
		break;
	case PIOS_HMC5883_ODR_7_5:
		sample_delay = MS2TICKS(1000 / 7.5f) + 0.99999f;
		break;
	case PIOS_HMC5883_ODR_15:
		sample_delay = MS2TICKS(1000 / 15.0f) + 0.99999f;
		break;
	case PIOS_HMC5883_ODR_30:
		sample_delay = MS2TICKS(1000 / 30.0f) + 0.99999f;
		break;
	case PIOS_HMC5883_ODR_75:
	default:
		sample_delay = MS2TICKS(1000 / 75.0f) + 0.99999f;
		break;
	}

	portTickType now = xTaskGetTickCount();

	while (1) {
		if (dev->cfg->Mode == PIOS_HMC5883_MODE_CONTINUOUS) {
			if (PIOS_Semaphore_Take(dev->data_ready_sema, PIOS_SEMAPHORE_TIMEOUT_MAX) != true) {
				vTaskDelay(MS2TICKS(100));
				continue;
			}
		} else {
			vTaskDelayUntil(&now, sample_delay);
		}

		struct pios_sensor_mag_data mag_data;
		if (PIOS_HMC5883_ReadMag(&mag_data) == 0)
			xQueueSend(dev->queue, (void *) &mag_data, 0);
	}
}

#endif /* PIOS_INCLUDE_HMC5883 */

/**
 * @}
 * @}
 */
