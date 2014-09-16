/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MPU6050 MPU6050 Functions
 * @brief Deals with the hardware interface to the 3-axis gyro
 * @{
 *
 * @file       pios_mpu6050.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      MPU6050 6-axis gyro and accel chip
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

#if defined(PIOS_INCLUDE_MPU6050)

#include "pios_semaphore.h"
#include "pios_thread.h"

/* Private constants */
#define MPU6050_TASK_PRIORITY		PIOS_THREAD_PRIO_HIGHEST
#define MPU6050_TASK_STACK_BYTES	484

/* Global Variables */

enum pios_mpu6050_dev_magic {
    PIOS_MPU6050_DEV_MAGIC = 0xf21d26a2,
};

#define PIOS_MPU6050_MAX_QUEUESIZE 2

struct mpu6050_dev {
	uint32_t i2c_id;
	uint8_t i2c_addr;
	enum pios_mpu60x0_range gyro_range;
	xQueueHandle gyro_queue;
#if defined(PIOS_MPU6050_ACCEL)
	enum pios_mpu60x0_accel_range accel_range;
	xQueueHandle accel_queue;
#endif /* PIOS_MPU6050_ACCEL */
	struct pios_thread *TaskHandle;
	struct pios_semaphore *data_ready_sema;
	const struct pios_mpu60x0_cfg *cfg;
	enum pios_mpu6050_dev_magic magic;
	enum pios_mpu60x0_filter filter;
};

//! Global structure for this device device
static struct mpu6050_dev *pios_mpu6050_dev;

//! Private functions
static struct mpu6050_dev *PIOS_MPU6050_alloc(void);
static int32_t PIOS_MPU6050_Validate(struct mpu6050_dev *dev);
static void PIOS_MPU6050_Config(const struct pios_mpu60x0_cfg *cfg);
static int32_t PIOS_MPU6050_SetReg(uint8_t address, uint8_t buffer);
static int32_t PIOS_MPU6050_GetReg(uint8_t address);
static int32_t PIOS_MPU6050_ReadID();
static void PIOS_MPU6050_Task(void *parameters);

/**
 * @brief Allocate a new device
 */
static struct mpu6050_dev *PIOS_MPU6050_alloc(void)
{
	struct mpu6050_dev *mpu6050_dev;

	mpu6050_dev = (struct mpu6050_dev *)PIOS_malloc(sizeof(*mpu6050_dev));

	if (!mpu6050_dev) return (NULL);

	mpu6050_dev->magic = PIOS_MPU6050_DEV_MAGIC;

#if defined(PIOS_MPU6050_ACCEL)
	mpu6050_dev->accel_queue = xQueueCreate(PIOS_MPU6050_MAX_QUEUESIZE, sizeof(struct pios_sensor_accel_data));

	if (mpu6050_dev->accel_queue == NULL) {
		vPortFree(mpu6050_dev);
		return NULL;
	}
#endif /* PIOS_MPU6050_ACCEL */

	mpu6050_dev->gyro_queue = xQueueCreate(PIOS_MPU6050_MAX_QUEUESIZE, sizeof(struct pios_sensor_gyro_data));

	if (mpu6050_dev->gyro_queue == NULL) {
		vPortFree(mpu6050_dev);
		return NULL;
	}

	mpu6050_dev->data_ready_sema = PIOS_Semaphore_Create();

	if (mpu6050_dev->data_ready_sema == NULL) {
		vPortFree(mpu6050_dev);
		return NULL;
	}

	return mpu6050_dev;
}

/**
 * @brief Validate the handle to the i2c device
 * @returns 0 for valid device or -1 otherwise
 */
static int32_t PIOS_MPU6050_Validate(struct mpu6050_dev *dev)
{
	if (dev == NULL)
		return -1;

	if (dev->magic != PIOS_MPU6050_DEV_MAGIC)
		return -2;

	if (dev->i2c_id == 0)
		return -3;

	return 0;
}

/**
 * @brief Initialize the MPU6050 3-axis gyro sensor.
 * @return 0 for success, -1 for failure
 */
int32_t PIOS_MPU6050_Init(uint32_t i2c_id, uint8_t i2c_addr, const struct pios_mpu60x0_cfg *cfg)
{
	pios_mpu6050_dev = PIOS_MPU6050_alloc();

	if (pios_mpu6050_dev == NULL)
		return -1;

	pios_mpu6050_dev->i2c_id = i2c_id;
	pios_mpu6050_dev->i2c_addr = i2c_addr;
	pios_mpu6050_dev->cfg = cfg;

	/* Configure the MPU6050 Sensor */
	PIOS_MPU6050_Config(cfg);

	pios_mpu6050_dev->TaskHandle = PIOS_Thread_Create(
			PIOS_MPU6050_Task, "pios_mpu6050", MPU6050_TASK_STACK_BYTES, NULL, MPU6050_TASK_PRIORITY);
	PIOS_Assert(pios_mpu6050_dev->TaskHandle != NULL);

	/* Set up EXTI line */
	PIOS_EXTI_Init(cfg->exti_cfg);

#if defined(PIOS_MPU6050_ACCEL)
	PIOS_SENSORS_Register(PIOS_SENSOR_ACCEL, pios_mpu6050_dev->accel_queue);
#endif /* PIOS_MPU6050_ACCEL */

	PIOS_SENSORS_Register(PIOS_SENSOR_GYRO, pios_mpu6050_dev->gyro_queue);

	return 0;
}

/**
 * @brief Initialize the MPU6050 3-axis gyro sensor
 * \return none
 * \param[in] PIOS_MPU6050_ConfigTypeDef struct to be used to configure sensor.
*
*/
static void PIOS_MPU6050_Config(struct pios_mpu60x0_cfg const *cfg)
{
#if defined(PIOS_MPU6050_SIMPLE_INIT_SEQUENCE)

	// Reset chip
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, PIOS_MPU60X0_PWRMGMT_IMU_RST);

	// Reset sensors signal path
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_USER_CTRL_REG, PIOS_MPU60X0_USERCTL_GYRO_RST);

	// Give chip some time to initialize
	PIOS_DELAY_WaitmS(10);

	//Power management configuration
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, cfg->Pwr_mgmt_clk);

	// User control
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_USER_CTRL_REG, cfg->User_ctl);

	// Digital low-pass filter and scale
	// set this before sample rate else sample rate calculation will fail
	PIOS_MPU6050_SetLPF(cfg->default_filter);

	// Sample rate
	PIOS_MPU6050_SetSampleRate(cfg->default_samplerate);

	// Set the gyro scale
	PIOS_MPU6050_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);

#if defined(PIOS_MPU6050_ACCEL)
	// Set the accel scale
	PIOS_MPU6050_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
#endif /* PIOS_MPU6050_ACCEL */

	// Interrupt configuration
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_INT_CFG_REG, cfg->interrupt_cfg);

	// Interrupt enable
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_INT_EN_REG, cfg->interrupt_en);

#else /* PIOS_MPU6050_SIMPLE_INIT_SEQUENCE */

	/* This init sequence should really be dropped in favor of something
	 * less redundant but it seems to be hard to get it running well
	 * on all different targets.
	 */

	// Reset chip
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, 0x80 | cfg->Pwr_mgmt_clk);
	do {
		PIOS_DELAY_WaitmS(5);
	} while (PIOS_MPU6050_GetReg(PIOS_MPU60X0_PWR_MGMT_REG) & 0x80);

	PIOS_DELAY_WaitmS(25);

	// Reset chip and fifo
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_USER_CTRL_REG, 0x80 | 0x01 | 0x02 | 0x04);
	do {
		PIOS_DELAY_WaitmS(5);
	} while (PIOS_MPU6050_GetReg(PIOS_MPU60X0_USER_CTRL_REG) & 0x07);

	PIOS_DELAY_WaitmS(25);

	//Power management configuration
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, cfg->Pwr_mgmt_clk);

	// Interrupt configuration
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_INT_CFG_REG, cfg->interrupt_cfg);

	// Interrupt enable
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_INT_EN_REG, cfg->interrupt_en);

#if defined(PIOS_MPU6050_ACCEL)
	// Set the accel scale
	PIOS_MPU6050_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
#endif

	// Digital low-pass filter and scale
	// set this before sample rate else sample rate calculation will fail
	PIOS_MPU6050_SetLPF(cfg->default_filter);

	// Sample rate
	PIOS_MPU6050_SetSampleRate(cfg->default_samplerate);

	// Set the gyro scale
	PIOS_MPU6050_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);

	// User control
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_USER_CTRL_REG, cfg->User_ctl);

	//Power management configuration
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, cfg->Pwr_mgmt_clk);

	// Interrupt configuration
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_INT_CFG_REG, cfg->interrupt_cfg);

	// Interrupt enable
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_INT_EN_REG, cfg->interrupt_en);

#endif /* PIOS_MPU6050_SIMPLE_INIT_SEQUENCE */
}

/**
 * Set the gyro range and store it locally for scaling
 */
void PIOS_MPU6050_SetGyroRange(enum pios_mpu60x0_range gyro_range)
{
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_GYRO_CFG_REG, gyro_range);

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

	pios_mpu6050_dev->gyro_range = gyro_range;
}

/**
 * Set the accel range and store it locally for scaling
 */
#if defined(PIOS_MPU6050_ACCEL)
void PIOS_MPU6050_SetAccelRange(enum pios_mpu60x0_accel_range accel_range)
{
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_ACCEL_CFG_REG, accel_range);

	pios_mpu6050_dev->accel_range = accel_range;
}
#endif /* PIOS_MPU6050_ACCEL */

/**
 * Set the sample rate in Hz by determining the nearest divisor
 * @param[in] sample rate in Hz
 */
void PIOS_MPU6050_SetSampleRate(uint16_t samplerate_hz)
{
	uint16_t filter_frequency = 8000;

	if (pios_mpu6050_dev->filter != PIOS_MPU60X0_LOWPASS_256_HZ)
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

	PIOS_MPU6050_SetReg(PIOS_MPU60X0_SMPLRT_DIV_REG, (uint8_t)divisor);
}

/**
 * Set the MPU6050 to act as a pass-through
 */
void PIOS_MPU6050_SetPassThrough(bool passThrough)
{
	int32_t int_cfg_reg = PIOS_MPU6050_GetReg(PIOS_MPU60X0_INT_CFG_REG);
	
	if(passThrough)
		PIOS_MPU6050_SetReg(PIOS_MPU60X0_INT_CFG_REG, int_cfg_reg | (1 << 1));	// Set bit1 (I2C_BYPASS_EN)
	else
		PIOS_MPU6050_SetReg(PIOS_MPU60X0_INT_CFG_REG, int_cfg_reg & ~(1 << 1));	// Clear bit1 (I2C_BYPASS_EN)
		
	int32_t user_ctrl_reg = PIOS_MPU6050_GetReg(PIOS_MPU60X0_USER_CTRL_REG);	// USER_CTRL
	
	if(passThrough)
		PIOS_MPU6050_SetReg(PIOS_MPU60X0_USER_CTRL_REG, user_ctrl_reg & ~(1 << 5));	// Clear bit1 (I2C_MST_EN)
	else
		PIOS_MPU6050_SetReg(PIOS_MPU60X0_USER_CTRL_REG, user_ctrl_reg | (1 << 5));	// Set bit1 (I2C_MST_EN)
}

/**
 * Configure the digital low-pass filter
 */
void PIOS_MPU6050_SetLPF(enum pios_mpu60x0_filter filter)
{
	PIOS_MPU6050_SetReg(PIOS_MPU60X0_DLPF_CFG_REG, filter);

	pios_mpu6050_dev->filter = filter;
}

/**
 * Check if an MPU6050 is detected at the requested address
 * @return 0 if detected, -1 if successfully probed but wrong id
 *  -2 no device at address
 */
int32_t PIOS_MPU6050_Probe(uint32_t i2c_id, uint8_t i2c_addr)
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
 * \param[in] address MPU6050 register address (depends on size)
 * \param[out] buffer destination buffer
 * \param[in] len number of bytes which should be read
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_MPU6050_Read(uint8_t address, uint8_t *buffer, uint8_t len)
{
	uint8_t addr_buffer[] = {
		address,
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = pios_mpu6050_dev->i2c_addr,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(addr_buffer),
			.buf = addr_buffer,
		},
		{
			.info = __func__,
			.addr = pios_mpu6050_dev->i2c_addr,
			.rw = PIOS_I2C_TXN_READ,
			.len = len,
			.buf = buffer,
		}
	};

	return PIOS_I2C_Transfer(pios_mpu6050_dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
 * @brief Writes one or more bytes to the MPU6050
 * \param[in] address Register address
 * \param[in] buffer source buffer
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_MPU6050_Write(uint8_t address, uint8_t buffer)
{
	uint8_t data[] = {
		address,
		buffer,
	};

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = pios_mpu6050_dev->i2c_addr,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = sizeof(data),
			.buf = data,
		},
	};

	return PIOS_I2C_Transfer(pios_mpu6050_dev->i2c_id, txn_list, NELEMENTS(txn_list));
}

/**
 * @brief Read a register from MPU6050
 * @returns The register value or -1 if failure to get bus
 * @param reg[in] Register address to be read
 */
static int32_t PIOS_MPU6050_GetReg(uint8_t reg)
{
	uint8_t data;

	int32_t retval = PIOS_MPU6050_Read(reg, &data, sizeof(data));

	if (retval != 0)
		return retval;
	else
		return data;
}

/**
 * @brief Writes one byte to the MPU6050
 * \param[in] reg Register address
 * \param[in] data Byte to write
 * \return 0 if operation was successful
 * \return -1 if unable to claim SPI bus
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_MPU6050_SetReg(uint8_t reg, uint8_t data)
{
	return PIOS_MPU6050_Write(reg, data);
}

/*
 * @brief Read the identification bytes from the MPU6050 sensor
 * \return ID read from MPU6050 or -1 if failure
*/
static int32_t PIOS_MPU6050_ReadID()
{
	int32_t mpu6050_id = PIOS_MPU6050_GetReg(PIOS_MPU60X0_WHOAMI);

	if (mpu6050_id < 0)
		return -1;

	return mpu6050_id;
}


static float PIOS_MPU6050_GetGyroScale()
{
	switch (pios_mpu6050_dev->gyro_range) {
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
static float PIOS_MPU6050_GetAccelScale()
{
	switch (pios_mpu6050_dev->accel_range) {
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
uint8_t PIOS_MPU6050_Test(void)
{
	/* Verify that ID matches (MPU6050 ID is 0x68) */
	int32_t mpu6050_id = PIOS_MPU6050_ReadID();

	if (mpu6050_id < 0)
		return -1;

	if (mpu6050_id != 0x68)
		return -2;

	return 0;
}

/**
* @brief IRQ Handler.  Read all the data from onboard buffer
*/
bool PIOS_MPU6050_IRQHandler(void)
{
	if (PIOS_MPU6050_Validate(pios_mpu6050_dev) != 0)
		return false;

	bool woken = false;

	PIOS_Semaphore_Give_FromISR(pios_mpu6050_dev->data_ready_sema, &woken);

	return woken;
}

static void PIOS_MPU6050_Task(void *parameters)
{
	while (1) {
		//Wait for data ready interrupt
		if (PIOS_Semaphore_Take(pios_mpu6050_dev->data_ready_sema, PIOS_SEMAPHORE_TIMEOUT_MAX) != true)
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

		uint8_t mpu6050_rec_buf[BUFFER_SIZE];

		if (PIOS_MPU6050_Read(PIOS_MPU60X0_ACCEL_X_OUT_MSB, mpu6050_rec_buf, sizeof(mpu6050_rec_buf)) < 0) {
			continue;
		}

		// Rotate the sensor to OP convention.  The datasheet defines X as towards the right
		// and Y as forward.  OP convention transposes this.  Also the Z is defined negatively
		// to our convention

#if defined(PIOS_MPU6050_ACCEL)

		// Currently we only support rotations on top so switch X/Y accordingly
		struct pios_sensor_accel_data accel_data;
		struct pios_sensor_gyro_data gyro_data;

		switch (pios_mpu6050_dev->cfg->orientation) {
		case PIOS_MPU60X0_TOP_0DEG:
			accel_data.y = (int16_t)(mpu6050_rec_buf[IDX_ACCEL_XOUT_H] << 8 | mpu6050_rec_buf[IDX_ACCEL_XOUT_L]);
			accel_data.x = (int16_t)(mpu6050_rec_buf[IDX_ACCEL_YOUT_H] << 8 | mpu6050_rec_buf[IDX_ACCEL_YOUT_L]);
			gyro_data.y  = (int16_t)(mpu6050_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_XOUT_L]);
			gyro_data.x  = (int16_t)(mpu6050_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_YOUT_L]);
			break;
		case PIOS_MPU60X0_TOP_90DEG:
			accel_data.y = - (int16_t)(mpu6050_rec_buf[IDX_ACCEL_YOUT_H] << 8 | mpu6050_rec_buf[IDX_ACCEL_YOUT_L]);
			accel_data.x = (int16_t)(mpu6050_rec_buf[IDX_ACCEL_XOUT_H] << 8 | mpu6050_rec_buf[IDX_ACCEL_XOUT_L]);
			gyro_data.y  = - (int16_t)(mpu6050_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_YOUT_L]);
			gyro_data.x  = (int16_t)(mpu6050_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_XOUT_L]);
			break;
		case PIOS_MPU60X0_TOP_180DEG:
			accel_data.y = - (int16_t)(mpu6050_rec_buf[IDX_ACCEL_XOUT_H] << 8 | mpu6050_rec_buf[IDX_ACCEL_XOUT_L]);
			accel_data.x = - (int16_t)(mpu6050_rec_buf[IDX_ACCEL_YOUT_H] << 8 | mpu6050_rec_buf[IDX_ACCEL_YOUT_L]);
			gyro_data.y  = - (int16_t)(mpu6050_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_XOUT_L]);
			gyro_data.x  = - (int16_t)(mpu6050_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_YOUT_L]);
			break;
		case PIOS_MPU60X0_TOP_270DEG:
			accel_data.y = (int16_t)(mpu6050_rec_buf[IDX_ACCEL_YOUT_H] << 8 | mpu6050_rec_buf[IDX_ACCEL_YOUT_L]);
			accel_data.x = - (int16_t)(mpu6050_rec_buf[IDX_ACCEL_XOUT_H] << 8 | mpu6050_rec_buf[IDX_ACCEL_XOUT_L]);
			gyro_data.y  = (int16_t)(mpu6050_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_YOUT_L]);
			gyro_data.x  = - (int16_t)(mpu6050_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_XOUT_L]);
			break;
		}

		gyro_data.z  = - (int16_t)(mpu6050_rec_buf[IDX_GYRO_ZOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_ZOUT_L]);
		accel_data.z = - (int16_t)(mpu6050_rec_buf[IDX_ACCEL_ZOUT_H] << 8 | mpu6050_rec_buf[IDX_ACCEL_ZOUT_L]);

		int16_t raw_temp = (int16_t)(mpu6050_rec_buf[IDX_TEMP_OUT_H] << 8 | mpu6050_rec_buf[IDX_TEMP_OUT_L]);
		float temperature = 35.0f + ((float)raw_temp + 512.0f) / 340.0f;

		// Apply sensor scaling
		float accel_scale = PIOS_MPU6050_GetAccelScale();
		accel_data.x *= accel_scale;
		accel_data.y *= accel_scale;
		accel_data.z *= accel_scale;
		accel_data.temperature = temperature;

		float gyro_scale = PIOS_MPU6050_GetGyroScale();
		gyro_data.x *= gyro_scale;
		gyro_data.y *= gyro_scale;
		gyro_data.z *= gyro_scale;
		gyro_data.temperature = temperature;

		xQueueSendToBack(pios_mpu6050_dev->accel_queue, (void *)&accel_data, 0);

		xQueueSendToBack(pios_mpu6050_dev->gyro_queue, (void *)&gyro_data, 0);

#else

		struct pios_sensor_gyro_data gyro_data;

		switch (pios_mpu6050_dev->cfg->orientation) {
		case PIOS_MPU60X0_TOP_0DEG:
			gyro_data.y  = (int16_t)(mpu6050_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_XOUT_L]);
			gyro_data.x  = (int16_t)(mpu6050_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_YOUT_L]);
			break;
		case PIOS_MPU60X0_TOP_90DEG:
			gyro_data.y  = - (int16_t)(mpu6050_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_YOUT_L]);
			gyro_data.x  = (int16_t)(mpu6050_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_XOUT_L]);
			break;
		case PIOS_MPU60X0_TOP_180DEG:
			gyro_data.y  = - (int16_t)(mpu6050_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_XOUT_L]);
			gyro_data.x  = - (int16_t)(mpu6050_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_YOUT_L]);
			break;
		case PIOS_MPU60X0_TOP_270DEG:
			gyro_data.y  = (int16_t)(mpu6050_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_YOUT_L]);
			gyro_data.x  = - (int16_t)(mpu6050_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_XOUT_L]);
			break;
		}

		gyro_data.z = - (int16_t)(mpu6050_rec_buf[IDX_GYRO_ZOUT_H] << 8 | mpu6050_rec_buf[IDX_GYRO_ZOUT_L]);

		int32_t raw_temp = (int16_t)(mpu6050_rec_buf[IDX_TEMP_OUT_H] << 8 | mpu6050_rec_buf[IDX_TEMP_OUT_L]);
		float temperature = 35.0f + ((float)raw_temp + 512.0f) / 340.0f;

		// Apply sensor scaling
		float gyro_scale = PIOS_MPU6050_GetGyroScale();
		gyro_data.x *= gyro_scale;
		gyro_data.y *= gyro_scale;
		gyro_data.z *= gyro_scale;
		gyro_data.temperature = temperature;

		xQueueSendToBack(pios_mpu6050_dev->gyro_queue, (void *)&gyro_data, 0);

#endif /* PIOS_MPU6050_ACCEL */
	}
}

#endif

/**
 * @}
 * @}
 */
