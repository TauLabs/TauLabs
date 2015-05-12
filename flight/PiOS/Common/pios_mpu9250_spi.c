/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MPU9250 MPU9250 Functions
 * @brief Deals with the hardware interface to the 3-axis gyro
 * @{
 *
 * @file       pios_mpu9250.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      MPU9250 9-axis gyro accel and mag chip
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

#if defined(PIOS_INCLUDE_MPU9250_SPI)

#include "physical_constants.h"
#include "pios_mpu9250.h"
#include "pios_semaphore.h"
#include "pios_thread.h"
#include "pios_queue.h"

/* Private constants */
#define MPU9250_TASK_PRIORITY    PIOS_THREAD_PRIO_HIGHEST
#define MPU9250_TASK_STACK_BYTES 512
#define PIOS_MPU9250_MAX_DOWNSAMPLE 2

#define MPU9250_WHOAMI_ID       0x71

#ifdef PIOS_MPU9250_SPI_HIGH_SPEED
#define MPU9250_SPI_HIGH_SPEED              PIOS_MPU9250_SPI_HIGH_SPEED
#else
#define MPU9250_SPI_HIGH_SPEED              20000000
#endif
#define MPU9250_SPI_LOW_SPEED               300000

#define PIOS_MPU9250_ACCEL_DLPF_CFG_REG     0x1D

#define PIOS_MPU9250_AK8963_ADDR            0x0C
#define AK8963_WHOAMI_REG                   0x00
#define AK8963_WHOAMI_ID                    0x48
#define AK8963_ST1_REG                      0x02
#define AK8963_ST2_REG                      0x09
#define AK8963_ST1_DOR                      0x02
#define AK8963_ST1_DRDY                     0x01
#define AK8963_ST2_BITM                     0x10
#define AK8963_ST2_HOFL                     0x08
#define AK8963_CNTL1_REG                    0x0A
#define AK8963_CNTL2_REG                    0x0A
#define AK8963_CNTL2_SRST                   0x01
#define AK8963_MODE_CONTINUOUS_FAST_16B     0x16

/* Global Variables */

enum pios_mpu9250_dev_magic {
	PIOS_MPU9250_DEV_MAGIC = 0xb8a9624f,
};

struct mpu9250_dev {
	uint32_t spi_id;
	uint32_t slave_num;
	enum pios_mpu60x0_accel_range accel_range;
	enum pios_mpu60x0_range gyro_range;
	struct pios_queue *gyro_queue;
	struct pios_queue *accel_queue;
	struct pios_queue *mag_queue;
	struct pios_thread *TaskHandle;
	struct pios_semaphore *data_ready_sema;
	const struct pios_mpu9250_cfg *cfg;
	enum pios_mpu9250_gyro_filter gyro_filter;
	enum pios_mpu9250_accel_filter accel_filter;
	enum pios_mpu9250_dev_magic magic;
};

//! Global structure for this device device
static struct mpu9250_dev *dev;

//! Private functions
static struct mpu9250_dev *PIOS_MPU9250_alloc(const struct pios_mpu9250_cfg *cfg);
static int32_t PIOS_MPU9250_Validate(struct mpu9250_dev *dev);
static void PIOS_MPU9250_Task(void *parameters);
static uint8_t PIOS_MPU9250_ReadReg(uint8_t reg);
static int32_t PIOS_MPU9250_WriteReg(uint8_t reg, uint8_t data);
static int32_t PIOS_MPU9250_ClaimBus(bool lowspeed);
static int32_t PIOS_MPU9250_ReleaseBus(bool lowspeed);

/**
 * @brief Allocate a new device
 */
static struct mpu9250_dev *PIOS_MPU9250_alloc(const struct pios_mpu9250_cfg *cfg)
{
	struct mpu9250_dev *mpu9250_dev;

	mpu9250_dev = (struct mpu9250_dev *)PIOS_malloc(sizeof(*mpu9250_dev));
	if (!mpu9250_dev)
		return NULL;

	mpu9250_dev->magic = PIOS_MPU9250_DEV_MAGIC;

	mpu9250_dev->accel_queue = PIOS_Queue_Create(PIOS_MPU9250_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_accel_data));
	if (mpu9250_dev->accel_queue == NULL) {
		PIOS_free(mpu9250_dev);
		return NULL;
	}

	mpu9250_dev->gyro_queue = PIOS_Queue_Create(PIOS_MPU9250_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_gyro_data));
	if (mpu9250_dev->gyro_queue == NULL) {
		PIOS_Queue_Delete(dev->accel_queue);
		PIOS_free(mpu9250_dev);
		return NULL;
	}

	if (cfg->use_magnetometer) {
		mpu9250_dev->mag_queue = PIOS_Queue_Create(PIOS_MPU9250_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_mag_data));
		if (mpu9250_dev->mag_queue == NULL) {
			PIOS_Queue_Delete(dev->accel_queue);
			PIOS_Queue_Delete(dev->gyro_queue);
			PIOS_free(mpu9250_dev);
			return NULL;
		}
	}

	mpu9250_dev->data_ready_sema = PIOS_Semaphore_Create();
	if (mpu9250_dev->data_ready_sema == NULL) {
		PIOS_Queue_Delete(dev->accel_queue);
		PIOS_Queue_Delete(dev->gyro_queue);
		if (cfg->use_magnetometer)
			PIOS_Queue_Delete(dev->mag_queue);
		PIOS_free(mpu9250_dev);
		return NULL;
	}

	return mpu9250_dev;
}

/**
 * @brief Validate the handle to the device
 * @returns 0 for valid device or -1 otherwise
 */
static int32_t PIOS_MPU9250_Validate(struct mpu9250_dev *dev)
{
	if (dev == NULL)
		return -1;
	if (dev->magic != PIOS_MPU9250_DEV_MAGIC)
		return -2;
	if (dev->spi_id == 0)
		return -3;
	return 0;
}

/**
 * @brief Claim the SPI bus for the communications and select this chip
 * \param[in] flag controls if low speed access for control registers should be used
 * @return 0 if successful, -1 for invalid device, -2 if unable to claim bus
 */
static int32_t PIOS_MPU9250_ClaimBus(bool lowspeed)
{
	if (PIOS_MPU9250_Validate(dev) != 0)
		return -1;

	if (PIOS_SPI_ClaimBus(dev->spi_id) != 0)
		return -2;

	if (lowspeed)
		PIOS_SPI_SetClockSpeed(dev->spi_id, MPU9250_SPI_LOW_SPEED);

	PIOS_SPI_RC_PinSet(dev->spi_id, dev->slave_num, 0);

	return 0;
}

/**
 * @brief Release the SPI bus for the communications and end the transaction
 * \param[in] must be true when bus was claimed in lowspeed mode
 * @return 0 if successful
 */
static int32_t PIOS_MPU9250_ReleaseBus(bool lowspeed)
{
	if (PIOS_MPU9250_Validate(dev) != 0)
		return -1;

	PIOS_SPI_RC_PinSet(dev->spi_id, dev->slave_num, 1);

	if (lowspeed)
		PIOS_SPI_SetClockSpeed(dev->spi_id, MPU9250_SPI_HIGH_SPEED);

	PIOS_SPI_ReleaseBus(dev->spi_id);

	return 0;
}

/**
 * @brief Read a register from MPU9250
 * @returns The register value
 * @param reg[in] Register address to be read
 */
static uint8_t PIOS_MPU9250_ReadReg(uint8_t reg)
{
	uint8_t data;

	PIOS_MPU9250_ClaimBus(true);

	PIOS_SPI_TransferByte(dev->spi_id, 0x80 | reg); // request byte
	data = PIOS_SPI_TransferByte(dev->spi_id, 0);   // receive response

	PIOS_MPU9250_ReleaseBus(true);

	return data;
}

/**
 * @brief Writes one byte to the MPU9250 register
 * \param[in] reg Register address
 * \param[in] data Byte to write
 * @returns 0 when success
 */
static int32_t PIOS_MPU9250_WriteReg(uint8_t reg, uint8_t data)
{
	if (PIOS_MPU9250_ClaimBus(true) != 0)
		return -1;

	PIOS_SPI_TransferByte(dev->spi_id, 0x7f & reg);
	PIOS_SPI_TransferByte(dev->spi_id, data);

	PIOS_MPU9250_ReleaseBus(true);

	return 0;
}

/**
 * @brief Writes one byte to the AK8963 register using MPU9250 I2C master
 * \param[in] reg Register address
 * \param[in] data Byte to write
 * @returns 0 when success
 */
static int32_t PIOS_MPU9250_Mag_WriteReg(uint8_t reg, uint8_t data)
{
	// we will use I2C SLV4 to manipulate with AK8963 control registers
	if (PIOS_MPU9250_WriteReg(PIOS_MPU60X0_SLV4_REG_REG, reg) != 0)
		return -1;
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_SLV4_ADDR_REG, PIOS_MPU9250_AK8963_ADDR);
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_SLV4_DO_REG, data);
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_SLV4_CTRL_REG, PIOS_MPU60X0_I2CSLV_EN);
	uint32_t timeout = 0;

	// wait for I2C transaction done, use simple safety
	// escape counter to prevent endless loop in case
	// MPU9250 is broken
	uint8_t status = 0;
	do {
		if (timeout++ > 50)
			return -2;

		status = PIOS_MPU9250_ReadReg(PIOS_MPU60X0_I2C_MST_STATUS_REG);
	} while ((status & PIOS_MPU60X0_I2C_MST_SLV4_DONE) == 0);

	if (status & PIOS_MPU60X0_I2C_MST_SLV4_NACK)
		return -3;

	return 0;
}

/**
 * @brief Reads one byte from the AK8963 register using MPU9250 I2C master
 * \param[in] reg Register address
 * \param[in] data Byte to write
 */
static uint8_t PIOS_MPU9250_Mag_ReadReg(uint8_t reg)
{
	// we will use I2C SLV4 to manipulate with AK8963 control registers
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_SLV4_REG_REG, reg);
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_SLV4_ADDR_REG, PIOS_MPU9250_AK8963_ADDR | 0x80);
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_SLV4_CTRL_REG, PIOS_MPU60X0_I2CSLV_EN);
	uint32_t timeout = 0;

	// wait for I2C transaction done, use simple safety
	// escape counter to prevent endless loop in case
	// MPU9250 is broken
	uint8_t status = 0;
	do {
		if (timeout++ > 50)
			return 0;

		status = PIOS_MPU9250_ReadReg(PIOS_MPU60X0_I2C_MST_STATUS_REG);
	} while ((status & PIOS_MPU60X0_I2C_MST_SLV4_DONE) == 0);

	return PIOS_MPU9250_ReadReg(PIOS_MPU60X0_SLV4_DI_REG);
}

/**
 * @brief Initialize the AK8963 magnetometer inside MPU9250
 * \return 0 if success
 *
 */
static int32_t PIOS_MPU9250_Mag_Config(void)
{
	uint8_t id = PIOS_MPU9250_Mag_ReadReg(AK8963_WHOAMI_REG);
	if (id != AK8963_WHOAMI_ID)
		return -2;

	// reset AK8963
	if (PIOS_MPU9250_Mag_WriteReg(AK8963_CNTL2_REG, AK8963_CNTL2_SRST) != 0)
		return -3;

	// give chip some time to initialize
	PIOS_DELAY_WaitmS(2);

	// set magnetometer sampling rate to 100Hz and 16-bit resolution
	PIOS_MPU9250_Mag_WriteReg(AK8963_CNTL1_REG, AK8963_MODE_CONTINUOUS_FAST_16B);

	// configure mpu9250 to read ak8963 data range from STATUS1 to STATUS2 at ODR
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_SLV0_REG_REG, AK8963_ST1_REG);
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_SLV0_ADDR_REG, PIOS_MPU9250_AK8963_ADDR | 0x80);
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_SLV0_CTRL_REG, PIOS_MPU60X0_I2CSLV_EN | 8);

	return 0;
}

/**
 * @brief Initialize the MPU9250 gyro & accel registers
 * \return 0 if successful
 * \param[in] pios_mpu9250_cfg struct to be used to configure sensor.
 *
 */
static int32_t PIOS_MPU9250_Config(struct pios_mpu9250_cfg const *cfg)
{
	// reset chip
	if (PIOS_MPU9250_WriteReg(PIOS_MPU60X0_PWR_MGMT_REG, PIOS_MPU60X0_PWRMGMT_IMU_RST) != 0)
		return -1;

	// give chip some time to initialize
	PIOS_DELAY_WaitmS(50);

	uint8_t id = PIOS_MPU9250_ReadReg(PIOS_MPU60X0_WHOAMI);
	if (id != MPU9250_WHOAMI_ID)
		return -2;

	// power management config
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_PWR_MGMT_REG, PIOS_MPU60X0_PWRMGMT_PLL_X_CLK);

	// user control
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_USER_CTRL_REG, PIOS_MPU60X0_USERCTL_DIS_I2C | PIOS_MPU60X0_USERCTL_I2C_MST_EN);

	if (dev->cfg->use_magnetometer)
		if (PIOS_MPU9250_Mag_Config() != 0)
			return -3;

	// Digital low-pass filter and scale
	// set this before sample rate else sample rate calculation will fail
	PIOS_MPU9250_SetAccelLPF(cfg->default_accel_filter);
	PIOS_MPU9250_SetGyroLPF(cfg->default_gyro_filter);

	// Sample rate
	if (PIOS_MPU9250_SetSampleRate(cfg->default_samplerate) != 0)
		return -4;

	// Set the gyro scale
	PIOS_MPU9250_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);

	// Set the accel scale
	PIOS_MPU9250_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);

	// Interrupt configuration
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_INT_CFG_REG, cfg->interrupt_cfg);

	// Interrupt enable
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_INT_EN_REG, PIOS_MPU60X0_INTEN_DATA_RDY);

	return 0;
}

/**
 * @brief Initialize the MPU9250 9-axis sensor.
 * @return 0 for success, -1 for failure to allocate, -10 for failure to get irq
 */
int32_t PIOS_MPU9250_SPI_Init(uint32_t spi_id, uint32_t slave_num, const struct pios_mpu9250_cfg *cfg)
{
	dev = PIOS_MPU9250_alloc(cfg);
	if (dev == NULL)
		return -1;

	dev->spi_id = spi_id;
	dev->slave_num = slave_num;
	dev->cfg = cfg;

	/* Configure the MPU9250 Sensor */
	if (PIOS_MPU9250_Config(cfg) != 0)
		return -2;

	/* Set up EXTI line */
	PIOS_EXTI_Init(cfg->exti_cfg);

	// Wait 20 ms for data ready interrupt and make sure it happens
	// twice
	if ((PIOS_Semaphore_Take(dev->data_ready_sema, 20) != true) ||
		(PIOS_Semaphore_Take(dev->data_ready_sema, 20) != true)) {
		return -10;
	}

	dev->TaskHandle = PIOS_Thread_Create(
			PIOS_MPU9250_Task, "pios_mpu9250", MPU9250_TASK_STACK_BYTES, NULL, MPU9250_TASK_PRIORITY);
	PIOS_Assert(dev->TaskHandle != NULL);

	PIOS_SENSORS_Register(PIOS_SENSOR_ACCEL, dev->accel_queue);
	PIOS_SENSORS_Register(PIOS_SENSOR_GYRO, dev->gyro_queue);

	if (dev->cfg->use_magnetometer)
		PIOS_SENSORS_Register(PIOS_SENSOR_MAG, dev->mag_queue);

	return 0;
}

/**
 * @brief Test MPU9250 presence on the bus
 * @returns 0 if success
 */
int32_t PIOS_MPU9250_Test(void)
{
	uint8_t id = PIOS_MPU9250_ReadReg(PIOS_MPU60X0_WHOAMI);
	if (id != MPU9250_WHOAMI_ID)
		return 1;

	id = PIOS_MPU9250_Mag_ReadReg(AK8963_WHOAMI_REG);
	if (id != AK8963_WHOAMI_ID)
		return -2;

	return 0;
}

/**
 * @brief Set gyroscope range
 * @returns 0 if successful
 * @param range[in] gyroscope range
 */
int32_t PIOS_MPU9250_SetGyroRange(enum pios_mpu60x0_range range)
{
	if (PIOS_MPU9250_WriteReg(PIOS_MPU60X0_GYRO_CFG_REG, range) != 0)
		return -1;

	switch (range) {
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

	dev->gyro_range = range;
	return 0;
}

/**
 * @brief Set accelerometer range
 * @returns 0 if success
 * @param range[in] accelerometer range
 */
int32_t PIOS_MPU9250_SetAccelRange(enum pios_mpu60x0_accel_range range)
{
	if (PIOS_MPU9250_WriteReg(PIOS_MPU60X0_ACCEL_CFG_REG, range) != 0)
		return -1;
	dev->accel_range = range;

	return 0;
}

/**
 * @brief Set sampling frequency of accels and gyros axes
 * @returns 0 if successful
 * @param samplerate_hz[in] Sampling frequency in Hz
 */
int32_t PIOS_MPU9250_SetSampleRate(uint16_t samplerate_hz)
{
	// mpu9250 ODR divider is unable to run from 8kHz clock like mpu60x0 :(
	// check if someone want to use 250Hz DLPF and don't want 8kHz sampling
	// and politely refuse him
	if ((dev->gyro_filter == PIOS_MPU9250_GYRO_LOWPASS_250_HZ) && (samplerate_hz != 8000))
		return -1;

	uint16_t filter_frequency = 1000;

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

	return PIOS_MPU9250_WriteReg(PIOS_MPU60X0_SMPLRT_DIV_REG, (uint8_t)divisor);
}

/**
 * @brief Set gyroscope lowpass filter cut-off frequency
 * @param filter[in] Filter frequency
 */
void PIOS_MPU9250_SetGyroLPF(enum pios_mpu9250_gyro_filter filter)
{
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_DLPF_CFG_REG, filter);

	dev->gyro_filter = filter;
}

/**
 * @brief Set accelerometer lowpass filter cut-off frequency
 * @param filter[in] Filter frequency
 */
void PIOS_MPU9250_SetAccelLPF(enum pios_mpu9250_accel_filter filter)
{
	// Note this sets the a_fchoice_b to 0 which enables the DLPF
	// which is what is desired.
	PIOS_MPU9250_WriteReg(PIOS_MPU60X0_ACCEL_CFG2_REG, filter);

	dev->accel_filter = filter;
}

/**
 * @brief Get current gyro scale for deg/s
 * @returns scale
 */
static float PIOS_MPU9250_GetGyroScale(void)
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

/**
 * @brief Get current gyro scale for ms^-2
 * @returns scale
 */
static float PIOS_MPU9250_GetAccelScale(void)
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

/**
* @brief IRQ Handler.  Notice MPU9250 task to read all sensors data.
*/
bool PIOS_MPU9250_IRQHandler(void)
{
	if (PIOS_MPU9250_Validate(dev) != 0)
		return false;

	bool need_yield = false;

	PIOS_Semaphore_Give_FromISR(dev->data_ready_sema, &need_yield);

	return need_yield;
}

static void PIOS_MPU9250_Task(void *parameters)
{
	while (1) {
		//Wait for data ready interrupt
		if (PIOS_Semaphore_Take(dev->data_ready_sema, PIOS_SEMAPHORE_TIMEOUT_MAX) != true)
			continue;

		enum {
			IDX_REG = 0,
			IDX_ACCEL_XOUT_H,
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
			IDX_MAG_ST1,
			IDX_MAG_XOUT_L,
			IDX_MAG_XOUT_H,
			IDX_MAG_YOUT_L,
			IDX_MAG_YOUT_H,
			IDX_MAG_ZOUT_L,
			IDX_MAG_ZOUT_H,
			IDX_MAG_ST2,
			BUFFER_SIZE,
		};

		uint8_t mpu9250_rec_buf[BUFFER_SIZE];
		uint8_t mpu9250_tx_buf[BUFFER_SIZE] = {PIOS_MPU60X0_ACCEL_X_OUT_MSB | 0x80, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

		uint8_t transfer_size = (dev->cfg->use_magnetometer) ? BUFFER_SIZE : BUFFER_SIZE - 8;
		// claim bus in high speed mode
		if (PIOS_MPU9250_ClaimBus(false) != 0)
			continue;

		if (PIOS_SPI_TransferBlock(dev->spi_id, mpu9250_tx_buf, mpu9250_rec_buf, transfer_size, 0) < 0) {
			PIOS_MPU9250_ReleaseBus(false);
			continue;
		}

		PIOS_MPU9250_ReleaseBus(false);

		struct pios_sensor_accel_data accel_data;
		struct pios_sensor_gyro_data gyro_data;
		struct pios_sensor_mag_data mag_data;

		float accel_x = (int16_t)(mpu9250_rec_buf[IDX_ACCEL_XOUT_H] << 8 | mpu9250_rec_buf[IDX_ACCEL_XOUT_L]);
		float accel_y = (int16_t)(mpu9250_rec_buf[IDX_ACCEL_YOUT_H] << 8 | mpu9250_rec_buf[IDX_ACCEL_YOUT_L]);
		float accel_z = (int16_t)(mpu9250_rec_buf[IDX_ACCEL_ZOUT_H] << 8 | mpu9250_rec_buf[IDX_ACCEL_ZOUT_L]);
		float gyro_x = (int16_t)(mpu9250_rec_buf[IDX_GYRO_XOUT_H] << 8 | mpu9250_rec_buf[IDX_GYRO_XOUT_L]);
		float gyro_y = (int16_t)(mpu9250_rec_buf[IDX_GYRO_YOUT_H] << 8 | mpu9250_rec_buf[IDX_GYRO_YOUT_L]);
		float gyro_z = (int16_t)(mpu9250_rec_buf[IDX_GYRO_ZOUT_H] << 8 | mpu9250_rec_buf[IDX_GYRO_ZOUT_L]);
		float mag_x = (int16_t)(mpu9250_rec_buf[IDX_MAG_XOUT_H] << 8 | mpu9250_rec_buf[IDX_MAG_XOUT_L]);
		float mag_y = (int16_t)(mpu9250_rec_buf[IDX_MAG_YOUT_H] << 8 | mpu9250_rec_buf[IDX_MAG_YOUT_L]);
		float mag_z = (int16_t)(mpu9250_rec_buf[IDX_MAG_ZOUT_H] << 8 | mpu9250_rec_buf[IDX_MAG_ZOUT_L]);

		// Rotate the sensor to TL convention.  The datasheet defines X as towards the right
		// and Y as forward. TL convention transposes this.  Also the Z is defined negatively
		// to our convention. This is true for accels and gyros. Magnetometer corresponds TL convention.
		switch (dev->cfg->orientation) {
		case PIOS_MPU9250_TOP_0DEG:
			accel_data.y = accel_x;
			accel_data.x = accel_y;
			accel_data.z = -accel_z;
			gyro_data.y  = gyro_x;
			gyro_data.x  = gyro_y;
			gyro_data.z  = -gyro_z;
			mag_data.x   = mag_x;
			mag_data.y   = mag_y;
			mag_data.z   = mag_z;
			break;
		case PIOS_MPU9250_TOP_90DEG:
			accel_data.y = -accel_y;
			accel_data.x = accel_x;
			accel_data.z = -accel_z;
			gyro_data.y  = -gyro_y;
			gyro_data.x  = gyro_x;
			gyro_data.z  = -gyro_z;
			mag_data.x   = -mag_y;
			mag_data.y   = mag_x;
			mag_data.z   = mag_z;
			break;
		case PIOS_MPU9250_TOP_180DEG:
			accel_data.y = -accel_x;
			accel_data.x = -accel_y;
			accel_data.z = -accel_z;
			gyro_data.y  = -gyro_x;
			gyro_data.x  = -gyro_y;
			gyro_data.z  = -gyro_z;
			mag_data.x   = -mag_x;
			mag_data.y   = -mag_y;
			mag_data.z   = mag_z;

			break;
		case PIOS_MPU9250_TOP_270DEG:
			accel_data.y = accel_y;
			accel_data.x = -accel_x;
			accel_data.z = -accel_z;
			gyro_data.y  = gyro_y;
			gyro_data.x  = -gyro_x;
			gyro_data.z  = -gyro_z;
			mag_data.x   = mag_y;
			mag_data.y   = -mag_x;
			mag_data.z   = mag_z;
			break;
		case PIOS_MPU9250_BOTTOM_0DEG:
			accel_data.y = -accel_x;
			accel_data.x = accel_y;
			accel_data.z = accel_z;
			gyro_data.y  = -gyro_x;
			gyro_data.x  = gyro_y;
			gyro_data.z  = gyro_z;
			mag_data.x   = mag_x;
			mag_data.y   = -mag_y;
			mag_data.z   = -mag_z;
			break;

		case PIOS_MPU9250_BOTTOM_90DEG:
			accel_data.y = -accel_y;
			accel_data.x = -accel_x;
			accel_data.z = accel_z;
			gyro_data.y  = -gyro_y;
			gyro_data.x  = -gyro_x;
			gyro_data.z  = gyro_z;
			mag_data.x   = -mag_y;
			mag_data.y   = -mag_x;
			mag_data.z   = -mag_z;
			break;

		case PIOS_MPU9250_BOTTOM_180DEG:
			accel_data.y = accel_x;
			accel_data.x = -accel_y;
			accel_data.z = accel_z;
			gyro_data.y  = gyro_x;
			gyro_data.x  = -gyro_y;
			gyro_data.z  = gyro_z;
			mag_data.x   = -mag_x;
			mag_data.y   = mag_y;
			mag_data.z   = -mag_z;
			break;

		case PIOS_MPU9250_BOTTOM_270DEG:
			accel_data.y = accel_y;
			accel_data.x = accel_x;
			gyro_data.y  = gyro_y;
			gyro_data.x  = gyro_x;
			gyro_data.z  = gyro_z;
			accel_data.z = accel_z;
			mag_data.x   = mag_y;
			mag_data.y   = mag_x;
			mag_data.z   = -mag_z;
			break;

		}

		int16_t raw_temp = (int16_t)(mpu9250_rec_buf[IDX_TEMP_OUT_H] << 8 | mpu9250_rec_buf[IDX_TEMP_OUT_L]);
		float temperature = 21.0f + ((float)raw_temp) / 333.87f;

		// Apply sensor scaling
		float accel_scale = PIOS_MPU9250_GetAccelScale();
		accel_data.x *= accel_scale;
		accel_data.y *= accel_scale;
		accel_data.z *= accel_scale;
		accel_data.temperature = temperature;

		float gyro_scale = PIOS_MPU9250_GetGyroScale();
		gyro_data.x *= gyro_scale;
		gyro_data.y *= gyro_scale;
		gyro_data.z *= gyro_scale;
		gyro_data.temperature = temperature;

		PIOS_Queue_Send(dev->accel_queue, &accel_data, 0);
		PIOS_Queue_Send(dev->gyro_queue, &gyro_data, 0);

		if (dev->cfg->use_magnetometer) {
			uint8_t st1 = mpu9250_rec_buf[IDX_MAG_ST1];
			if (st1 & AK8963_ST1_DRDY) {
				mag_data.x *= 1.5f;
				mag_data.y *= 1.5f;
				mag_data.z *= 1.5f;
				PIOS_Queue_Send(dev->mag_queue, &mag_data, 0);
			}
		}
	}
}

#endif

/**
 * @}
 * @}
 */
