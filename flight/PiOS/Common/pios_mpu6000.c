/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_MPU6000 MPU6000 Functions
 * @brief Deals with the hardware interface to the 3-axis gyro
 * @{
 *
 * @file       pios_mpu6000.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      MPU6000 6-axis gyro and accel chip
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

#if defined(PIOS_INCLUDE_MPU6000)

#include "pios_queue.h"
#include "physical_constants.h"
#include "pios_semaphore.h"
#include "pios_thread.h"

// Math libraries
#include "misc_math.h"
#ifndef SMALLF1
#include "cic.h"
#endif

/* Private macros */
#define PACK_SENSOR_VALUE16(h_byte, l_byte) ((float)((int16_t)(mpu6000_rec_buf[h_byte] << 8 | mpu6000_rec_buf[l_byte])))

#ifndef SMALLF1
#define RAW_ACCEL_X  (accel_data_temp.x / TaskAllSubSamplesCnt)
#define RAW_ACCEL_Y  (accel_data_temp.y / TaskAllSubSamplesCnt)
#define RAW_ACCEL_Z  (accel_data_temp.z / TaskAllSubSamplesCnt)
#define RAW_GYRO_X   (cic_get_decimation_output(&gyro_filter_xyz[X1]) / gyro_lpf_gain)
#define RAW_GYRO_Y   (cic_get_decimation_output(&gyro_filter_xyz[Y1]) / gyro_lpf_gain)
#define RAW_GYRO_Z   (cic_get_decimation_output(&gyro_filter_xyz[Z1]) / gyro_lpf_gain)
#define RAW_TEMP     (raw_temp / TaskAllSubSamplesCnt)
#else
#define RAW_ACCEL_X PACK_SENSOR_VALUE16(IDX_ACCEL_XOUT_H, IDX_ACCEL_XOUT_L)
#define RAW_ACCEL_Y PACK_SENSOR_VALUE16(IDX_ACCEL_YOUT_H, IDX_ACCEL_YOUT_L)
#define RAW_ACCEL_Z PACK_SENSOR_VALUE16(IDX_ACCEL_ZOUT_H, IDX_ACCEL_ZOUT_L)
#define RAW_GYRO_X  PACK_SENSOR_VALUE16(IDX_GYRO_XOUT_H, IDX_GYRO_XOUT_L)
#define RAW_GYRO_Y  PACK_SENSOR_VALUE16(IDX_GYRO_YOUT_H, IDX_GYRO_YOUT_L)
#define RAW_GYRO_Z  PACK_SENSOR_VALUE16(IDX_GYRO_ZOUT_H, IDX_GYRO_ZOUT_L)
#define RAW_TEMP    PACK_SENSOR_VALUE16(IDX_TEMP_OUT_H, IDX_TEMP_OUT_L)
#endif



/* Private constants */
#define MPU6000_TASK_PRIORITY	PIOS_THREAD_PRIO_HIGHEST

#ifndef SMALLF1
#define MPU6000_TASK_STACK		512 // seems to need higher value for the CIC filtering; 488 seems also to work, but to be safe using 512, the same as MPU9250
#else
#define MPU6000_TASK_STACK		484
#endif

#ifdef PIOS_MPU6000_SPI_HIGH_SPEED
#define MPU6000_SPI_HIGH_SPEED              PIOS_MPU6000_SPI_HIGH_SPEED
#else
#define MPU6000_SPI_HIGH_SPEED              20000000	// should result in 10.5MHz clock on F4 targets like Quanton, and 18MHz on F1 targets like CC3D
#endif
#define MPU6000_SPI_LOW_SPEED               1000000

// Private variables
#ifndef SMALLF1
struct gyro_downsampling_dev {
	uint8_t Drop;
	uint8_t LPF;
	uint8_t LPF_Order;
	uint8_t LPF_DDelay;
}gyro_downsampling_settings = { .Drop = 1, .LPF = 1, .LPF_Order = 0, .LPF_DDelay = 1 }; // Init with no down-sampling

enum {X1 = 0, Y1, Z1, AXES_SIZE};

static struct cic_filter_data gyro_filter_xyz[3];

float gyro_lpf_gain = 1;
#endif

/* Global Variables */

enum pios_mpu6000_dev_magic {
    PIOS_MPU6000_DEV_MAGIC = 0x9da9b3ed,
};

#define PIOS_MPU6000_MAX_QUEUESIZE 2

struct mpu6000_dev {
	uint32_t spi_id;
	uint32_t slave_num;
	enum pios_mpu60x0_range gyro_range;
	struct pios_queue *gyro_queue;
#if defined(PIOS_MPU6000_ACCEL)
	enum pios_mpu60x0_accel_range accel_range;
	struct pios_queue *accel_queue;
#endif /* PIOS_MPU6000_ACCEL */
	const struct pios_mpu60x0_cfg *cfg;
	volatile bool configured;
	enum pios_mpu6000_dev_magic magic;
	enum pios_mpu60x0_filter filter;
	struct pios_thread *threadp;
	struct pios_semaphore *data_ready_sema;
	uint16_t gyro_samplerate_hz;
};

//! Global structure for this device device
static struct mpu6000_dev *pios_mpu6000_dev;

//! Private functions
static struct mpu6000_dev *PIOS_MPU6000_alloc(void);
static int32_t PIOS_MPU6000_Validate(struct mpu6000_dev *dev);
static void PIOS_MPU6000_Config(const struct pios_mpu60x0_cfg *cfg);
static int32_t PIOS_MPU6000_ClaimBus(bool lowspeed);
static int32_t PIOS_MPU6000_ReleaseBus(bool lowspeed);
static int32_t PIOS_MPU6000_SetReg(uint8_t address, uint8_t buffer);
static int32_t PIOS_MPU6000_GetReg(uint8_t address);
static void PIOS_MPU6000_Task(void *parameters);

/**
 * @brief Allocate a new device
 */
static struct mpu6000_dev *PIOS_MPU6000_alloc(void)
{
	struct mpu6000_dev *mpu6000_dev;

	mpu6000_dev = (struct mpu6000_dev *)PIOS_malloc(sizeof(*mpu6000_dev));

	if (!mpu6000_dev) return (NULL);

	mpu6000_dev->magic = PIOS_MPU6000_DEV_MAGIC;

	mpu6000_dev->configured = false;

#if defined(PIOS_MPU6000_ACCEL)
	mpu6000_dev->accel_queue = PIOS_Queue_Create(PIOS_MPU6000_MAX_QUEUESIZE, sizeof(struct pios_sensor_accel_data));

	if (mpu6000_dev->accel_queue == NULL) {
		PIOS_free(mpu6000_dev);
		return NULL;
	}
#endif /* PIOS_MPU6000_ACCEL */

	mpu6000_dev->gyro_queue = PIOS_Queue_Create(PIOS_MPU6000_MAX_QUEUESIZE, sizeof(struct pios_sensor_gyro_data));

	if (mpu6000_dev->gyro_queue == NULL) {
		PIOS_free(mpu6000_dev);
		return NULL;
	}

	mpu6000_dev->data_ready_sema = PIOS_Semaphore_Create();

	if (mpu6000_dev->data_ready_sema == NULL) {
		PIOS_free(mpu6000_dev);
		return NULL;
	}

	return mpu6000_dev;
}

/**
 * @brief Validate the handle to the spi device
 * @returns 0 for valid device or -1 otherwise
 */
static int32_t PIOS_MPU6000_Validate(struct mpu6000_dev *dev)
{
	if (dev == NULL)
		return -1;

	if (dev->magic != PIOS_MPU6000_DEV_MAGIC)
		return -2;

	if (dev->spi_id == 0)
		return -3;

	return 0;
}

/**
 * @brief Initialize the MPU6000 3-axis gyro sensor.
 * @return 0 for success, -1 for failure
 */
int32_t PIOS_MPU6000_Init(uint32_t spi_id, uint32_t slave_num, const struct pios_mpu60x0_cfg *cfg)
{
	pios_mpu6000_dev = PIOS_MPU6000_alloc();

	if (pios_mpu6000_dev == NULL)
		return -1;

	pios_mpu6000_dev->spi_id = spi_id;
	pios_mpu6000_dev->slave_num = slave_num;
	pios_mpu6000_dev->cfg = cfg;

	/* Configure the MPU6000 Sensor */
	PIOS_SPI_SetClockSpeed(pios_mpu6000_dev->spi_id, MPU6000_SPI_LOW_SPEED);
	PIOS_MPU6000_Config(cfg);
	PIOS_SPI_SetClockSpeed(pios_mpu6000_dev->spi_id, MPU6000_SPI_HIGH_SPEED);

	pios_mpu6000_dev->threadp = PIOS_Thread_Create(
			PIOS_MPU6000_Task, "pios_mpu6000", MPU6000_TASK_STACK, NULL, MPU6000_TASK_PRIORITY);
	PIOS_Assert(pios_mpu6000_dev->threadp != NULL);

	/* Set up EXTI line */
	PIOS_EXTI_Init(cfg->exti_cfg);

#if defined(PIOS_MPU6000_ACCEL)
	PIOS_SENSORS_Register(PIOS_SENSOR_ACCEL, pios_mpu6000_dev->accel_queue);
#endif /* PIOS_MPU6000_ACCEL */

	PIOS_SENSORS_Register(PIOS_SENSOR_GYRO, pios_mpu6000_dev->gyro_queue);

	return 0;
}

/**
 * @brief Initialize the MPU6000 3-axis gyro sensor
 * \return none
 * \param[in] PIOS_MPU6000_ConfigTypeDef struct to be used to configure sensor.
*
*/
static void PIOS_MPU6000_Config(const struct pios_mpu60x0_cfg *cfg)
{
#if defined(PIOS_MPU6000_SIMPLE_INIT_SEQUENCE)

	// Reset chip registers
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, PIOS_MPU60X0_PWRMGMT_IMU_RST);

	// Reset sensors signal path
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_USER_CTRL_REG, PIOS_MPU60X0_USERCTL_GYRO_RST);

	// Give chip some time to initialize
	PIOS_DELAY_WaitmS(10);

	//Power management configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, cfg->Pwr_mgmt_clk);

	// User control
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_USER_CTRL_REG, cfg->User_ctl);

	// Digital low-pass filter and scale
	// set this before sample rate else sample rate calculation will fail
	PIOS_MPU6000_SetLPF(cfg->default_filter);

	// Sample rate
	PIOS_MPU6000_SetSampleRate(cfg->default_samplerate);

	// Set the gyro scale
	PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);

#if defined(PIOS_MPU6000_ACCEL)
	// Set the accel scale
	PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
#endif /* PIOS_MPU6000_ACCEL */

	// Interrupt configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_INT_CFG_REG, cfg->interrupt_cfg);

	// Interrupt enable
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_INT_EN_REG, cfg->interrupt_en);

#else /* PIOS_MPU6000_SIMPLE_INIT_SEQUENCE */

	/* This init sequence should really be dropped in favor of something
	 * less redundant but it seems to be hard to get it running well
	 * on all different targets.
	 */

	PIOS_MPU6000_ClaimBus(true);
	PIOS_DELAY_WaitmS(1);
	PIOS_MPU6000_ReleaseBus(true);
	PIOS_DELAY_WaitmS(10);

	// Reset chip
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, 0x80 | cfg->Pwr_mgmt_clk);
	do {
		PIOS_DELAY_WaitmS(5);
	} while (PIOS_MPU6000_GetReg(PIOS_MPU60X0_PWR_MGMT_REG) & 0x80);

	PIOS_DELAY_WaitmS(25);

	// Reset chip and fifo
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_USER_CTRL_REG, 0x80 | 0x01 | 0x02 | 0x04);
	do {
		PIOS_DELAY_WaitmS(5);
	} while (PIOS_MPU6000_GetReg(PIOS_MPU60X0_USER_CTRL_REG) & 0x07);

	PIOS_DELAY_WaitmS(25);

	//Power management configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, cfg->Pwr_mgmt_clk);

	// Interrupt configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_INT_CFG_REG, cfg->interrupt_cfg);

	// Interrupt configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_INT_EN_REG, cfg->interrupt_en);

#if defined(PIOS_MPU6000_ACCEL)
	// Set the accel scale
	PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
#endif

	// Digital low-pass filter and scale
	// set this before sample rate else sample rate calculation will fail
	PIOS_MPU6000_SetLPF(cfg->default_filter);

	// Sample rate
	PIOS_MPU6000_SetSampleRate(cfg->default_samplerate);

	// Set the gyro scale
	PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);

	// Interrupt configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_USER_CTRL_REG, cfg->User_ctl);

	//Power management configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, cfg->Pwr_mgmt_clk);

	// Interrupt configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_INT_CFG_REG, cfg->interrupt_cfg);

	// Interrupt configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_INT_EN_REG, cfg->interrupt_en);

#endif /* PIOS_MPU6000_SIMPLE_INIT_SEQUENCE */

	pios_mpu6000_dev->configured = true;
}

/**
 * Set the gyro range and store it locally for scaling
 */
void PIOS_MPU6000_SetGyroRange(enum pios_mpu60x0_range gyro_range)
{
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_GYRO_CFG_REG, gyro_range);

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

	pios_mpu6000_dev->gyro_range = gyro_range;
}

/**
 * Set the accel range and store it locally for scaling
 */
#if defined(PIOS_MPU6000_ACCEL)
void PIOS_MPU6000_SetAccelRange(enum pios_mpu60x0_accel_range accel_range)
{
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_ACCEL_CFG_REG, accel_range);

	pios_mpu6000_dev->accel_range = accel_range;
}
#endif /* PIOS_MPU6000_ACCEL */

/**
 * Set the sample rate in Hz by determining the nearest divisor
 * @param[in] sample rate in Hz
 */
void PIOS_MPU6000_SetSampleRate(uint16_t samplerate_hz)
{
	uint16_t filter_frequency = 8000;

	if (pios_mpu6000_dev->filter != PIOS_MPU60X0_LOWPASS_256_HZ)
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

	pios_mpu6000_dev->gyro_samplerate_hz = (uint16_t)(filter_frequency / (divisor + 1));

	PIOS_MPU6000_SetReg(PIOS_MPU60X0_SMPLRT_DIV_REG, (uint8_t)divisor);
}

/**
 * Configure the digital low-pass filter
 */
void PIOS_MPU6000_SetLPF(enum pios_mpu60x0_filter filter)
{
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_DLPF_CFG_REG, filter);

	pios_mpu6000_dev->filter = filter;
}

/**
 * @brief Claim the SPI bus for the accel communications and select this chip
 * \param[in] flag controls if low speed access for control registers should be used
 * @return 0 if successful, -1 for invalid device, -2 if unable to claim bus
 */
static int32_t PIOS_MPU6000_ClaimBus(bool lowspeed)
{
	if (PIOS_MPU6000_Validate(pios_mpu6000_dev) != 0)
		return -1;

	if (PIOS_SPI_ClaimBus(pios_mpu6000_dev->spi_id) != 0)
		return -2;

	if (lowspeed)
			PIOS_SPI_SetClockSpeed(pios_mpu6000_dev->spi_id, MPU6000_SPI_LOW_SPEED);

	PIOS_SPI_RC_PinSet(pios_mpu6000_dev->spi_id, pios_mpu6000_dev->slave_num, 0);
	return 0;
}

/**
 * @brief Release the SPI bus for the accel communications and end the transaction
 * \param[in] must be true when bus was claimed in lowspeed mode
 * @return 0 if successful
 */
static int32_t PIOS_MPU6000_ReleaseBus(bool lowspeed)
{
	if (PIOS_MPU6000_Validate(pios_mpu6000_dev) != 0)
		return -1;

	PIOS_SPI_RC_PinSet(pios_mpu6000_dev->spi_id, pios_mpu6000_dev->slave_num, 1);

	if (lowspeed)
		PIOS_SPI_SetClockSpeed(pios_mpu6000_dev->spi_id, MPU6000_SPI_HIGH_SPEED);

	return PIOS_SPI_ReleaseBus(pios_mpu6000_dev->spi_id);
}

/**
 * @brief Read a register from MPU6000
 * @returns The register value or -1 if failure to get bus
 * @param reg[in] Register address to be read
 */
static int32_t PIOS_MPU6000_GetReg(uint8_t reg)
{
	uint8_t data;

	if (PIOS_MPU6000_ClaimBus(true) != 0)
		return -1;

	PIOS_SPI_TransferByte(pios_mpu6000_dev->spi_id, (0x80 | reg)); // request byte
	data = PIOS_SPI_TransferByte(pios_mpu6000_dev->spi_id, 0);     // receive response

	PIOS_MPU6000_ReleaseBus(true);
	return data;
}

/**
 * @brief Writes one byte to the MPU6000
 * \param[in] reg Register address
 * \param[in] data Byte to write
 * \return 0 if operation was successful
 * \return -1 if unable to claim SPI bus
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_MPU6000_SetReg(uint8_t reg, uint8_t data)
{
	if (PIOS_MPU6000_ClaimBus(true) != 0)
		return -1;

	if (PIOS_SPI_TransferByte(pios_mpu6000_dev->spi_id, 0x7f & reg) != 0) {
		PIOS_MPU6000_ReleaseBus(true);
		return -2;
	}

	if (PIOS_SPI_TransferByte(pios_mpu6000_dev->spi_id, data) != 0) {
		PIOS_MPU6000_ReleaseBus(true);
		return -3;
	}

	PIOS_MPU6000_ReleaseBus(true);

	return 0;
}

/*
 * @brief Read the identification bytes from the MPU6000 sensor
 * \return ID read from MPU6000 or -1 if failure
*/
static int32_t PIOS_MPU6000_ReadID()
{
	int32_t mpu6000_id = PIOS_MPU6000_GetReg(PIOS_MPU60X0_WHOAMI);

	if (mpu6000_id < 0)
		return -1;

	return mpu6000_id;
}

/**
 * Get the gyro scale based on the active device settings
 * @return Scale in (deg/s) / LSB
 */
static float PIOS_MPU6000_GetGyroScale()
{
	switch (pios_mpu6000_dev->gyro_range) {
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
 * Get the accel scale based on the active settings
 * @returns Scale in (m/s^2) / LSB
 */
#if defined(PIOS_MPU6000_ACCEL)
static float PIOS_MPU6000_GetAccelScale()
{
	switch (pios_mpu6000_dev->accel_range) {
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
#endif /* PIOS_MPU6000_ACCEL */

/**
 * @brief Run self-test operation.
 * \return 0 if test succeeded
 * \return non-zero value if test succeeded
 */
int32_t PIOS_MPU6000_Test(void)
{
	/* Verify that ID matches (MPU6000 ID is 0x68) */
	int32_t mpu6000_id = PIOS_MPU6000_ReadID();

	if (mpu6000_id < 0)
		return -1;

	if (mpu6000_id != 0x68)
		return -2;

	return 0;
}

/**
* @brief IRQ Handler.  Read all the data from onboard buffer
*/
bool PIOS_MPU6000_IRQHandler(void)
{
	if (PIOS_MPU6000_Validate(pios_mpu6000_dev) != 0)
		return false;

	bool woken = false;

#ifndef SMALLF1
	static uint8_t IRQGyroSubSamplesCnt = 0; // definition & init of the gyro down-sampling counter, "Drop"

	if (++IRQGyroSubSamplesCnt >= gyro_downsampling_settings.Drop) {

		IRQGyroSubSamplesCnt = 0; // reset of the gyro downsampling counter

		PIOS_Semaphore_Give_FromISR(pios_mpu6000_dev->data_ready_sema, &woken);
	}
#else
	PIOS_Semaphore_Give_FromISR(pios_mpu6000_dev->data_ready_sema, &woken);
#endif

	return woken;
}

static void PIOS_MPU6000_Task(void *parameters)
{

#ifndef SMALLF1
	// definition & init of the down-sampling counter variables
	static uint8_t TaskGyroSubSamplesCnt = 0; // "LPF"
	static uint8_t TaskAllSubSamplesCnt  = 0;
#endif

	const uint8_t *mpu6000_send_buf;
	uint8_t *mpu6000_rec_buf;

	uint8_t transfer_size = 0;

	while (1) {
		//Wait for data ready interrupt
		if (PIOS_Semaphore_Take(pios_mpu6000_dev->data_ready_sema, PIOS_SEMAPHORE_TIMEOUT_MAX) != true)
			continue;

		enum {
		    IDX_SPI_DUMMY_BYTE = 0,
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
		    BUFFER_SIZE,
		};

		// buffer for sampling all data
		const uint8_t mpu6000_send_all_buf[BUFFER_SIZE] = { PIOS_MPU60X0_ACCEL_X_OUT_MSB | 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		uint8_t mpu6000_rec_all_buf[BUFFER_SIZE];

#ifndef SMALLF1
		enum {
			IDX2_SPI_DUMMY_BYTE = 0,
			IDX2_GYRO_XOUT_H,
			IDX2_GYRO_XOUT_L,
			IDX2_GYRO_YOUT_H,
			IDX2_GYRO_YOUT_L,
			IDX2_GYRO_ZOUT_H,
			IDX2_GYRO_ZOUT_L,
			BUFFER2_SIZE,
		};

		static bool AllSamplingFlg = false; // flag that indicates that all data should be sampled
		static bool DoOutputFlg    = false;   // flag that indicates that we have to compute an new output at the down sampled frequency

		// buffer for sampling only gyro data
		const uint8_t mpu6000_send_gyro_buf[BUFFER2_SIZE] = { PIOS_MPU60X0_GYRO_X_OUT_MSB | 0x80 };
		uint8_t mpu6000_rec_gyro_buf[BUFFER2_SIZE];


		//  get all data & compute an output at the down sampled frequency
		if (++TaskGyroSubSamplesCnt >= gyro_downsampling_settings.LPF) {
			DoOutputFlg = true;
			AllSamplingFlg = true;
		}

		// when all data should be sampled because we have to compute new output data, or we don't won't to miss new accel data
		if ((AllSamplingFlg == true) || ((pios_mpu6000_dev->gyro_samplerate_hz >= 1000) && ((pios_mpu6000_dev->gyro_samplerate_hz / TaskGyroSubSamplesCnt) <= 1000))) {
			AllSamplingFlg = true;

			mpu6000_send_buf = &mpu6000_send_all_buf[0];
			mpu6000_rec_buf = &mpu6000_rec_all_buf[0];
			transfer_size = sizeof(mpu6000_send_all_buf);

			TaskAllSubSamplesCnt++;
		}
		// otherwise sample only gyro data at high speed
		else {
			mpu6000_send_buf = &mpu6000_send_gyro_buf[0];
			mpu6000_rec_buf = &mpu6000_rec_gyro_buf[0];
			transfer_size = sizeof(mpu6000_send_gyro_buf);
		}
#else // #ifndef SMALLF1
		mpu6000_send_buf = &mpu6000_send_all_buf[0];
		mpu6000_rec_buf = &mpu6000_rec_all_buf[0];
		transfer_size = sizeof(mpu6000_send_all_buf);
#endif

		// claim bus in high speed mode
		if (PIOS_MPU6000_ClaimBus(false) != 0)
			continue;

		if (PIOS_SPI_TransferBlock(pios_mpu6000_dev->spi_id, mpu6000_send_buf, mpu6000_rec_buf, transfer_size, NULL) < 0) {
			PIOS_MPU6000_ReleaseBus(false);
			continue;
		}

		PIOS_MPU6000_ReleaseBus(false);

#if defined(PIOS_MPU6000_ACCEL)

		// Currently we only support rotations on top so switch X/Y accordingly
		static struct pios_sensor_accel_data accel_data;
		static struct pios_sensor_gyro_data gyro_data;

#ifndef SMALLF1
		static struct pios_sensor_accel_data accel_data_temp;
		static int32_t raw_temp = 0;

		// reset the variables, where the data is directly needed for the output queue and no CIC filtering (moving average) is applied  //TODO
		if (TaskGyroSubSamplesCnt == 1) {
			if (gyro_downsampling_settings.LPF_Order == 0) {
				gyro_filter_xyz[X1].integrateState0 = 0.0;
				gyro_filter_xyz[Y1].integrateState0 = 0.0;
				gyro_filter_xyz[Z1].integrateState0 = 0.0;
			}
			accel_data_temp.x = 0.0;
			accel_data_temp.y = 0.0;
			accel_data_temp.z = 0.0;

			raw_temp = 0.0;
		}


		// data conversion & 1st order CIC and boxcar (order = 0) integration
		if (AllSamplingFlg == true) {
			accel_data_temp.x                   += PACK_SENSOR_VALUE16(IDX_ACCEL_XOUT_H, IDX_ACCEL_XOUT_L);
			accel_data_temp.y                   += PACK_SENSOR_VALUE16(IDX_ACCEL_YOUT_H, IDX_ACCEL_YOUT_L);
			accel_data_temp.z                   += PACK_SENSOR_VALUE16(IDX_ACCEL_ZOUT_H, IDX_ACCEL_ZOUT_L);
			gyro_filter_xyz[X1].integrateState0 += PACK_SENSOR_VALUE16(IDX_GYRO_XOUT_H, IDX_GYRO_XOUT_L);
			gyro_filter_xyz[Y1].integrateState0 += PACK_SENSOR_VALUE16(IDX_GYRO_YOUT_H, IDX_GYRO_YOUT_L);
			gyro_filter_xyz[Z1].integrateState0 += PACK_SENSOR_VALUE16(IDX_GYRO_ZOUT_H, IDX_GYRO_ZOUT_L);
			raw_temp                            += PACK_SENSOR_VALUE16(IDX_TEMP_OUT_H, IDX_TEMP_OUT_L);

			AllSamplingFlg = false;
		}
		else {
			gyro_filter_xyz[X1].integrateState0 += PACK_SENSOR_VALUE16(IDX2_GYRO_XOUT_H, IDX2_GYRO_XOUT_L);
			gyro_filter_xyz[Y1].integrateState0 += PACK_SENSOR_VALUE16(IDX2_GYRO_YOUT_H, IDX2_GYRO_YOUT_L);
			gyro_filter_xyz[Z1].integrateState0 += PACK_SENSOR_VALUE16(IDX2_GYRO_ZOUT_H, IDX2_GYRO_ZOUT_L);

		}

		// 1st order CIC and boxcar (order = 0) integration already done in the conversion above
		// for the higher order integration stages and all axes:
		for(uint8_t i=0; i< AXES_SIZE; i++) {
			if (gyro_downsampling_settings.LPF_Order > 1)
				cic_apply_higher_order_int_stage(&gyro_filter_xyz[i]);
		}


		// frequency decimation
		if (DoOutputFlg == true) {

			// CIC comb stage
			for(uint8_t i=0; i< AXES_SIZE; i++) {
				if (gyro_downsampling_settings.LPF_Order > 0) // only for CIC filter, not for boxcar
					cic_apply_comb_stage(&gyro_filter_xyz[i]);
			}
#endif // #ifndef SMALLF1

			// Rotate the sensor to TL convention.  The datasheet defines X as towards the right
			// and Y as forward.  TL convention transposes this.  Also the Z is defined negatively to our convention
			switch (pios_mpu6000_dev->cfg->orientation) {
			case PIOS_MPU60X0_TOP_0DEG:
				accel_data.y = RAW_ACCEL_X;
				accel_data.x = RAW_ACCEL_Y;
				accel_data.z = - RAW_ACCEL_Z;
				gyro_data.y  = RAW_GYRO_X;
				gyro_data.x  = RAW_GYRO_Y;
				gyro_data.z  = - RAW_GYRO_Z;
				break;
			case PIOS_MPU60X0_TOP_90DEG:
				accel_data.y = - RAW_ACCEL_Y;
				accel_data.x = RAW_ACCEL_X;
				accel_data.z = - RAW_ACCEL_Z;
				gyro_data.y  = - RAW_GYRO_Y;
				gyro_data.x  = RAW_GYRO_X;
				gyro_data.z  = - RAW_GYRO_Z;
				break;
			case PIOS_MPU60X0_TOP_180DEG:
				accel_data.y = - RAW_ACCEL_X;
				accel_data.x = - RAW_ACCEL_Y;
				accel_data.z = - RAW_ACCEL_Z;
				gyro_data.y  = - RAW_GYRO_X;
				gyro_data.x  = - RAW_GYRO_Y;
				gyro_data.z  = - RAW_GYRO_Z;
				break;
			case PIOS_MPU60X0_TOP_270DEG:
				accel_data.y = RAW_ACCEL_Y;
				accel_data.x = - RAW_ACCEL_X;
				accel_data.z = - RAW_ACCEL_Z;
				gyro_data.y  = RAW_GYRO_Y;
				gyro_data.x  = - RAW_GYRO_X;
				gyro_data.z  = - RAW_GYRO_Z;
				break;
			case PIOS_MPU60X0_BOTTOM_0DEG:
				accel_data.y = - RAW_ACCEL_X;
				accel_data.x = RAW_ACCEL_Y;
				accel_data.z = RAW_ACCEL_Z;
				gyro_data.y  = - RAW_GYRO_X;
				gyro_data.x  = RAW_GYRO_Y;
				gyro_data.z  = RAW_GYRO_Z;
				break;
			case PIOS_MPU60X0_BOTTOM_90DEG:
				accel_data.y = RAW_ACCEL_Y;
				accel_data.x = - RAW_ACCEL_X;
				accel_data.z = RAW_ACCEL_Z;
				gyro_data.y  = RAW_GYRO_Y;
				gyro_data.x  = - RAW_GYRO_X;
				gyro_data.z  = RAW_GYRO_Z;
				break;
			case PIOS_MPU60X0_BOTTOM_180DEG:
				accel_data.y = RAW_ACCEL_X;
				accel_data.x = - RAW_ACCEL_Y;
				accel_data.z = RAW_ACCEL_Z;
				gyro_data.y  = RAW_GYRO_X;
				gyro_data.x  = - RAW_GYRO_Y;
				gyro_data.z  = RAW_GYRO_Z;
				break;
			case PIOS_MPU60X0_BOTTOM_270DEG:
				accel_data.y = - RAW_ACCEL_Y;
				accel_data.x = RAW_ACCEL_X;
				accel_data.z = RAW_ACCEL_Z;
				gyro_data.y  = - RAW_GYRO_Y;
				gyro_data.x  = RAW_GYRO_X;
				gyro_data.z  = RAW_GYRO_Z;
				break;
			}

		    // Apply sensor scaling
			float temperature = (35.0f + ((float)RAW_TEMP + 512.0f) / 340.0f);

			float gyro_scale = PIOS_MPU6000_GetGyroScale();
			gyro_data.x *= gyro_scale;
			gyro_data.y *= gyro_scale;
			gyro_data.z *= gyro_scale;
			gyro_data.temperature = temperature;

			float accel_scale = PIOS_MPU6000_GetAccelScale();
			accel_data.x *= accel_scale;
			accel_data.y *= accel_scale;
			accel_data.z *= accel_scale;
			accel_data.temperature = temperature;

#ifndef SMALLF1
			// reset of the down-sampling variables
			TaskGyroSubSamplesCnt = 0;
			TaskAllSubSamplesCnt = 0;
			DoOutputFlg = false;
#endif

			PIOS_Queue_Send(pios_mpu6000_dev->accel_queue, &accel_data, 0);

			PIOS_Queue_Send(pios_mpu6000_dev->gyro_queue, &gyro_data, 0);

#ifndef SMALLF1
		}
#endif


#else // #if defined(PIOS_MPU6000_ACCEL)

		static struct pios_sensor_gyro_data gyro_data;

#ifndef SMALLF1
		static int32_t raw_temp = 0;

		// reset the variables, where the data is directly needed for the output queue and no CIC filtering (moving average) is applied  //TODO
		if ( (TaskGyroSubSamplesCnt == 1) && (gyro_downsampling_settings.LPF_Order == 0) ) {
			gyro_filter_xyz[X1].integrateState0 = 0.0;
			gyro_filter_xyz[Y1].integrateState0 = 0.0;
			gyro_filter_xyz[Z1].integrateState0 = 0.0;

			raw_temp = 0.0;
		}

		// data conversion & 1st order CIC and boxcar (order = 0) integration
		if (AllSamplingFlg == true) {
			gyro_filter_xyz[X1].integrateState0 += PACK_SENSOR_VALUE16(IDX_GYRO_XOUT_H, IDX_GYRO_XOUT_L);
			gyro_filter_xyz[Y1].integrateState0 += PACK_SENSOR_VALUE16(IDX_GYRO_YOUT_H, IDX_GYRO_YOUT_L);
			gyro_filter_xyz[Z1].integrateState0 += PACK_SENSOR_VALUE16(IDX_GYRO_ZOUT_H, IDX_GYRO_ZOUT_L);
			raw_temp                            += PACK_SENSOR_VALUE16(IDX_TEMP_OUT_H, IDX_TEMP_OUT_L);

			AllSamplingFlg = false;
		}
		else {
			gyro_filter_xyz[X1].integrateState0 += PACK_SENSOR_VALUE16(IDX2_GYRO_XOUT_H, IDX2_GYRO_XOUT_L);
			gyro_filter_xyz[Y1].integrateState0 += PACK_SENSOR_VALUE16(IDX2_GYRO_YOUT_H, IDX2_GYRO_YOUT_L);
			gyro_filter_xyz[Z1].integrateState0 += PACK_SENSOR_VALUE16(IDX2_GYRO_ZOUT_H, IDX2_GYRO_ZOUT_L);

		}

		// 1st order CIC and boxcar (order = 0) integration already done in the conversion above
		// for the higher order integration stages and all axes:
		for(uint8_t i=0; i< AXES_SIZE; i++) {
			if (gyro_downsampling_settings.LPF_Order > 1)
				cic_apply_higher_order_int_stage(&gyro_filter_xyz[i]);
		}


		// frequency decimation
		if (DoOutputFlg == true) {

			// CIC comb stage
			for(uint8_t i=0; i< AXES_SIZE; i++) {
				if (gyro_downsampling_settings.LPF_Order > 0) // only for CIC filter, not for boxcar
					cic_apply_comb_stage(&gyro_filter_xyz[i]);
			}
#endif // #ifndef SMALLF1

			// Rotate the sensor to OP convention.  The datasheet defines X as towards the right
			// and Y as forward.  OP convention transposes this.  Also the Z is defined negatively to our convention
			switch (pios_mpu6000_dev->cfg->orientation) {
			case PIOS_MPU60X0_TOP_0DEG:
				gyro_data.y = RAW_GYRO_X;
				gyro_data.x = RAW_GYRO_Y;
				break;
			case PIOS_MPU60X0_TOP_90DEG:
				gyro_data.y = - RAW_GYRO_Y;
				gyro_data.x = RAW_GYRO_X;
				break;
			case PIOS_MPU60X0_TOP_180DEG:
				gyro_data.y = - RAW_GYRO_X;
				gyro_data.x = - RAW_GYRO_Y;
				break;
			case PIOS_MPU60X0_TOP_270DEG:
				gyro_data.y = RAW_GYRO_Y;
				gyro_data.x = - RAW_GYRO_X;
				break;
			}

			gyro_data.z = - RAW_GYRO_Z;

			// Apply sensor scaling
			float temperature = (35.0f + ((float)RAW_TEMP + 512.0f) / 340.0f);

			float gyro_scale = PIOS_MPU6000_GetGyroScale();
			gyro_data.x *= gyro_scale;
			gyro_data.y *= gyro_scale;
			gyro_data.z *= gyro_scale;
			gyro_data.temperature = temperature;

#ifndef SMALLF1
			// reset of the gyro down-sampling variables
			TaskGyroSubSamplesCnt = 0;
			TaskAllSubSamplesCnt = 0;
			DoOutputFlg = false;
#endif

			PIOS_Queue_Send(pios_mpu6000_dev->gyro_queue, &gyro_data, 0);
#ifndef SMALLF1
		}
#endif

#endif /* PIOS_MPU6000_ACCEL */
	}
}


#ifndef SMALLF1
/**
 * Set the gyro down-sampling settings and store it locally for fast access without the overhead of importing a UAV Object
 */
void PIOS_MPU6000_SetGyroDownSamling(const uint8_t *gyro_downsampling)
{
	// get the unchecked data
	gyro_downsampling_settings.Drop = bound_min_max(gyro_downsampling[0], 1, 255);
	gyro_downsampling_settings.LPF = gyro_downsampling[1];
	gyro_downsampling_settings.LPF_Order = gyro_downsampling[2];
	gyro_downsampling_settings.LPF_DDelay = gyro_downsampling[3];

	for(uint8_t i=0; i< AXES_SIZE; i++) {
			cic_configure(&gyro_filter_xyz[i], gyro_downsampling_settings.LPF_Order, gyro_downsampling_settings.LPF_DDelay, gyro_downsampling_settings.LPF);
	}

	// if something was not in the limits,  it is corrected during the configure above
	gyro_downsampling_settings.LPF        = gyro_filter_xyz[0].filter_decimation;
	gyro_downsampling_settings.LPF_Order  = gyro_filter_xyz[0].filter_order;
	gyro_downsampling_settings.LPF_DDelay = gyro_filter_xyz[0].filter_ddelay;


	gyro_lpf_gain = cic_get_gain(&gyro_filter_xyz[0]);
}
#endif


#endif

/**
 * @}
 * @}
 */
