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

/* Global Variables */

enum pios_mpu6000_dev_magic {
	PIOS_MPU6000_DEV_MAGIC = 0x9da9b3ed,
};

#define PIOS_MPU6000_MAX_DOWNSAMPLE 1
struct mpu6000_dev {
	uint32_t spi_id;
	uint32_t slave_num;
	enum pios_mpu60x0_accel_range accel_range;
	enum pios_mpu60x0_range gyro_range;
	xQueueHandle gyro_queue;
	xQueueHandle accel_queue;
	const struct pios_mpu60x0_cfg * cfg;
	bool configured;
	enum pios_mpu6000_dev_magic magic;
};

//! Global structure for this device device
static struct mpu6000_dev * dev;

//! Private functions
static struct mpu6000_dev * PIOS_MPU6000_alloc(void);
static int32_t PIOS_MPU6000_Validate(struct mpu6000_dev * dev);
static void PIOS_MPU6000_Config(struct pios_mpu60x0_cfg const * cfg);
static int32_t PIOS_MPU6000_ClaimBus();
static int32_t PIOS_MPU6000_ReleaseBus();
static int32_t PIOS_MPU6000_SetReg(uint8_t address, uint8_t buffer);
static int32_t PIOS_MPU6000_GetReg(uint8_t address);

#define GRAV 9.81f

/**
 * @brief Allocate a new device
 */
static struct mpu6000_dev * PIOS_MPU6000_alloc(void)
{
	struct mpu6000_dev * mpu6000_dev;
	
	mpu6000_dev = (struct mpu6000_dev *)pvPortMalloc(sizeof(*mpu6000_dev));
	if (!mpu6000_dev) return (NULL);
	
	mpu6000_dev->magic = PIOS_MPU6000_DEV_MAGIC;

	mpu6000_dev->configured = false;
	
	mpu6000_dev->accel_queue = xQueueCreate(PIOS_MPU6000_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_gyro_data));
	if(mpu6000_dev->accel_queue == NULL) {
		vPortFree(mpu6000_dev);
		return NULL;
	}

	mpu6000_dev->gyro_queue = xQueueCreate(PIOS_MPU6000_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_gyro_data));
	if(mpu6000_dev->gyro_queue == NULL) {
		vPortFree(mpu6000_dev);
		return NULL;
	}

	return(mpu6000_dev);
}

/**
 * @brief Validate the handle to the spi device
 * @returns 0 for valid device or -1 otherwise
 */
static int32_t PIOS_MPU6000_Validate(struct mpu6000_dev * dev)
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
int32_t PIOS_MPU6000_Init(uint32_t spi_id, uint32_t slave_num, const struct pios_mpu60x0_cfg * cfg)
{
	dev = PIOS_MPU6000_alloc();
	if(dev == NULL)
		return -1;
	
	dev->spi_id = spi_id;
	dev->slave_num = slave_num;
	dev->cfg = cfg;

	/* Configure the MPU6000 Sensor */
	PIOS_SPI_SetClockSpeed(dev->spi_id, PIOS_SPI_PRESCALER_256);
	PIOS_MPU6000_Config(cfg);
	PIOS_SPI_SetClockSpeed(dev->spi_id, PIOS_SPI_PRESCALER_16);

	/* Set up EXTI line */
	PIOS_EXTI_Init(cfg->exti_cfg);

	PIOS_SENSORS_Register(PIOS_SENSOR_ACCEL, dev->accel_queue);
	PIOS_SENSORS_Register(PIOS_SENSOR_GYRO, dev->gyro_queue);

	return 0;
}

/**
 * @brief Initialize the MPU6000 3-axis gyro sensor
 * \return none
 * \param[in] PIOS_MPU6000_ConfigTypeDef struct to be used to configure sensor.
*
*/
static void PIOS_MPU6000_Config(struct pios_mpu60x0_cfg const * cfg)
{
	PIOS_MPU6000_ClaimBus();
	PIOS_DELAY_WaitmS(1);
	PIOS_MPU6000_ReleaseBus();
	PIOS_DELAY_WaitmS(10);

	// Reset chip
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, 0x80 | cfg->Pwr_mgmt_clk);
	do {
		PIOS_DELAY_WaitmS(5);
	} while (PIOS_MPU6000_GetReg(PIOS_MPU60X0_PWR_MGMT_REG) & 0x80);

	PIOS_DELAY_WaitmS(25);

	// Reset chip and fifo
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_USER_CTRL_REG, 0x80 | 0x01 | 0x02 | 0x04);;
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

	// FIFO storage
#if defined(PIOS_MPU6000_ACCEL)
	// Set the accel scale
	PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
	
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_FIFO_EN_REG, cfg->Fifo_store | PIOS_MPU60X0_ACCEL_OUT);
#else
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_FIFO_EN_REG, cfg->Fifo_store);
#endif

	// Sample rate divider
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_SMPLRT_DIV_REG, cfg->Smpl_rate_div);

	// Digital low-pass filter and scale
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_DLPF_CFG_REG, cfg->filter);

	// Digital low-pass filter and scale
	PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);

	// Interrupt configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_USER_CTRL_REG, cfg->User_ctl);

	//Power management configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_PWR_MGMT_REG, cfg->Pwr_mgmt_clk);

	// Interrupt configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_INT_CFG_REG, cfg->interrupt_cfg);

	// Interrupt configuration
	PIOS_MPU6000_SetReg(PIOS_MPU60X0_INT_EN_REG, cfg->interrupt_en);

	dev->configured = true;
}

/**
 * Set the gyro range and store it locally for scaling
 */
void PIOS_MPU6000_SetGyroRange(enum pios_mpu60x0_range gyro_range)
{
	while (PIOS_MPU6000_SetReg(PIOS_MPU60X0_GYRO_CFG_REG, gyro_range) != 0);
	dev->gyro_range = gyro_range;
}

/**
 * Set the accel range and store it locally for scaling
 */
void PIOS_MPU6000_SetAccelRange(enum pios_mpu60x0_accel_range accel_range)
{
	while (PIOS_MPU6000_SetReg(PIOS_MPU60X0_ACCEL_CFG_REG, accel_range) != 0);
	dev->accel_range = accel_range;
}

/**
 * @brief Claim the SPI bus for the accel communications and select this chip
 * @return 0 if successful, -1 for invalid device, -2 if unable to claim bus
 */
static int32_t PIOS_MPU6000_ClaimBus()
{
	if(PIOS_MPU6000_Validate(dev) != 0)
		return -1;
	
	if(PIOS_SPI_ClaimBus(dev->spi_id) != 0)
		return -2;
	
	PIOS_SPI_RC_PinSet(dev->spi_id,dev->slave_num,0);
	return 0;
}

/**
 * @brief Release the SPI bus for the accel communications and end the transaction
 * @return 0 if successful
 */
static int32_t PIOS_MPU6000_ReleaseBus()
{
	if(PIOS_MPU6000_Validate(dev) != 0)
		return -1;
	
	PIOS_SPI_RC_PinSet(dev->spi_id,dev->slave_num,1);
	
	return PIOS_SPI_ReleaseBus(dev->spi_id);
}

/**
 * @brief Read a register from MPU6000
 * @returns The register value or -1 if failure to get bus
 * @param reg[in] Register address to be read
 */
static int32_t PIOS_MPU6000_GetReg(uint8_t reg)
{
	uint8_t data;
	
	if(PIOS_MPU6000_ClaimBus() != 0)
		return -1;	
	
	PIOS_SPI_TransferByte(dev->spi_id,(0x80 | reg) ); // request byte
	data = PIOS_SPI_TransferByte(dev->spi_id,0 );     // receive response
	
	PIOS_MPU6000_ReleaseBus();
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
	if(PIOS_MPU6000_ClaimBus() != 0)
		return -1;
	
	if(PIOS_SPI_TransferByte(dev->spi_id, 0x7f & reg) != 0) {
		PIOS_MPU6000_ReleaseBus();
		return -2;
	}
	
	if(PIOS_SPI_TransferByte(dev->spi_id, data) != 0) {
		PIOS_MPU6000_ReleaseBus();
		return -3;
	}
	
	PIOS_MPU6000_ReleaseBus();
	
	return 0;
}

/*
 * @brief Read the identification bytes from the MPU6000 sensor
 * \return ID read from MPU6000 or -1 if failure
*/
static int32_t PIOS_MPU6000_ReadID()
{
	int32_t mpu6000_id = PIOS_MPU6000_GetReg(PIOS_MPU60X0_WHOAMI);
	if(mpu6000_id < 0)
		return -1;
	return mpu6000_id;
}

/**
 * Get the gyro scale based on the active device settings
 * @return Scale in (deg/s) / LSB
 */
static float PIOS_MPU6000_GetGyroScale() 
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
 * Get the accel scale based on the active settings
 * @returns Scale in (m/s^2) / LSB
 */
static float PIOS_MPU6000_GetAccelScale()
{
	switch (dev->accel_range) {
		case PIOS_MPU60X0_ACCEL_2G:
			return GRAV / 16384.0f;
		case PIOS_MPU60X0_ACCEL_4G:
			return GRAV / 8192.0f;
		case PIOS_MPU60X0_ACCEL_8G:
			return GRAV / 4096.0f;
		case PIOS_MPU60X0_ACCEL_16G:
			return GRAV / 2048.0f;
	}
	return 0;
}

/**
 * @brief Run self-test operation.
 * \return 0 if test succeeded
 * \return non-zero value if test succeeded
 */
int32_t PIOS_MPU6000_Test(void)
{
	/* Verify that ID matches (MPU6000 ID is 0x69) */
	int32_t mpu6000_id = PIOS_MPU6000_ReadID();
	if(mpu6000_id < 0)
		return -1;
	
	if(mpu6000_id != 0x68)
		return -2;
	
	return 0;
}

/**
 * @brief Get the number of elements in the FIFO.
 * \return number of elements if successful
 * \return negative value if failed
 */
static int32_t PIOS_MPU6000_FifoDepth(void)
{
	uint8_t mpu6000_send_buf[3] = {PIOS_MPU60X0_FIFO_CNT_MSB | 0x80, 0, 0};
	uint8_t mpu6000_rec_buf[3];

	if(PIOS_MPU6000_ClaimBus() != 0)
		return -1;

	if(PIOS_SPI_TransferBlock(dev->spi_id, &mpu6000_send_buf[0], &mpu6000_rec_buf[0], sizeof(mpu6000_send_buf), NULL) < 0) {
		PIOS_MPU6000_ReleaseBus();
		return -1;
	}

	PIOS_MPU6000_ReleaseBus();

	return (mpu6000_rec_buf[1] << 8) | mpu6000_rec_buf[2];
}


/**
 * @brief Get the status code.
 * \return the status code if successful
 * \return negative value if failed
 */
static int32_t PIOS_MPU6000_GetStatus(void)
{
	uint8_t mpu6000_send_buf[2] = {PIOS_MPU60X0_INT_STATUS_REG | 0x80, 0};
	uint8_t mpu6000_rec_buf[2];

	if(PIOS_MPU6000_ClaimBus() != 0)
		return -1;

	if(PIOS_SPI_TransferBlock(dev->spi_id, &mpu6000_send_buf[0], &mpu6000_rec_buf[0], sizeof(mpu6000_send_buf), NULL) < 0) {
		PIOS_MPU6000_ReleaseBus();
		return -1;
	}

	PIOS_MPU6000_ReleaseBus();

	return mpu6000_rec_buf[1];
}

/**
* @brief IRQ Handler.  Read all the data from onboard buffer
*/
bool PIOS_MPU6000_IRQHandler(void)
{
	if (PIOS_MPU6000_Validate(dev) != 0)
		return false;

	if(!dev->configured)
		return false;

	int32_t status = PIOS_MPU6000_GetStatus();
	if (status & PIOS_MPU60X0_INT_STATUS_OVERFLOW) {
		dev->configured = false;
		return false;
	}

	uint32_t mpu6000_count = PIOS_MPU6000_FifoDepth();
	if(mpu6000_count < sizeof(struct pios_mpu60x0_data))
		return false;

	if(PIOS_MPU6000_ClaimBus() != 0)
		return false;

	uint8_t mpu6000_send_buf[1+sizeof(struct pios_mpu60x0_data)] = {PIOS_MPU60X0_FIFO_REG | 0x80, 0, 0, 0, 0, 0, 0, 0, 0};
	uint8_t mpu6000_rec_buf[1+sizeof(struct pios_mpu60x0_data)];

	if(PIOS_SPI_TransferBlock(dev->spi_id, &mpu6000_send_buf[0], &mpu6000_rec_buf[0], sizeof(mpu6000_send_buf), NULL) < 0) {
		PIOS_MPU6000_ReleaseBus();
		return false;
	}

	PIOS_MPU6000_ReleaseBus();

	struct pios_mpu60x0_data data;

	// In the case where extras samples backed up grabbed an extra
	if (mpu6000_count >= (sizeof(data) * 2)) {
		if(PIOS_MPU6000_ClaimBus() != 0)
			return false;		
		
		if(PIOS_SPI_TransferBlock(dev->spi_id, &mpu6000_send_buf[0], &mpu6000_rec_buf[0], sizeof(mpu6000_send_buf), NULL) < 0) {
			PIOS_MPU6000_ReleaseBus();
			return false;
		}
		
		PIOS_MPU6000_ReleaseBus();
	}
	
	// Rotate the sensor to OP convention.  The datasheet defines X as towards the right
	// and Y as forward.  OP convention transposes this.  Also the Z is defined negatively
	// to our convention

#if defined(PIOS_MPU6000_ACCEL)

	// Currently we only support rotations on top so switch X/Y accordingly
	struct pios_sensor_accel_data accel_data;
	struct pios_sensor_gyro_data gyro_data;

	switch(dev->cfg->orientation) {
	case PIOS_MPU60X0_TOP_0DEG:
		accel_data.y = (int16_t) (mpu6000_rec_buf[1] << 8 | mpu6000_rec_buf[2]);    // chip X
		accel_data.x = (int16_t) (mpu6000_rec_buf[3] << 8 | mpu6000_rec_buf[4]);    // chip Y
		gyro_data.y  = (int16_t) (mpu6000_rec_buf[9] << 8  | mpu6000_rec_buf[10]);  // chip X
		gyro_data.x  = (int16_t) (mpu6000_rec_buf[11] << 8 | mpu6000_rec_buf[12]);  // chip Y
		break;
	case PIOS_MPU60X0_TOP_90DEG:
		accel_data.y = (int16_t) -(mpu6000_rec_buf[3] << 8 | mpu6000_rec_buf[4]);   // chip Y
		accel_data.x = (int16_t)  (mpu6000_rec_buf[1] << 8 | mpu6000_rec_buf[2]);   // chip X
		gyro_data.y  = (int16_t) -(mpu6000_rec_buf[11] << 8 | mpu6000_rec_buf[12]); // chip Y
		gyro_data.x  = (int16_t)  (mpu6000_rec_buf[9] << 8  | mpu6000_rec_buf[10]); // chip X
		break;
	case PIOS_MPU60X0_TOP_180DEG:
		accel_data.y = (int16_t) -(mpu6000_rec_buf[1] << 8 | mpu6000_rec_buf[2]);   // chip X
		accel_data.x = (int16_t) -(mpu6000_rec_buf[3] << 8 | mpu6000_rec_buf[4]);   // chip Y
		gyro_data.y  = (int16_t) -(mpu6000_rec_buf[9] << 8  | mpu6000_rec_buf[10]); // chip X
		gyro_data.x  = (int16_t) -(mpu6000_rec_buf[11] << 8 | mpu6000_rec_buf[12]); // chip Y
		break;
	case PIOS_MPU60X0_TOP_270DEG:
		accel_data.y = (int16_t)  (mpu6000_rec_buf[3] << 8 | mpu6000_rec_buf[4]);   // chip Y
		accel_data.x = (int16_t) -(mpu6000_rec_buf[1] << 8 | mpu6000_rec_buf[2]);   // chip X
		gyro_data.y  = (int16_t)  (mpu6000_rec_buf[11] << 8 | mpu6000_rec_buf[12]); // chip Y
		gyro_data.x  = (int16_t) -(mpu6000_rec_buf[9] << 8  | mpu6000_rec_buf[10]); // chip X
		break;
	}
	gyro_data.z  = (int16_t) -(mpu6000_rec_buf[13] << 8 | mpu6000_rec_buf[14]);
	accel_data.z = (int16_t) -(mpu6000_rec_buf[5] << 8 | mpu6000_rec_buf[6]);

	int16_t raw_temp = mpu6000_rec_buf[7] << 8 | mpu6000_rec_buf[8];
	float temperature = 35.0f + ((float) raw_temp + 512.0f) / 340.0f;

	// Apply sensor scaling
	float accel_scale = PIOS_MPU6000_GetAccelScale();
	accel_data.x *= accel_scale;
	accel_data.y *= accel_scale;
	accel_data.z *= accel_scale;
	accel_data.temperature = temperature;

	float gyro_scale = PIOS_MPU6000_GetGyroScale();
	gyro_data.x *= gyro_scale;
	gyro_data.y *= gyro_scale;
	gyro_data.z *= gyro_scale;
	gyro_data.temperature = temperature;

	portBASE_TYPE xHigherPriorityTaskWoken_accel;
	xQueueSendToBackFromISR(dev->accel_queue, (void *) &accel_data, &xHigherPriorityTaskWoken_accel);

	portBASE_TYPE xHigherPriorityTaskWoken_gyro;
	xQueueSendToBackFromISR(dev->gyro_queue, (void *) &gyro_data, &xHigherPriorityTaskWoken_gyro);

	return (xHigherPriorityTaskWoken_accel == pdTRUE) || (xHigherPriorityTaskWoken_gyro == pdTRUE);

#else

	struct pios_sensor_gyro_data gyro_data;
	switch(dev->cfg->orientation) {
	case PIOS_MPU60X0_TOP_0DEG:
		gyro_data.y  = (int16_t) (mpu6000_rec_buf[3] << 8 | mpu6000_rec_buf[4]);
		gyro_data.x  = (int16_t) (mpu6000_rec_buf[5] << 8 | mpu6000_rec_buf[6]);
		break;
	case PIOS_MPU60X0_TOP_90DEG:
		gyro_data.y  = (int16_t) -(mpu6000_rec_buf[5] << 8 | mpu6000_rec_buf[6]); // chip Y
		gyro_data.x  = (int16_t)  (mpu6000_rec_buf[3] << 8 | mpu6000_rec_buf[4]); // chip X
		break;
	case PIOS_MPU60X0_TOP_180DEG:
		gyro_data.y  = (int16_t) -(mpu6000_rec_buf[3] << 8 | mpu6000_rec_buf[4]);
		gyro_data.x  = (int16_t) -(mpu6000_rec_buf[5] << 8 | mpu6000_rec_buf[6]);
		break;
	case PIOS_MPU60X0_TOP_270DEG:
		gyro_data.y  = (int16_t)  (mpu6000_rec_buf[5] << 8 | mpu6000_rec_buf[6]); // chip Y
		gyro_data.x  = (int16_t) -(mpu6000_rec_buf[3] << 8 | mpu6000_rec_buf[4]); // chip X
		break;
	}
	gyro_data.z = (int16_t) -(mpu6000_rec_buf[7] << 8 | mpu6000_rec_buf[8]);

	int32_t raw_temp = mpu6000_rec_buf[1] << 8 | mpu6000_rec_buf[2];
	float temperature = 35.0f + ((float) raw_temp + 512.0f) / 340.0f;

	// Apply sensor scaling
	float gyro_scale = PIOS_MPU6000_GetGyroScale();
	gyro_data.x *= gyro_scale;
	gyro_data.y *= gyro_scale;
	gyro_data.z *= gyro_scale;
	gyro_data.temperature = temperature;

	portBASE_TYPE xHigherPriorityTaskWoken_gyro;
	xQueueSendToBackFromISR(dev->gyro_queue, (void *) &gyro_data, &xHigherPriorityTaskWoken_gyro);

	return (xHigherPriorityTaskWoken_gyro == pdTRUE);

#endif

}

#endif

/**
 * @}
 * @}
 */
