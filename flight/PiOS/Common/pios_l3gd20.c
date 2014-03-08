/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_L3GD20 L3GD20 Functions
 * @brief Deals with the hardware interface to the 3-axis gyro
 * @{
 *
 * @file       pios_l3gd20.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      L3GD20 3-axis gyro chip
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

#if defined(PIOS_INCLUDE_L3GD20)

#include "fifo_buffer.h"

/* Global Variables */
enum pios_l3gd20_dev_magic {
    PIOS_L3GD20_DEV_MAGIC = 0x9d39bced,
};

#define PIOS_L3GD20_QUEUESIZE 2

//! Local types
struct l3gd20_dev {
	uint32_t spi_id;
	uint32_t slave_num;
	xQueueHandle queue;
	const struct pios_l3gd20_cfg *cfg;
	enum pios_l3gd20_filter bandwidth;
	enum pios_l3gd20_range range;
	enum pios_l3gd20_dev_magic magic;
	volatile bool configured;
};

struct pios_l3gd20_data {
	int16_t gyro_x;
	int16_t gyro_y;
	int16_t gyro_z;
	int16_t temperature;
};

//! Global structure for this device device
static struct l3gd20_dev *pios_l3gd20_dev;

//! Private functions
static struct l3gd20_dev *PIOS_L3GD20_alloc(void);
static int32_t PIOS_L3GD20_Validate(struct l3gd20_dev *dev);
static void PIOS_L3GD20_Config(const struct pios_l3gd20_cfg *cfg);
static int32_t PIOS_L3GD20_SetReg(uint8_t address, uint8_t buffer);
static int32_t PIOS_L3GD20_GetReg(uint8_t address);
static int32_t PIOS_L3GD20_GetRegIsr(uint8_t address, bool *woken);
static int32_t PIOS_L3GD20_ClaimBus();
static int32_t PIOS_L3GD20_ClaimBusIsr();
static int32_t PIOS_L3GD20_ReleaseBus();
static int32_t PIOS_L3GD20_ReleaseBusIsr();
static int32_t PIOS_L3GD20_ReadGyros(struct pios_l3gd20_data *buffer);

/* Local Variables */

/**
 * @brief Allocate a new device
 */
static struct l3gd20_dev *PIOS_L3GD20_alloc(void)
{
	struct l3gd20_dev *l3gd20_dev;

	l3gd20_dev = (struct l3gd20_dev *)PIOS_malloc(sizeof(*l3gd20_dev));

	if (!l3gd20_dev) return (NULL);

	l3gd20_dev->magic = PIOS_L3GD20_DEV_MAGIC;

	l3gd20_dev->configured = false;

	l3gd20_dev->queue = xQueueCreate(PIOS_L3GD20_QUEUESIZE, sizeof(struct pios_sensor_gyro_data));

	if (l3gd20_dev->queue == NULL) {
		vPortFree(l3gd20_dev);
		return NULL;
	}

	return l3gd20_dev;
}

/**
 * @brief Validate the handle to the spi device
 * @returns 0 for valid device or -1 otherwise
 */
static int32_t PIOS_L3GD20_Validate(struct l3gd20_dev *dev)
{
	if (dev == NULL)
		return -1;

	if (dev->magic != PIOS_L3GD20_DEV_MAGIC)
		return -2;

	if (dev->spi_id == 0)
		return -3;

	return 0;
}

/**
 * @brief Initialize the L3GD20 3-axis gyro sensor.
 * @return none
 */
int32_t PIOS_L3GD20_Init(uint32_t spi_id, uint32_t slave_num, const struct pios_l3gd20_cfg *cfg)
{
	pios_l3gd20_dev = PIOS_L3GD20_alloc();

	if (pios_l3gd20_dev == NULL)
		return -1;

	pios_l3gd20_dev->spi_id = spi_id;
	pios_l3gd20_dev->slave_num = slave_num;
	pios_l3gd20_dev->cfg = cfg;

	/* Configure the L3GD20 Sensor */
	PIOS_L3GD20_Config(cfg);

	/* Set up EXTI */
	PIOS_EXTI_Init(cfg->exti_cfg);

	// An initial read is needed to get it running
	struct pios_l3gd20_data data;
	PIOS_L3GD20_ReadGyros(&data);

	PIOS_SENSORS_Register(PIOS_SENSOR_GYRO, pios_l3gd20_dev->queue);

	return 0;
}

/**
 * @brief Initialize the L3GD20 3-axis gyro sensor
 * \return none
 * \param[in] PIOS_L3GD20_ConfigTypeDef struct to be used to configure sensor.
*
*/
static void PIOS_L3GD20_Config(const struct pios_l3gd20_cfg *cfg)
{
	// This register enables the channels and sets the bandwidth
	while (PIOS_L3GD20_SetReg(PIOS_L3GD20_CTRL_REG1, PIOS_L3GD20_CTRL1_380HZ_100HZ |
	                          PIOS_L3GD20_CTRL1_PD | PIOS_L3GD20_CTRL1_ZEN |
	                          PIOS_L3GD20_CTRL1_YEN | PIOS_L3GD20_CTRL1_XEN) != 0);

	// Disable the high pass filters
	while (PIOS_L3GD20_SetReg(PIOS_L3GD20_CTRL_REG2, 0) != 0);

	// Set int2 to go high on data ready
	while (PIOS_L3GD20_SetReg(PIOS_L3GD20_CTRL_REG3, 0x08) != 0);

	// Select SPI interface, 500 deg/s, endianness?
	while (PIOS_L3GD20_SetRange(cfg->range) != 0);

	// disable HPF
	while (PIOS_L3GD20_SetReg(PIOS_L3GD20_CTRL_REG5, 0x00) != 0);

	pios_l3gd20_dev->configured = true;
}

/**
 * @brief Sets the maximum range of the L3GD20
 * @returns 0 for success, -1 for invalid device, -2 if unable to set register
 */
int32_t PIOS_L3GD20_SetRange(enum pios_l3gd20_range range)
{
	if (PIOS_L3GD20_Validate(pios_l3gd20_dev) != 0)
		return -1;

	pios_l3gd20_dev->range = range;

	if (PIOS_L3GD20_SetReg(PIOS_L3GD20_CTRL_REG4, pios_l3gd20_dev->range) != 0)
		return -2;

	switch(range) {
	case PIOS_L3GD20_SCALE_250_DEG:
		PIOS_SENSORS_SetMaxGyro(250);
		break;
	case PIOS_L3GD20_SCALE_500_DEG:
		PIOS_SENSORS_SetMaxGyro(500);
		break;
	case PIOS_L3GD20_SCALE_2000_DEG:
		PIOS_SENSORS_SetMaxGyro(2000);
		break;
	}
	return 0;
}

/**
 * @brief Claim the SPI bus for the accel communications and select this chip
 * @return 0 if successful, -1 for invalid device, -2 if unable to claim bus
 */
static int32_t PIOS_L3GD20_ClaimBus()
{
	if (PIOS_L3GD20_Validate(pios_l3gd20_dev) != 0)
		return -1;

	if (PIOS_SPI_ClaimBus(pios_l3gd20_dev->spi_id) != 0)
		return -2;

	PIOS_SPI_RC_PinSet(pios_l3gd20_dev->spi_id, pios_l3gd20_dev->slave_num, 0);

	return 0;
}

/**
 * @brief Claim the SPI bus for the accel communications and select this chip
 * \param[in] pointer which receives if a task has been woken
 * @return 0 if successful, -1 for invalid device, -2 if unable to claim bus
 */
static int32_t PIOS_L3GD20_ClaimBusIsr(bool *woken)
{
	if (PIOS_L3GD20_Validate(pios_l3gd20_dev) != 0)
		return -1;

	if (PIOS_SPI_ClaimBusISR(pios_l3gd20_dev->spi_id, woken) < 0)
		return -2;

	PIOS_SPI_RC_PinSet(pios_l3gd20_dev->spi_id, pios_l3gd20_dev->slave_num, 0);

	return 0;
}

/**
 * @brief Release the SPI bus for the accel communications and end the transaction
 * @return 0 if successful, -1 for invalid device
 */
static int32_t PIOS_L3GD20_ReleaseBus()
{
	if (PIOS_L3GD20_Validate(pios_l3gd20_dev) != 0)
		return -1;

	PIOS_SPI_RC_PinSet(pios_l3gd20_dev->spi_id, pios_l3gd20_dev->slave_num, 1);

	return PIOS_SPI_ReleaseBus(pios_l3gd20_dev->spi_id);
}

/**
 * @brief Release the SPI bus for the accel communications and end the transaction
 * \param[in] pointer which receives if a task has been woken
 * @return 0 if successful, -1 for invalid device
 */
static int32_t PIOS_L3GD20_ReleaseBusIsr(bool *woken)
{
	if (PIOS_L3GD20_Validate(pios_l3gd20_dev) != 0)
		return -1;

	PIOS_SPI_RC_PinSet(pios_l3gd20_dev->spi_id, pios_l3gd20_dev->slave_num, 1);

	return PIOS_SPI_ReleaseBusISR(pios_l3gd20_dev->spi_id, woken);
}

/**
 * @brief Read a register from L3GD20
 * @returns The register value or -1 if failure to get bus
 * @param reg[in] Register address to be read
 */
static int32_t PIOS_L3GD20_GetReg(uint8_t reg)
{
	uint8_t data;

	if (PIOS_L3GD20_ClaimBus() != 0)
		return -1;

	PIOS_SPI_TransferByte(pios_l3gd20_dev->spi_id, (0x80 | reg)); // request byte
	data = PIOS_SPI_TransferByte(pios_l3gd20_dev->spi_id, 0);     // receive response

	PIOS_L3GD20_ReleaseBus();

	return data;
}

/**
 * @brief Read a register from L3GD20 from ISR context
 * @returns The register value or -1 if failure to get bus
 * @param reg[in] Register address to be read
 * \param[in] task woken
 */
static int32_t PIOS_L3GD20_GetRegIsr(uint8_t reg, bool *woken)
{
	uint8_t data;

	if (PIOS_L3GD20_ClaimBusIsr(woken) != 0)
		return -1;

	PIOS_SPI_TransferByte(pios_l3gd20_dev->spi_id, (0x80 | reg)); // request byte
	data = PIOS_SPI_TransferByte(pios_l3gd20_dev->spi_id, 0);     // receive response

	PIOS_L3GD20_ReleaseBusIsr(woken);

	return data;
}

/**
 * @brief Writes one byte to the L3GD20
 * \param[in] reg Register address
 * \param[in] data Byte to write
 * \return 0 if operation was successful
 * \return -1 if unable to claim SPI bus
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_L3GD20_SetReg(uint8_t reg, uint8_t data)
{
	if (PIOS_L3GD20_ClaimBus() != 0)
		return -1;

	PIOS_SPI_TransferByte(pios_l3gd20_dev->spi_id, 0x7f & reg);
	PIOS_SPI_TransferByte(pios_l3gd20_dev->spi_id, data);

	PIOS_L3GD20_ReleaseBus();

	return 0;
}

/**
 * @brief Read current X, Z, Y values (in that order)
 * \param[out] int16_t array of size 3 to store X, Z, and Y magnetometer readings
 * \returns The number of samples remaining in the fifo
 */
static int32_t PIOS_L3GD20_ReadGyros(struct pios_l3gd20_data *data)
{
	uint8_t buf[7] = { PIOS_L3GD20_GYRO_X_OUT_LSB | 0x80 | 0x40, 0, 0, 0, 0, 0, 0 };
	uint8_t rec[7];

	if (PIOS_L3GD20_ClaimBus() != 0)
		return -1;

	if (PIOS_SPI_TransferBlock(pios_l3gd20_dev->spi_id, &buf[0], &rec[0], sizeof(buf), NULL) < 0) {
		PIOS_L3GD20_ReleaseBus();
		data->gyro_x = 0;
		data->gyro_y = 0;
		data->gyro_z = 0;
		data->temperature = 0;
		return -2;
	}

	PIOS_L3GD20_ReleaseBus();

	memcpy((uint8_t *)&data->gyro_x, &rec[1], 6);
	data->temperature = PIOS_L3GD20_GetReg(PIOS_L3GD20_OUT_TEMP);

	return 0;
}

/**
 * @brief Read the identification bytes from the L3GD20 sensor
 * \return ID read from L3GD20 or -1 if failure
*/
int32_t PIOS_L3GD20_ReadID()
{
	int32_t l3gd20_id = PIOS_L3GD20_GetReg(PIOS_L3GD20_WHOAMI);

	if (l3gd20_id < 0)
		return -1;

	return l3gd20_id;
}

static float PIOS_L3GD20_GetScale()
{
	if (PIOS_L3GD20_Validate(pios_l3gd20_dev) != 0)
		return -1;

	switch (pios_l3gd20_dev->range) {
	case PIOS_L3GD20_SCALE_250_DEG:
		return 0.00875f;
	case PIOS_L3GD20_SCALE_500_DEG:
		return 0.01750f;
	case PIOS_L3GD20_SCALE_2000_DEG:
		return 0.070f;
	}

	return 0;
}

/**
 * @brief Run self-test operation.
 * \return 0 if test succeeded
 * \return non-zero value if test succeeded
 */
uint8_t PIOS_L3GD20_Test(void)
{
	int32_t l3gd20_id = PIOS_L3GD20_ReadID();

	if (l3gd20_id < 0)
		return -1;

	uint8_t id = l3gd20_id;

	if (id == 0xD4)
		return 0;

	return -2;
}

/**
* @brief IRQ Handler.  Read all the data from onboard buffer
*/
bool PIOS_L3GD20_IRQHandler(void)
{
	if (PIOS_L3GD20_Validate(pios_l3gd20_dev) != 0 || pios_l3gd20_dev->configured == false)
		return false;

	struct pios_l3gd20_data data;
	uint8_t buf[7] = { PIOS_L3GD20_GYRO_X_OUT_LSB | 0x80 | 0x40, 0, 0, 0, 0, 0, 0 };
	uint8_t rec[7];

	/* This code duplicates ReadGyros above but uses ClaimBusIsr */
	bool woken = false;

	if (PIOS_L3GD20_ClaimBusIsr(&woken) != 0)
		return woken;

	if (PIOS_SPI_TransferBlock(pios_l3gd20_dev->spi_id, &buf[0], &rec[0], sizeof(buf), NULL) < 0) {
		PIOS_L3GD20_ReleaseBusIsr(&woken);
		return woken;
	}

	PIOS_L3GD20_ReleaseBusIsr(&woken);

	memcpy((uint8_t *)&data.gyro_x, &rec[1], 6);

	// TODO: This reordering is specific to the FlyingF3 chip placement.  Whenever
	// this code is used on another board add an orientation mapping to the configuration
	struct pios_sensor_gyro_data normalized_data;
	float scale = PIOS_L3GD20_GetScale();
	normalized_data.y = data.gyro_x * scale;
	normalized_data.x = data.gyro_y * scale;
	normalized_data.z = -data.gyro_z * scale;
	normalized_data.temperature = PIOS_L3GD20_GetRegIsr(PIOS_L3GD20_OUT_TEMP, &woken);

	portBASE_TYPE xHigherPriorityTaskWoken;
	xQueueSendToBackFromISR(pios_l3gd20_dev->queue, (void *)&normalized_data, &xHigherPriorityTaskWoken);

	return woken || (xHigherPriorityTaskWoken == pdTRUE);
}

#endif /* PIOS_INCLUDE_L3GD20 */

/**
 * @}
 * @}
 */
