/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_HMC5983 HMC5983 Functions
 * @brief Deals with the hardware interface to the magnetometers
 * @{
 * @file       pios_hmc5983.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      HMC5983 Magnetic Sensor Functions from AHRS
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

#if defined(PIOS_INCLUDE_HMC5983)

/* Private constants */
#define HMC5983_TASK_PRIORITY        (tskIDLE_PRIORITY + configMAX_PRIORITIES - 1)  // max priority
#define HMC5983_TASK_STACK	         (512 / 4)
#define PIOS_HMC5983_MAX_DOWNSAMPLE  1

/* Global Variables */

/* Local Types */
enum pios_hmc5983_dev_magic {
	PIOS_HMC5983_DEV_MAGIC = 0x3e8f18ac,
};

struct hmc5983_dev {
	uint32_t spi_id;
	uint32_t slave_num;
	const struct pios_hmc5983_cfg *cfg;
	xQueueHandle queue;
	xTaskHandle task;
	xSemaphoreHandle data_ready_sema;
	enum pios_hmc5983_dev_magic magic;
};

/* Local Variables */
static int32_t PIOS_HMC5983_Config(const struct pios_hmc5983_cfg *cfg);
static int32_t PIOS_HMC5983_ClaimBus(void);
static int32_t PIOS_HMC5983_ReleaseBus(void);
static int32_t PIOS_HMC5983_Read(uint8_t address, uint8_t *buffer, uint8_t len);
static int32_t PIOS_HMC5983_Write(uint8_t address, uint8_t buffer);
static void PIOS_HMC5983_Task(void *parameters);

static struct hmc5983_dev *dev;

/**
 * @brief Allocate a new device
 */
static struct hmc5983_dev *PIOS_HMC5983_alloc(void) {
	struct hmc5983_dev *hmc5983_dev;

	hmc5983_dev = (struct hmc5983_dev *)PIOS_malloc(sizeof(*hmc5983_dev));
	if (!hmc5983_dev) return (NULL);

	hmc5983_dev->magic = PIOS_HMC5983_DEV_MAGIC;

	hmc5983_dev->queue = xQueueCreate(PIOS_HMC5983_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_mag_data));
	if (hmc5983_dev->queue == NULL) {
		vPortFree(hmc5983_dev);
		return NULL;
	}

	hmc5983_dev->data_ready_sema = xSemaphoreCreateMutex();
	if (hmc5983_dev->data_ready_sema == NULL) {
		vPortFree(hmc5983_dev);
		return NULL;
	}

	return(hmc5983_dev);
}

/**
 * @brief Validate the handle to the spi device
 * @returns 0 for valid device or -1 otherwise
 */
static int32_t PIOS_HMC5983_Validate(struct hmc5983_dev *dev)
{
	if (dev == NULL)
		return -1;
	if (dev->magic != PIOS_HMC5983_DEV_MAGIC)
		return -2;
	if (dev->spi_id == 0)
		return -3;
	return 0;
}

/**
 * @brief Initialize the HMC5983 magnetometer sensor.
 * @return none
 */
int32_t PIOS_HMC5983_Init(uint32_t spi_id, uint32_t slave_num, const struct pios_hmc5983_cfg *cfg)
{
	dev = (struct hmc5983_dev *)PIOS_HMC5983_alloc();
	if (dev == NULL)
		return -1;

	dev->cfg = cfg;
	dev->spi_id = spi_id;
	dev->slave_num = slave_num;

#ifdef PIOS_HMC5983_HAS_GPIOS
	PIOS_EXTI_Init(cfg->exti_cfg);
#endif

	if (PIOS_HMC5983_Config(cfg) != 0)
		return -2;

	PIOS_SENSORS_Register(PIOS_SENSOR_MAG, dev->queue);

	int result = xTaskCreate(PIOS_HMC5983_Task, (const signed char *)"pios_hmc5983",
				HMC5983_TASK_STACK, NULL, HMC5983_TASK_PRIORITY,
				&dev->task);

	PIOS_Assert(result == pdPASS);

	dev->data_ready_sema = xSemaphoreCreateMutex();

	return 0;
}

/**
 * @brief Initialize the HMC5983 magnetometer sensor
 * \return none
 * \param[in] PIOS_HMC5983_ConfigTypeDef struct to be used to configure sensor.
 *
 * CTRL_REGA: Control Register A
 * Read Write
 * Default value: 0x10
 * 7 TS: Enable temperature sensor. It will be measured at each magnetic measurement,
 *       enable for automatic compensation of sensitivity over temperature.
 * 6:5 MA1-MA0: Number of averaged samples
 *             MA1 | MA0 | Number of samples averaged
 *            ------------------------------------------
 *              0  |  0  | 1 (default)
 *              0  |  1  | 2
 *              1  |  0  | 4
 *              1  |  1  | 8
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
 *              1  |  1   |  1   |           220
 * 1:0 MS1-MS0: Measurement Configuration Bits
 *             MS1 | MS0 |   MODE
 *            ------------------------------
 *              0  |  0   |  Normal
 *              0  |  1   |  Positive Bias
 *              1  |  0   |  Negative Bias
 *              1  |  1   |  Temperature sensor only. Magnetic sensor will not be
 *                           enabled during measurement.
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
 * 7   HS: Set this pin to enable I2C High Speed mode, 3400kHz
 * 6:      Clear this bit for correct operation
 * 5   LP: Lowest  power mode. When set, ODR = 0.75Hz and averaging = 1
 * 4-3: Clear this bit for correct functionality
 * 2  SIM: SPI serial interface mode selection, 0: 4-wire spi, 1: 3-wire SPI
 * 1:0 MD1-MD0: Mode Select Bits
 *             MD1 | MD0 |   MODE
 *            ------------------------------
 *              0  |  0   |  Continuous-Conversion Mode.
 *              0  |  1   |  Single-Conversion Mode
 *              1  |  0   |  Idle Mode
 *              1  |  1   |  Idle Mode
 */


static uint8_t CTRLB = 0x00;
static int32_t PIOS_HMC5983_Config(const struct pios_hmc5983_cfg *cfg)
{
	uint8_t CTRLA = 0x00;
	uint8_t MODE = 0x00;
	CTRLB = 0;

	CTRLA |= (uint8_t)(PIOS_HMC5983_ENABLE_TEMP_SENSOR | cfg->Averaging | cfg->M_ODR | cfg->Meas_Conf);
	CTRLB |= (uint8_t)(cfg->Gain);
	MODE |= (uint8_t)(cfg->Mode);

	// CTRL_REGA
	if (PIOS_HMC5983_Write(PIOS_HMC5983_CONFIG_REG_A, CTRLA) != 0)
		return -1;

	// CTRL_REGB
	if (PIOS_HMC5983_Write(PIOS_HMC5983_CONFIG_REG_B, CTRLB) != 0)
		return -1;

	// Mode register
	if (PIOS_HMC5983_Write(PIOS_HMC5983_MODE_REG, MODE) != 0)
		return -1;

	return 0;
}

/**
 * @brief Read current X, Z, Y values (in that order)
 * \param[out] int16_t array of size 3 to store X, Z, and Y magnetometer readings
 * \return 0 for success or -1 for failure
 */
static int32_t PIOS_HMC5983_ReadMag(struct pios_sensor_mag_data *mag_data)
{
	if (PIOS_HMC5983_Validate(dev) != 0)
		return -1;

	uint8_t buffer[6];
	int32_t sensitivity;

	if (PIOS_HMC5983_Read(PIOS_HMC5983_DATAOUT_XMSB_REG, buffer, 6) != 0)
		return -1;

	switch (CTRLB & 0xE0) {
	case 0x00:
		sensitivity = PIOS_HMC5983_Sensitivity_0_88Ga;
		break;
	case 0x20:
		sensitivity = PIOS_HMC5983_Sensitivity_1_3Ga;
		break;
	case 0x40:
		sensitivity = PIOS_HMC5983_Sensitivity_1_9Ga;
		break;
	case 0x60:
		sensitivity = PIOS_HMC5983_Sensitivity_2_5Ga;
		break;
	case 0x80:
		sensitivity = PIOS_HMC5983_Sensitivity_4_0Ga;
		break;
	case 0xA0:
		sensitivity = PIOS_HMC5983_Sensitivity_4_7Ga;
		break;
	case 0xC0:
		sensitivity = PIOS_HMC5983_Sensitivity_5_6Ga;
		break;
	case 0xE0:
		sensitivity = PIOS_HMC5983_Sensitivity_8_1Ga;
		break;
	default:
		PIOS_Assert(0);
	}

	int16_t mag_x, mag_y, mag_z;
	mag_x = ((int16_t)((uint16_t) buffer[0] << 8) + buffer[1]) * 1000 / sensitivity;
	mag_z = ((int16_t)((uint16_t) buffer[2] << 8) + buffer[3]) * 1000 / sensitivity;
	mag_y = ((int16_t)((uint16_t) buffer[4] << 8) + buffer[5]) * 1000 / sensitivity;

	// Define "0" when the fiducial is in the front left of the board
	switch (dev->cfg->orientation) {
	case PIOS_HMC5983_TOP_0DEG:
		mag_data->x = -mag_x;
		mag_data->y = mag_y;
		mag_data->z = -mag_z;
		break;
	case PIOS_HMC5983_TOP_90DEG:
		mag_data->x = -mag_y;
		mag_data->y = -mag_x;
		mag_data->z = -mag_z;
		break;
	case PIOS_HMC5983_TOP_180DEG:
		mag_data->x = mag_x;
		mag_data->y = -mag_y;
		mag_data->z = -mag_z;
		break;
	case PIOS_HMC5983_TOP_270DEG:
		mag_data->x = mag_y;
		mag_data->y = mag_x;
		mag_data->z = -mag_z;
		break;
	}

#if PIOS_HMC5983_READ_TEMPERATURE
	uint8_t temperature[2];
	float mag_temperature;
	if (PIOS_HMC5983_Read(PIOS_HMC5983_DATAOUT_TEMPMSB_REG, temperature, 2) == 0)
		mag_temperature = (((uint16_t)temperature[0] << 8) + temperature[1]) / 128 + 25;
#endif

	// This should not be necessary but for some reason it is coming out of continuous conversion mode
	PIOS_HMC5983_Write(PIOS_HMC5983_MODE_REG, PIOS_HMC5983_MODE_CONTINUOUS);

	return 0;
}

/**
 * @brief Read the identification bytes from the HMC5983 sensor
 * \param[out] uint8_t array of size 4 to store HMC5983 ID.
 * \return 0 if successful, -1 if not
 */
static uint8_t PIOS_HMC5983_ReadID(uint8_t out[4])
{
	uint8_t retval = PIOS_HMC5983_Read(PIOS_HMC5983_DATAOUT_IDA_REG, out, 3);
	out[3] = '\0';
	return retval;
}

/**
 * @brief Claim the SPI bus for compass communications and select this chip
 * @return 0 if successful, -1 for invalid device, -2 if unable to claim bus
 */
static int32_t PIOS_HMC5983_ClaimBus(void)
{
	if (PIOS_HMC5983_Validate(dev) != 0)
		return -1;

	if (PIOS_SPI_ClaimBus(dev->spi_id) != 0)
		return -2;

	PIOS_SPI_RC_PinSet(dev->spi_id, dev->slave_num, 0);
	return 0;
}

/**
 * @brief Release the SPI bus for the compass communications and end the transaction
 * @return 0 if successful
 */
static int32_t PIOS_HMC5983_ReleaseBus(void)
{
	if (PIOS_HMC5983_Validate(dev) != 0)
		return -1;

	PIOS_SPI_RC_PinSet(dev->spi_id, dev->slave_num, 1);

	return PIOS_SPI_ReleaseBus(dev->spi_id);
}

/**
 * @brief Reads one or more bytes into a buffer
 * \param[in] address HMC5983 register address (depends on size)
 * \param[out] buffer destination buffer
 * \param[in] len number of bytes which should be read
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_HMC5983_Read(uint8_t address, uint8_t *buffer, uint8_t len)
{
	if (PIOS_HMC5983_ClaimBus() != 0)
		return -1;

	PIOS_SPI_TransferByte(dev->spi_id, PIOS_HMC5983_READ_MODE | address);
	if (PIOS_SPI_TransferBlock(dev->spi_id, NULL, buffer, len, NULL) < 0) {
		PIOS_HMC5983_ReleaseBus();
		return -2;
	}

	return PIOS_HMC5983_ReleaseBus();
}

/**
 * @brief Writes one or more bytes to the HMC5983
 * \param[in] address Register address
 * \param[in] buffer source buffer
 * \return 0 if operation was successful
 * \return -1 if error during I2C transfer
 * \return -2 if unable to claim i2c device
 */
static int32_t PIOS_HMC5983_Write(uint8_t address, uint8_t buffer)
{
	if (PIOS_HMC5983_ClaimBus() != 0)
		return -1;

	PIOS_SPI_TransferByte(dev->spi_id, 0x3F & address);
	PIOS_SPI_TransferByte(dev->spi_id, buffer);

	return PIOS_HMC5983_ReleaseBus();
}

/**
 * @brief Run self-test operation.  Do not call this during operational use!!
 * \return 0 if success, -1 if test failed
 */
int32_t PIOS_HMC5983_Test(void)
{
	int32_t failed = 0;
	uint8_t registers[3] = { 0, 0, 0 };
	uint8_t status;
	uint8_t ctrl_a_read;
	uint8_t ctrl_b_read;
	uint8_t mode_read;
	struct pios_sensor_mag_data values;

	/* Verify that ID matches (HMC5983 ID is null-terminated ASCII string "H43") */
	char id[4];
	PIOS_HMC5983_ReadID((uint8_t *)id);
	if ((id[0] != 'H') || (id[1] != '4') || (id[2] != '3')) // Expect H43
		return -1;

	/* Backup existing configuration */
	if (PIOS_HMC5983_Read(PIOS_HMC5983_CONFIG_REG_A, registers, 3) != 0)
		return -1;

	/* Stop the device and read out last value */
	PIOS_DELAY_WaitmS(10);
	if (PIOS_HMC5983_Write(PIOS_HMC5983_MODE_REG, PIOS_HMC5983_MODE_IDLE) != 0)
		return -1;
	if (PIOS_HMC5983_Read(PIOS_HMC5983_DATAOUT_STATUS_REG, &status, 1) != 0)
		return -1;
	if (PIOS_HMC5983_ReadMag(&values) != 0)
		return -1;

	/*
	 * Put HMC5983 into self test mode
	 * This is done by placing measurement config into positive (0x01) or negative (0x10) bias
	 * and then placing the mode register into single-measurement mode.  This causes the HMC5983
	 * to create an artificial magnetic field of ~1.1 Gauss.
	 *
	 * If gain were PIOS_HMC5983_GAIN_2_5, for example, X and Y will read around +766 LSB
	 * (1.16 Ga * 660 LSB/Ga) and Z would read around +713 LSB (1.08 Ga * 660 LSB/Ga)
	 *
	 * Changing measurement config back to PIOS_HMC5983_MEASCONF_NORMAL will leave self-test mode.
	 */
	PIOS_DELAY_WaitmS(10);
	if (PIOS_HMC5983_Write(PIOS_HMC5983_CONFIG_REG_A, PIOS_HMC5983_MEASCONF_BIAS_POS | PIOS_HMC5983_ODR_15) != 0)
		return -1;
	PIOS_DELAY_WaitmS(10);
	if (PIOS_HMC5983_Write(PIOS_HMC5983_CONFIG_REG_B, PIOS_HMC5983_GAIN_8_1) != 0)
		return -1;
	PIOS_DELAY_WaitmS(10);
	if (PIOS_HMC5983_Write(PIOS_HMC5983_MODE_REG, PIOS_HMC5983_MODE_SINGLE) != 0)
		return -1;

	/* Must wait for value to be updated */
	PIOS_DELAY_WaitmS(200);

	if (PIOS_HMC5983_ReadMag(&values) != 0)
		return -1;

	PIOS_HMC5983_Read(PIOS_HMC5983_CONFIG_REG_A, &ctrl_a_read, 1);
	PIOS_HMC5983_Read(PIOS_HMC5983_CONFIG_REG_B, &ctrl_b_read, 1);
	PIOS_HMC5983_Read(PIOS_HMC5983_MODE_REG, &mode_read, 1);
	PIOS_HMC5983_Read(PIOS_HMC5983_DATAOUT_STATUS_REG, &status, 1);

	/* Restore backup configuration */
	PIOS_DELAY_WaitmS(10);
	if (PIOS_HMC5983_Write(PIOS_HMC5983_CONFIG_REG_A, registers[0]) != 0)
		return -1;
	PIOS_DELAY_WaitmS(10);
	if (PIOS_HMC5983_Write(PIOS_HMC5983_CONFIG_REG_B, registers[1]) != 0)
		return -1;
	PIOS_DELAY_WaitmS(10);
	if (PIOS_HMC5983_Write(PIOS_HMC5983_MODE_REG, registers[2]) != 0)
		return -1;

	return failed;
}

/**
 * @brief IRQ Handler
 */
bool PIOS_HMC5983_IRQHandler(void)
{
	if (PIOS_HMC5983_Validate(dev) != 0)
		return false;

	portBASE_TYPE xHigherPriorityTaskWoken;
	xSemaphoreGiveFromISR(dev->data_ready_sema, &xHigherPriorityTaskWoken);

	return xHigherPriorityTaskWoken == pdTRUE;
}

/**
 * The HMC5983 task
 */
static void PIOS_HMC5983_Task(void *parameters)
{
	while (1) {
		if (PIOS_HMC5983_Validate(dev) != 0) {
			vTaskDelay(100 * portTICK_RATE_MS);
			continue;
		}

		if (xSemaphoreTake(dev->data_ready_sema, portMAX_DELAY) != pdTRUE) {
			vTaskDelay(100 * portTICK_RATE_MS);
			continue;
		}

		struct pios_sensor_mag_data mag_data;
		if (PIOS_HMC5983_ReadMag(&mag_data) == 0)
			xQueueSend(dev->queue, (void *)&mag_data, 0);
	}
}

#endif /* PIOS_INCLUDE_HMC5983 */

/**
 * @}
 * @}
 */
