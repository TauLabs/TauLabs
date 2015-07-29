/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_PX4FLOW PX4Flow Functions
 * @brief Deals with the hardware interface to the PixHawk optical flow sensor
 * @{
 * @file       pios_px4flow.c
 * @author     Kenn Sebesta, Copyright (C) 2014
 * @brief      PX4Flow optical flow sensor
 * @see        The GNU Public License (GPL) Version 3
 *
 ******************************************************************************
 */

/* Project Includes */
#include "pios.h"
#include "coordinate_conversions.h"
#include "physical_constants.h"

#if defined(PIOS_INCLUDE_PX4FLOW)
#include "pios_px4flow_priv.h"

#include "pios_semaphore.h"
#include "pios_thread.h"
#include "pios_queue.h"

/* Private constants */
#define PX4FLOW_TASK_PRIORITY        PIOS_THREAD_PRIO_HIGHEST
#define PX4FLOW_TASK_STACK_BYTES     512
#define PX4FLOW_SAMPLE_PERIOD_MS     5
#define PIOS_PX4FLOW_MAX_DOWNSAMPLE  1

/* Global Variables */

/* Local Types */
enum pios_px4flow_dev_magic {
	PIOS_PX4FLOW_DEV_MAGIC = 0x1dbef871, // md5 hash of the string "PIOS_PX4FLOW_DEV_MAGIC"
};

struct px4flow_dev {
	uint32_t i2c_id;
	const struct pios_px4flow_cfg *cfg;
	struct pios_queue *optical_flow_queue;
	struct pios_queue *sonar_queue;
	struct pios_thread *task;
	struct pios_semaphore *data_ready_sema;
	enum pios_px4flow_dev_magic magic;
	float Rsb[3][3];
};

/* Local Variables */
static int32_t PIOS_PX4Flow_Config(const struct pios_px4flow_cfg * cfg);
static void PIOS_PX4Flow_Task(void *parameters);

static struct px4flow_dev *dev;

/**
 * @brief Allocate a new device
 */
static struct px4flow_dev * PIOS_PX4Flow_alloc(void)
{
	struct px4flow_dev *px4flow_dev;
	
	px4flow_dev = (struct px4flow_dev *)PIOS_malloc(sizeof(*px4flow_dev));
	if (!px4flow_dev) return (NULL);
	
	px4flow_dev->magic = PIOS_PX4FLOW_DEV_MAGIC;
	
	px4flow_dev->optical_flow_queue = PIOS_Queue_Create(PIOS_PX4FLOW_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_optical_flow_data));
	px4flow_dev->sonar_queue = PIOS_Queue_Create(PIOS_PX4FLOW_MAX_DOWNSAMPLE, sizeof(struct pios_sensor_sonar_data));
	if (px4flow_dev->optical_flow_queue == NULL || px4flow_dev->sonar_queue == NULL) {
		PIOS_free(px4flow_dev);
		return NULL;
	}

	return px4flow_dev;
}

/**
 * @brief Validate the handle to the i2c device
 * @returns 0 for valid device or -1 otherwise
 */
static int32_t PIOS_PX4Flow_Validate(struct px4flow_dev *dev)
{
	if (dev == NULL)
		return -1;
	if (dev->magic != PIOS_PX4FLOW_DEV_MAGIC)
		return -2;
	if (dev->i2c_id == 0)
		return -3;
	return 0;
}

/**
 * @brief Initialize the PX4Flow optical flow sensor.
 * @return 0 on success
 */
int32_t PIOS_PX4Flow_Init(const struct pios_px4flow_cfg *cfg, const uint32_t i2c_id)
{
	dev = (struct px4flow_dev *) PIOS_PX4Flow_alloc();
	if (dev == NULL)
		return -1;

	dev->cfg = cfg;
	dev->i2c_id = i2c_id;
	PIOS_PX4Flow_SetRotation(cfg->rotation);

	if (PIOS_PX4Flow_Config(cfg) != 0)
		return -2;

	PIOS_SENSORS_Register(PIOS_SENSOR_OPTICAL_FLOW, dev->optical_flow_queue);
	PIOS_SENSORS_Register(PIOS_SENSOR_SONAR, dev->sonar_queue);

	dev->task = PIOS_Thread_Create(PIOS_PX4Flow_Task, "pios_px4flow", PX4FLOW_TASK_STACK_BYTES, NULL, PX4FLOW_TASK_PRIORITY);

	PIOS_Assert(dev->task != NULL);

	return 0;
}

/**
 * @brief Updates the PX4Flow's orientation.
 * @returns 0 for success or -1 for failure
 */
int32_t PIOS_PX4Flow_SetRotation(const struct Rotation rotation)
{
	if (PIOS_PX4Flow_Validate(dev) != 0)
		return -1;

	float rpy_R[3]  = {rotation.roll_D100/100.0f * DEG2RAD, rotation.pitch_D100/100.0f * DEG2RAD, rotation.yaw_D100/100.0f * DEG2RAD};

	Euler2R(rpy_R, dev->Rsb);


	return 0;
}

/**
 * @brief Initialize the PX4Flow optical flow sensor
 * \return none
 * \param[in] PIOS_PX4Flow_ConfigTypeDef struct to be used to configure sensor.
 *
 */
static int32_t PIOS_PX4Flow_Config(const struct pios_px4flow_cfg * cfg)
{
	// This function intentionally left blank.

	return 0;
}

/**
 * @brief Read current X, Z, Y values (in that order)
 * \param[out] int16_t array of size 3 to store X, Z, and Y magnetometer readings
 * \return 0 for success or -1 for failure
 */
static int32_t PIOS_PX4Flow_ReadData(struct pios_sensor_optical_flow_data *optical_flow_data, struct pios_sensor_sonar_data *sonar_data)
{
	if (PIOS_PX4Flow_Validate(dev) != 0) {
		return -1;
	}

	/* don't use PIOS_PX4Flow_Read and PIOS_PX4Flow_Write here because the task could be
	 * switched out of context in between which would give the sensor less time to capture
	 * the next sample.
	 */
	uint8_t addr_read[] = {
		PIOS_PX4FLOW_FRAMECOUNTER_LSB
	};

	/* Optical flow structs */
	struct I2C_Frame
	{
		uint16_t frame_count; // counts created I2C frames [#frames]
		int16_t pixel_flow_x_sum_px10; // latest x flow measurement in pixels*10 [pixels]
		int16_t pixel_flow_y_sum_px10; // latest y flow measurement in pixels*10 [pixels]
		int16_t flow_comp_x_m1000; // x velocity*1000 [meters/sec]
		int16_t flow_comp_y_m1000; // y velocity*1000 [meters/sec]
		int16_t qual; // Optical flow solution quality [0: bad, 255: maximum quality]
		int16_t gyro_x_R; // latest gyro x rate [rad/sec]
		int16_t gyro_y_R; // latest gyro y rate [rad/sec]
		int16_t gyro_z_R; // latest gyro z rate [rad/sec]
		uint8_t gyro_range; // gyro range [0 .. 7] equals [50 deg/sec .. 2000 deg/sec]
		uint8_t sonar_timestamp; // time since last sonar update [milliseconds]
		int16_t ground_distance_m1000; // Ground distance in meters*1000 [meters]. Positive value: distance known. Negative value: Unknown distance
	} __attribute__((packed)) i2c_frame;

/*
	// Leaving this struct definition in as aid for future implementation
	struct I2C_Integral_Frame
	{
		 uint16_t frame_count_since_last_readout; //number of flow measurements since last I2C readout [#frames]
		 int16_t pixel_flow_x_integral; //accumulated flow in radians*10000 around x axis since last I2C readout [rad*10000]
		 int16_t pixel_flow_y_integral; //accumulated flow in radians*10000 around y axis since last I2C readout [rad*10000]
		 int16_t gyro_x_rate_integral; //accumulated gyro x rates in radians*10000 since last I2C readout [rad*10000]
		 int16_t gyro_y_rate_integral; //accumulated gyro y rates in radians*10000 since last I2C readout [rad*10000]
		 int16_t gyro_z_rate_integral; //accumulated gyro z rates in radians*10000 since last I2C readout [rad*10000]
		 uint32_t integration_timespan; //accumulation timespan in microseconds since last I2C readout [microseconds]
		 uint32_t sonar_timestamp; // time since last sonar update [microseconds]
		 int16_t ground_distance_m1000; // Ground distance in meters*1000 [meters*1000]
		 int16_t gyro_temperature; // Temperature * 100 in centi-degrees Celsius [degcelsius*100]
		 uint8_t quality; // averaged quality of accumulated flow values [0:bad quality;255: max quality]
	} __attribute__((packed)) i2c_integral_frame;
*/
	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = PIOS_PX4FLOW_I2C_7_BIT_ADDR,
			.rw = PIOS_I2C_TXN_WRITE,
			.len = 1,
			.buf = addr_read,
		},
		{
			.info = __func__,
			.addr = PIOS_PX4FLOW_I2C_7_BIT_ADDR,
			.rw = PIOS_I2C_TXN_READ,
			.len = sizeof(i2c_frame),
			.buf = (uint8_t *)&i2c_frame,
		}
	};

	// Perform transfer, and return error if TX fails
	if (PIOS_I2C_Transfer(dev->i2c_id, txn_list, NELEMENTS(txn_list)) != 0)
		return -1;

	// Only update sonar queue if data is fresh
	if (i2c_frame.sonar_timestamp > 0) {
		sonar_data->x = 0;
		sonar_data->y = 0;
		sonar_data->z = i2c_frame.ground_distance_m1000 / 1000.0f;

		PIOS_Queue_Send(dev->sonar_queue, sonar_data, 0);
	}

	/* Rotate the flow from the sensor frame into the body frame. It's not
	  * good to set the z-axis velocity to 0, but the optical flow doesn't return
	  * any data along that direction, and moreover if the sensor is always
	  * mounted vertically then the sensor board mounting yaw angle will not
	  * affect the vertical results.
	  */
	float flow_sensor_frame[3] = {i2c_frame.flow_comp_x_m1000 / 1000.0f, i2c_frame.flow_comp_y_m1000 / 1000.0f, 0 / 1000.0f};
	float flow_rotated[3];
	rot_mult(dev->Rsb, flow_sensor_frame, flow_rotated, true);
	optical_flow_data->x_dot = flow_rotated[0];
	optical_flow_data->y_dot = flow_rotated[1];
	optical_flow_data->z_dot = flow_rotated[2];

	optical_flow_data->quality = i2c_frame.qual;

	PIOS_Queue_Send(dev->optical_flow_queue, optical_flow_data, 0);

	return 0;
}


/**
 * The PX4FLOW task
 */
static void PIOS_PX4Flow_Task(void *parameters)
{
	while (PIOS_PX4Flow_Validate(dev) != 0) {
		PIOS_Thread_Sleep(100);
	}

	uint32_t now = PIOS_Thread_Systime();

	while (1) {
		PIOS_Thread_Sleep_Until(&now, PX4FLOW_SAMPLE_PERIOD_MS);

		struct pios_sensor_optical_flow_data optical_flow_data;
		struct pios_sensor_sonar_data sonar_data;
		PIOS_PX4Flow_ReadData(&optical_flow_data, &sonar_data);
	}
}

#endif /* PIOS_INCLUDE_PX4FLOW */

/**
 * @}
 * @}
 */
