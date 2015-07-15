/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_PX4FLOW PX4Flow Functions
 * @brief Deals with the hardware interface to the PixHawk optical flow sensor
 * @{
 * @file       pios_px4flow.h
 * @author     Kenn Sebesta, Copyright (C) 2014
 * @brief      PX4Flow optical flow sensor
 *
 ******************************************************************************
 */

#ifndef PIOS_PX4FLOW_H
#define PIOS_PX4FLOW_H

#include <stdint.h>
#include <stdbool.h>

/* PX4FLOW Addresses */
#define PIOS_PX4FLOW_I2C_7_BIT_ADDR                              0x42
#define PIOS_PX4FLOW_FRAMECOUNTER_LSB                            0x00
#define PIOS_PX4FLOW_FRAMECOUNTER_SINCE_LAST_I2C_READING_LSB     0x16

struct Rotation {
	// Units are [deg*100]
	int16_t roll_D100;
	int16_t pitch_D100;
	int16_t yaw_D100;
};

struct pios_px4flow_cfg {
	struct Rotation rotation;
};

struct pios_px4flow_data {
	int16_t x_dot;
	int16_t y_dot;
	int16_t x;
	int16_t y;
};

/* Public Functions */
extern int32_t PIOS_PX4Flow_Init(const struct pios_px4flow_cfg *cfg, const uint32_t i2c_id);
extern int32_t PIOS_PX4Flow_Test(void);
extern int32_t PIOS_PX4Flow_SetRotation(const struct Rotation rotation);
extern bool PIOS_PX4Flow_IRQHandler();
#endif /* PIOS_PX4FLOW_H */

/** 
  * @}
  * @}
  */
