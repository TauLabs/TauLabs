/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_SENSORS Generic sensor interface functions
 * @brief Generic interface for sensors
 * @{
 *
 * @file       pios_sensors.h
 * @author     PhoenixPIlot, http://github.com/PhoenixPilot Copyright (C) 2012.
 * @brief      Generic interface for sensors
 * @see        The GNU Public License (GPL) Version 3
 *
 ******************************************************************************/
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

#ifndef PIOS_SENSOR_H
#define PIOS_SENSOR_H

#include "stdint.h"
#include "FreeRTOS.h"
#include "queue.h"

//! Pios sensor structure for generic gyro data
struct pios_sensor_gyro_data {
	float x;
	float y; 
	float z;
	float temperature;
};

//! Pios sensor structure for generic accel data
struct pios_sensor_accel_data {
	float x;
	float y; 
	float z;
	float temperature;
};

//! Pios sensor structure for generic mag data
struct pios_sensor_mag_data {
	float x;
	float y; 
	float z;
};

//! Pios sensor structure for generic baro data
struct pios_sensor_baro_data {
	float temperature;
	float pressure;
	float altitude;
};

//! The types of sensors this module supports
enum pios_sensor_type
{
	PIOS_SENSOR_ACCEL,
	PIOS_SENSOR_GYRO,
	PIOS_SENSOR_MAG,
	PIOS_SENSOR_BARO,
	PIOS_SENSOR_LAST
};

//! Structure to register the data
struct pios_sensor_registration {
	enum pios_sensor_type type;
	xQueueHandle queue;
};

//! Initialize the PIOS_SENSORS interface
int32_t PIOS_SENSORS_Init();

//! Register a sensor with the PIOS_SENSORS interface
int32_t PIOS_SENSORS_Register(enum pios_sensor_type type, xQueueHandle queue);

//! Get the data queue for a sensor type
xQueueHandle PIOS_SENSORS_GetQueue(enum pios_sensor_type type);

#endif /* PIOS_SENSOR_H */