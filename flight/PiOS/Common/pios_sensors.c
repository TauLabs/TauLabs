/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_SENSORS Generic sensor interface functions
 * @brief Generic interface for sensors
 * @{
 *
 * @file       pios_sensors.c
 * @author     PhoenixPilot, http://github.com/PhoenixPilot Copyright (C) 2012.
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

// TODO: Make this pios driver actually create the queue and set that to the 
// lower driver (??)

#include "pios_sensors.h"

//! The list of queue handles
static xQueueHandle queues[PIOS_SENSOR_LAST];

//! Initialize the sensors interface
int32_t PIOS_SENSORS_Init()
{
	for (uint32_t i = 0; i < PIOS_SENSOR_LAST; i++)
		queues[i] = NULL;

	return 0;
}

//! Register a sensor with the PIOS_SENSORS interface
int32_t PIOS_SENSORS_Register(enum pios_sensor_type type, xQueueHandle queue)
{
	if(queues[type] != NULL)
		return -1;

	queues[type] = queue;

	return 0;
}

//! Get the data queue for a sensor type
xQueueHandle PIOS_SENSORS_GetQueue(enum pios_sensor_type type)
{
	if (type < 0 || type >= PIOS_SENSOR_LAST)
		return NULL;

	return queues[type];
}