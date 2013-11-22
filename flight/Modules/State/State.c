/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @file       state.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      State estimation module which calls to specific drivers
 *
 * @see        The GNU Public License (GPL) Version 3
 *
 *****************************************************************************/
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

#include "pios.h"
#include "openpilot.h"

#include "filter_interface.h"
#include "inspgs_interface.h"

// Include particular filters to get their handles
#include "cf_interface.h"
#include "cfnav_interface.h"

#include "attitudeactual.h"
#include "gyrosbias.h"
#include "statefilter.h"

// Private constants
#define STACK_SIZE_BYTES 2448
#define TASK_PRIORITY (tskIDLE_PRIORITY+3)

// Private variables
static xTaskHandle stateTaskHandle;
static struct filter_driver *current_filter;
static uintptr_t running_filter_id;

// Private functions
static void StateTask(void *parameters);

/**
 * Initialise the module.  Called before the start function
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t StateInitialize(void)
{
	StateFilterInitialize();

	// Get the driver for the selected filter
	uint8_t selected_filter;
	StateFilterFilterGet(&selected_filter);
	switch(selected_filter) {
	case STATEFILTER_FILTER_COMPLEMENTARY:
		current_filter = &cf_filter_driver;
		break;
	case STATEFILTER_FILTER_COMPNAV:
		current_filter = &cfnav_filter_driver;
		break;
	default:
		goto FAIL;
	}	

	// Check this filter is safe to run
	if (!filter_interface_validate(current_filter))
		goto FAIL;
	if (current_filter->init(&running_filter_id) != 0)
		goto FAIL;

	// TODO: make sure system refuses to run without state module
	return 0;

FAIL:
	AlarmsSet(SYSTEMALARMS_ALARM_ATTITUDE, SYSTEMALARMS_ALARM_CRITICAL);
	return -1;
}

/**
 * Start the task.  Expects all objects to be initialized by this point.
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t StateStart(void)
{
	// Initialize quaternion
	AttitudeActualData attitude;
	AttitudeActualGet(&attitude);
	attitude.q1 = 1;
	attitude.q2 = 0;
	attitude.q3 = 0;
	attitude.q4 = 0;
	AttitudeActualSet(&attitude);

	GyrosBiasData gyrosBias;
	GyrosBiasGet(&gyrosBias);
	gyrosBias.x = 0;
	gyrosBias.y = 0;
	gyrosBias.z = 0;
	GyrosBiasSet(&gyrosBias);

	// Start main task
	xTaskCreate(StateTask, (signed char *)"StateEstimation", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &stateTaskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_ATTITUDE, stateTaskHandle);
	PIOS_WDG_RegisterFlag(PIOS_WDG_ATTITUDE);

	return 0;
}

MODULE_INITCALL(StateInitialize, StateStart)

/**
 * Module thread, should not return.
 */
static void StateTask(void *parameters)
{
	AlarmsClear(SYSTEMALARMS_ALARM_ATTITUDE);

	// Wait for all the sensors be to read
	vTaskDelay(100);

	// Start the infrastructure for this filter
	if (current_filter->start(running_filter_id) != 0)
		goto FAIL;

	// Reset the filter to a known state
	if (current_filter->reset(running_filter_id) != 0)
		goto FAIL;

	int32_t last_raw_time = 0;

	// Main task loop
	while (1) {
		// Get time since last call in seconds
		float dt = PIOS_DELAY_DiffuS(last_raw_time) * 1e-6f;
		last_raw_time = PIOS_DELAY_GetRaw();

		current_filter->process(current_filter, running_filter_id, dt);
		PIOS_WDG_UpdateFlag(PIOS_WDG_ATTITUDE);
	}

FAIL:
	AlarmsSet(SYSTEMALARMS_ALARM_ATTITUDE, SYSTEMALARMS_ALARM_CRITICAL);
	while(1) {
		vTaskDelay(100);
		PIOS_WDG_UpdateFlag(PIOS_WDG_ATTITUDE);
	}
}


/**
 * @}
 * @}
 */
