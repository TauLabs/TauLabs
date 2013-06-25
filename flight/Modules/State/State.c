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

#include "accels.h"
#include "attitudeactual.h"
#include "gyros.h"
#include "gyrosbias.h"
#include "magnetometer.h"
#include "positionactual.h"
#include "statefilter.h"
#include "velocityactual.h"

// Private constants
#define STACK_SIZE_BYTES 2448
#define TASK_PRIORITY (tskIDLE_PRIORITY+3)

// Private variables
static xTaskHandle stateTaskHandle;
static struct filter_driver *current_filter = NULL;
static uintptr_t running_filter_id;

// Private functions
static void StateTask(void *parameters);

// Mapping from UAVO setting to filters.  This might want to be an extern
// loaded from board specific information to indicate which filters are
// supported.
extern struct filter_driver cf_filter_driver;
static struct filter_driver filters[1];

/**
 * Initialise the module.  Called before the start function
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t StateInitialize(void)
{
	StateFilterInitialize();

	// Get the driver for the selected filter
	uint8_t selected_filter;
	StateFilterAttitudeFilterGet(&selected_filter);
	if (selected_filter < NELEMENTS(filters))
		current_filter = &filters[selected_filter];
	else
		return -1;
	current_filter = &cf_filter_driver;

	// Check this filter is safe to run
	if (!filter_interface_validate(current_filter, running_filter_id))
		return -1;
	if (current_filter->init(&running_filter_id) != 0)
		return -1;

	// TODO: make sure system refuses to run without state module
	return 0;
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

	// Cannot trust the values to init right above if BL runs
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

	// Main task loop
	while (1) {
		float dt = 0.003f; // FIXME

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
