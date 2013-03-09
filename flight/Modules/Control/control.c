/**
 ******************************************************************************
 * @addtogroup Modules Tau Labs Modules
 * @{
 * @addtogroup ControlModule Control Module
 * @brief Process the control sources and select the appropriate one.
 * @{
 *
 * @file       control.c
 * @author     Tau Labs, http://github.com/TauLabs Copyright (C) 2013.
 * @brief      ManualControl module. Handles safety R/C link and flight mode.
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

#include "openpilot.h"
#include "control.h"
#include "failsafe_control.h"
#include "tablet_control.h"
#include "transmitter_control.h"

// Private constants
#if defined(PIOS_CONTROL_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_MANUAL_STACK_SIZE
#else
#define STACK_SIZE_BYTES 1424
#endif

#define TASK_PRIORITY (tskIDLE_PRIORITY+4)
#define UPDATE_PERIOD_MS 20
#define THROTTLE_FAILSAFE -0.1f

// Private variables
static xTaskHandle taskHandle;
static portTickType lastSysTime;

// Private functions
static void controlTask(void *parameters);


/**
 * Module starting
 */
int32_t ControlStart()
{
	// Start main task
	xTaskCreate(controlTask, (signed char *)"Control", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_MANUALCONTROL, taskHandle);
	PIOS_WDG_RegisterFlag(PIOS_WDG_MANUAL);

	return 0;
}

/**
 * Module initialization
 */
int32_t ControlInitialize()
{
	failsafe_control_initialize();
	transmitter_control_initialize();
	tablet_control_initialize();


	return 0;
}

MODULE_INITCALL(ControlInitialize, ControlStart);

/**
 * Module task
 */
static void controlTask(void *parameters)
{
	// Main task loop
	lastSysTime = xTaskGetTickCount();
	while (1) {

		// Process periodic data for each of the controllers, including reading
		// all available inputs
		failsafe_control_update();
		transmitter_control_update();
		tablet_control_update();

		// Control logic to select the valid controller
		enum control_selection control_selection = transmitter_control_selected_controller();
		switch(control_selection) {
		case TRANMITTER_MISSING:
			failsafe_control_select();
			break;
		case TRANMITTER_PRESENT_AND_USED:
			transmitter_control_select();
			break;
		case TRANSMITTER_PRESENT_SELECT_TABLET:
			tablet_control_select();
			break;
		}

		// Wait until next update
		vTaskDelayUntil(&lastSysTime, UPDATE_PERIOD_MS / portTICK_RATE_MS);
		PIOS_WDG_UpdateFlag(PIOS_WDG_MANUAL);
	}
}

/**
  * @}
  * @}
  */
