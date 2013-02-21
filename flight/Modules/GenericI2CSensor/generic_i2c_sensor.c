/**
 ******************************************************************************
 * @file       generic_i2c_sensor.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @addtogroup I2C Sensor Module
 * @{
 * @addtogroup 
 * @{
 * @brief Runs the built-in or user defined program on the I2C Virtual Machine
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
#include "modulesettings.h"
#include "i2cvm.h"	   /* UAV Object (VM register file outputs) */
#include "i2cvmuserprogram.h"	/* UAV Object (bytecode to run) */

extern bool i2c_vm_run (const uint32_t * code, uint8_t code_len, uintptr_t i2c_adapter);

// Private constants
#define STACK_SIZE_BYTES 370
#define TASK_PRIORITY (tskIDLE_PRIORITY+1)

// Private variables
static xTaskHandle taskHandle;
static bool module_enabled = false;

// Private functions
static void GenericI2CSensorTask(void *parameters);

static const uint32_t * i2cvm_program = NULL; /* bytecode to run in the VM */
static uint16_t i2cvm_program_len = 0;	/* number of instructions in the program */

/**
* Start the module, called on startup
*/
static int32_t GenericI2CSensorStart(void)
{
	if (!module_enabled)
		return -1;

	// Start main task
	xTaskCreate(GenericI2CSensorTask, (signed char *)"GenericI2CSensor", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_GENERICI2CSENSOR, taskHandle);
	return 0;
}

/**
* Initialise the module, called on startup
*/
static int32_t GenericI2CSensorInitialize(void)
{

#ifdef MODULE_GenericI2CSensor_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_STATE_NUMELEM];
	ModuleSettingsStateGet(module_state);
	if (module_state[MODULESETTINGS_STATE_GENERICI2CSENSOR] == MODULESETTINGS_STATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (!module_enabled)
		return -1;

	/* Module is enabled, determine which program to run (if any) */
	uint8_t selected_program;
	ModuleSettingsI2CVMProgramSelectGet(&selected_program);

	switch (selected_program) {
	case MODULESETTINGS_I2CVMPROGRAMSELECT_USER:
		I2CVMUserProgramInitialize();
		uint32_t * user_program;
		user_program = pvPortMalloc(sizeof(((I2CVMUserProgramData *)0)->Program));
		if (!user_program) {
			/* Failed to allocate sufficient memory for the user program */
			return -1;
		}
		I2CVMUserProgramProgramGet(user_program);
		i2cvm_program = user_program;
		i2cvm_program_len = I2CVMUSERPROGRAM_PROGRAM_NUMELEM;
		break;
	case MODULESETTINGS_I2CVMPROGRAMSELECT_OPBAROALTIMETER:
		{
		extern const uint32_t vmprog_op_mag_baro[];
		extern const uint32_t vmprog_op_mag_baro_len;
		i2cvm_program = vmprog_op_mag_baro;
		i2cvm_program_len = vmprog_op_mag_baro_len;
		}
		break;
	case MODULESETTINGS_I2CVMPROGRAMSELECT_ENDIANTEST:
		{
		extern const uint32_t vmprog_endiantest[];
		extern const uint32_t vmprog_endiantest_len;
		i2cvm_program = vmprog_endiantest;
		i2cvm_program_len = vmprog_endiantest_len;
		}
		break;
	case MODULESETTINGS_I2CVMPROGRAMSELECT_MATHTEST:
		{
		extern const uint32_t vmprog_mathtest[];
		extern const uint32_t vmprog_mathtest_len;
		i2cvm_program = vmprog_mathtest;
		i2cvm_program_len = vmprog_mathtest_len;
		}
		break;
	case MODULESETTINGS_I2CVMPROGRAMSELECT_NONE:
	default:
		/* No program selected, module will not start */
		break;
	}

	/* Make sure we have something to run */
	if ((i2cvm_program == NULL) || (i2cvm_program_len == 0)) {
		module_enabled = false;
		return -1;
	}

	I2CVMInitialize();

	return 0;
}

MODULE_INITCALL(GenericI2CSensorInitialize, GenericI2CSensorStart)


static void GenericI2CSensorTask(void *parameters)
{
	// Main task loop
	while (1) {
		/* Run the selected program */
		if (i2c_vm_run(i2cvm_program, i2cvm_program_len, PIOS_I2C_MAIN_ADAPTER)) {
			/* Program ran to completion. This could be because the program is 
			 * empty or does not infinitely loop.
			 * Delay in order to prevent these programs from consuming all CPU.
			 */
			vTaskDelay(10 / portTICK_RATE_MS);
		} else {
			/* Program faulted
			 * Delay to prevent beoken programs from consuming all CPU
			 */
			vTaskDelay(100 / portTICK_RATE_MS);
		}
	}
}

/**
  * @}
 * @}
 */
