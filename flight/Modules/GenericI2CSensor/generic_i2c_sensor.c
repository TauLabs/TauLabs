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
#include "i2c_vm_asm.h"		/* Minimal assembler for I2C VM */

extern bool i2c_vm_run (const uint32_t * code, uint8_t code_len, uintptr_t i2c_adapter);

// Private constants
#define STACK_SIZE_BYTES 370
#define TASK_PRIORITY (tskIDLE_PRIORITY+1)

// Private variables
static xTaskHandle taskHandle;
static bool module_enabled = false;

#include "pios_hmc5883.h"	/* PIOS_HMC5883_* */
static const uint32_t op_mag_baro_program[] = {
	I2C_VM_ASM_DELAY(255),

	/* Read HMC5883L Magnetometer ID */
	/* Note: The datasheet claims address 0x1E but my device responds on 0x1C */
	I2C_VM_ASM_SET_DEV_ADDR(0x1C),   /* Set I2C device address (in 7-bit) */

	I2C_VM_ASM_STORE(PIOS_HMC5883_DATAOUT_IDA_REG, 0),
	I2C_VM_ASM_WRITE_I2C(0, 1),
	I2C_VM_ASM_DELAY(6),
	I2C_VM_ASM_READ_I2C(0,3),

	/* Configure HMC5883L Magnetometer */
	I2C_VM_ASM_STORE(PIOS_HMC5883_CONFIG_REG_A, 0),
	I2C_VM_ASM_STORE((0x3 << 5) | PIOS_HMC5883_ODR_15 | PIOS_HMC5883_MEASCONF_NORMAL, 1), /* CONFIG_A val */
	I2C_VM_ASM_STORE(PIOS_HMC5883_GAIN_1_9, 2), /* CONFIG_B val */
	I2C_VM_ASM_STORE(PIOS_HMC5883_MODE_CONTINUOUS, 3), /* MODE val */
	I2C_VM_ASM_WRITE_I2C(0, 4),
	I2C_VM_ASM_DELAY(6),

	/* Read the Magnetometer */
	I2C_VM_ASM_SET_DEV_ADDR(0x1C),   /* Set I2C device address (in 7-bit) */

	I2C_VM_ASM_STORE(PIOS_HMC5883_DATAOUT_XMSB_REG, 0),
	I2C_VM_ASM_WRITE_I2C(0, 1),
	I2C_VM_ASM_READ_I2C(0,7),

	I2C_VM_ASM_LOAD_BE(0, 2, VM_R0), /* mag_x */
	I2C_VM_ASM_LOAD_BE(2, 2, VM_R1), /* mag_y */
	I2C_VM_ASM_LOAD_BE(4, 2, VM_R2), /* mag_z */

	/* Temperature conversion */

	I2C_VM_ASM_SET_DEV_ADDR(0x77),   /* Set I2C device address (in 7-bit) */

	I2C_VM_ASM_STORE(0xF4, 0),        /* Store ctrl register address */
	I2C_VM_ASM_STORE(0x2E, 1),        /* Store temp conv address */
	I2C_VM_ASM_WRITE_I2C(0, 2),       /* Write two bytes */
	I2C_VM_ASM_DELAY(5),	          /* Wait for temperature conversion to complete */

	I2C_VM_ASM_STORE(0xF6, 0),        /* Store ADC MSB address */
	I2C_VM_ASM_WRITE_I2C(0, 1),       /* Write one byte */
	I2C_VM_ASM_READ_I2C(0, 2),        /* Read 2 byte ADC value */
	I2C_VM_ASM_LOAD_BE(0, 2, VM_R3),  /* Load 16-bit formatted bytes into first output reg */

	/* Pressure conversion */

	I2C_VM_ASM_STORE(0xF4, 0),        /* Store ctrl register address */
	I2C_VM_ASM_STORE(0x34 + (0x3 << 6), 1),	/* Store pressure conv address */
	I2C_VM_ASM_WRITE_I2C(0, 2),       /* Write two bytes */
	I2C_VM_ASM_DELAY(26),	          /* Wait for pressure conversion to complete */

	I2C_VM_ASM_STORE(0xF6, 0),        /* Store ADC MSB address */
	I2C_VM_ASM_WRITE_I2C(0, 1),       /* Write one byte */
	I2C_VM_ASM_READ_I2C(0, 3),        /* Read 3 byte ADC value */
	I2C_VM_ASM_LOAD_BE(0, 3, VM_R4),  /* Load 24-bit formatted bytes into first output reg */

	I2C_VM_ASM_SEND_UAVO(),	          /* Set the UAVObject */
	I2C_VM_ASM_JUMP(-25),             /* Jump back 25 instructions */
};

static const uint32_t basictest_program[] = {
	I2C_VM_ASM_SET_CTR(10),

	I2C_VM_ASM_STORE(0x0A, 0),
	I2C_VM_ASM_STORE(0x0B, 1),
	I2C_VM_ASM_STORE(0x0C, 2),
	I2C_VM_ASM_STORE(0x0D, 3),

	/* Test Little-endian conversion routines */
	I2C_VM_ASM_LOAD_LE(0, 2, VM_R0),
	I2C_VM_ASM_LOAD_LE(0, 3, VM_R1),
	I2C_VM_ASM_LOAD_LE(0, 4, VM_R2),

	/* Test Big-endian conversion routines */
	I2C_VM_ASM_LOAD_BE(0, 2, VM_R3),
	I2C_VM_ASM_LOAD_BE(0, 3, VM_R4),
	I2C_VM_ASM_LOAD_BE(0, 4, VM_R5),

	I2C_VM_ASM_SEND_UAVO(),	/* Set the UAVObject */

	I2C_VM_ASM_DEC_CTR(),
	I2C_VM_ASM_DELAY(20),
	I2C_VM_ASM_BNZ(-9),
};

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

	/* Make sure we have something to run */
	if ((i2cvm_program == NULL) || (i2cvm_program_len == 0))
		return -1;

	I2CVMInitialize();

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
	ModuleSettingsInitialize();

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
		i2cvm_program = op_mag_baro_program;
		i2cvm_program_len = NELEMENTS(op_mag_baro_program);
		break;
	case MODULESETTINGS_I2CVMPROGRAMSELECT_BASICTEST:
		i2cvm_program = basictest_program;
		i2cvm_program_len = NELEMENTS(basictest_program);
		break;
	case MODULESETTINGS_I2CVMPROGRAMSELECT_NONE:
	default:
		/* No program selected, module will not start */
		break;
	}

	return 0;
}

MODULE_INITCALL(GenericI2CSensorInitialize, GenericI2CSensorStart)


static void GenericI2CSensorTask(void *parameters)
{
	// Main task loop
	while (1) {
		/* Run the selected program */
		if (i2c_vm_run(i2cvm_program, i2cvm_program_len, PIOS_I2C_MAIN_ADAPTER)) {
			/* Program ran to completion
			 * Delay to prevent empty/short programs from consuming all CPU
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
