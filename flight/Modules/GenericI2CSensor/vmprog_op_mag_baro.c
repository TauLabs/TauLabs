/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup GenericI2CSensor Generic I2C sensor interface
 * @{
 *
 * @file       vmprog_op_mag_baro.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      I2C Virtual machine demo program for the OpenPilot Mag+Baro board
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
#include "i2c_vm_asm.h"		/* I2C_VM_* assembler */
#include <stdint.h>		/* uint32_t */
#include "pios_hmc5883_priv.h"	/* PIOS_HMC5883_* */

const uint32_t vmprog_op_mag_baro[] = {
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

	/*
	 * Sign extend the 16-bit values by logical shifting left to place the sign bit
	 * at the top of the register and then arithmetic shift right to extend the sign
	 */
	I2C_VM_ASM_SL_IMM(VM_R0, 16),
	I2C_VM_ASM_ASR_IMM(VM_R0, 16),

	I2C_VM_ASM_SL_IMM(VM_R1, 16),
	I2C_VM_ASM_ASR_IMM(VM_R1, 16),

	I2C_VM_ASM_SL_IMM(VM_R2, 16),
	I2C_VM_ASM_ASR_IMM(VM_R2, 16),

	/* Scale Magnetometer Readings */
	I2C_VM_ASM_MUL_IMM(VM_R0, 1000),
	I2C_VM_ASM_DIV_IMM(VM_R0, PIOS_HMC5883_Sensitivity_1_9Ga),

	I2C_VM_ASM_MUL_IMM(VM_R1, 1000),
	I2C_VM_ASM_DIV_IMM(VM_R1, PIOS_HMC5883_Sensitivity_1_9Ga),

	I2C_VM_ASM_MUL_IMM(VM_R1, 1000),
	I2C_VM_ASM_DIV_IMM(VM_R1, PIOS_HMC5883_Sensitivity_1_9Ga),

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

	/* Scale the pressure conversion by the oversampling factor (set when conversion started) */
	I2C_VM_ASM_LSR_IMM(VM_R4, 8 - 3),

	I2C_VM_ASM_SEND_UAVO(),	          /* Set the UAVObject */
	I2C_VM_ASM_JUMP(-38),             /* Jump back 38 instructions */
};

const uint32_t vmprog_op_mag_baro_len = NELEMENTS(vmprog_op_mag_baro);

/**
 * @}
 * @}
 */
