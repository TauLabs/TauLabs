/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup GenericI2CSensor Generic I2C sensor interface
 * @{
 *
 * @file       vmprog_endiantest.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      I2C Virtual machine demo program
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

#include "i2c_vm_asm.h"		/* I2C_VM_* assembler */
#include <stdint.h>		/* uint32_t */

const uint32_t vmprog_endiantest[] = {
	I2C_VM_ASM_SET_IMM (VM_R6, 10),

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

	I2C_VM_ASM_ADD_IMM(VM_R6, -1),
	I2C_VM_ASM_DELAY(20),
	I2C_VM_ASM_BNZ(VM_R6, -9),
};

#define NELEMENTS(x) (sizeof(x) / sizeof(*(x)))
const uint32_t vmprog_endiantest_len = NELEMENTS(vmprog_endiantest);

/**
 * @}
 * @}
 */
