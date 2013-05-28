/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup GenericI2CSensor Generic I2C sensor interface
 * @{
 *
 * @file       vmprog_mathtest.c
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

const uint32_t vmprog_mathtest[] = {

	/*
	 * Test Logical Shift Left
	 */
	I2C_VM_ASM_SET_IMM(VM_R0, 1),  /* initialize r0 to 1 */

	I2C_VM_ASM_SET_IMM(VM_R6, 32), /* set counter to 32 */
/* Start loop */
	I2C_VM_ASM_SEND_UAVO(),	       /* Set the UAVO */
	I2C_VM_ASM_DELAY(500),	       /* Pause */
	I2C_VM_ASM_SL_IMM(VM_R0, 1),   /* shift the bit left by 1 */
	I2C_VM_ASM_ADD_IMM(VM_R6, -1), /* decrement the counter */
/* End loop */
	I2C_VM_ASM_BNZ(VM_R6, -4),     /* loop until counter is zero */

	/*
	 * Test Arithmetic Shift Right
	 */
	/* Set r0 to 0x80000000 */
	I2C_VM_ASM_SET_IMM(VM_R1, 0x8000),
	I2C_VM_ASM_SL_IMM(VM_R1, 16),

	I2C_VM_ASM_SET_IMM(VM_R6, 32), /* set counter to 32 */
/* Start loop */
	I2C_VM_ASM_SEND_UAVO(),	       /* Set the UAVO */
	I2C_VM_ASM_DELAY(500),	       /* Pause */
	I2C_VM_ASM_ASR_IMM(VM_R1, 1),
	I2C_VM_ASM_ADD_IMM(VM_R6, -1), /* decrement the counter */
/* End loop */
	I2C_VM_ASM_BNZ(VM_R6, -4),     /* loop until counter is zero */

	/*
	 * Test Add Immediate
	 */
	I2C_VM_ASM_SET_IMM(VM_R1, 1000),
	I2C_VM_ASM_SET_IMM(VM_R6, 100), /* set counter to 100 */
/* Start loop */
	I2C_VM_ASM_SEND_UAVO(),	       /* Set the UAVO */
	I2C_VM_ASM_DELAY(500),	       /* Pause */
	I2C_VM_ASM_ADD_IMM(VM_R1, -10),
	I2C_VM_ASM_ADD_IMM(VM_R6, -1), /* decrement the counter */
/* End loop */
	I2C_VM_ASM_BNZ(VM_R6, -4),     /* loop until counter is zero */

	/*
	 * Test Add Registers
	 */
	I2C_VM_ASM_SET_IMM(VM_R1, 50000),
	I2C_VM_ASM_SET_IMM(VM_R2, 100),
	I2C_VM_ASM_SET_IMM(VM_R6, 19), /* set counter to 19 */
/* Start loop */
	I2C_VM_ASM_SEND_UAVO(),	       /* Set the UAVO */
	I2C_VM_ASM_DELAY(500),	       /* Pause */
	I2C_VM_ASM_ADD(VM_R1, VM_R1, VM_R1),
	I2C_VM_ASM_ADD_IMM(VM_R6, -1), /* decrement the counter */
/* End loop */
	I2C_VM_ASM_BNZ(VM_R6, -4),     /* loop until counter is zero */
};

#define NELEMENTS(x) (sizeof(x) / sizeof(*(x)))
const uint32_t vmprog_mathtest_len = NELEMENTS(vmprog_mathtest);

/**
 * @}
 * @}
 */
