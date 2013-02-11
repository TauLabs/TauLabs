/**
 ******************************************************************************
 * @file       vmprog_pcf8591.c
 * @author     TauLabs, http://github.com/TauLabs, Copyright (C) 2013
 * @addtogroup I2C Virtual Machine
 * @{
 * @addtogroup %CLASS%
 * @{
 * @brief I2C Virtual machine program for the PCF8591 chip
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
#include "pios_pcf8591.h"	/* PIOS_PCF8591_* */

const uint32_t vmprog_pcf8591[] = {
	I2C_VM_ASM_DELAY(255),
	I2C_VM_ASM_SET_DEV_ADDR(PCF8591),
	I2C_VM_ASM_STORE(PCF8591_ADC_AUTO_INCREMENT, 0),
	I2C_VM_ASM_WRITE_I2C(0, 1),
	I2C_VM_ASM_READ_I2C(0,1),
	I2C_VM_ASM_READ_I2C(0,4),
	I2C_VM_ASM_SEND_UAVO(),
	I2C_VM_ASM_JUMP(-2)
};

const uint32_t vmprog_pcf8591_len = NELEMENTS(vmprog_pcf8591);
