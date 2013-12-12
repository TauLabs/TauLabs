/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup GenericI2CSensor Generic I2C sensor interface
 * @{
 *
 * @file       i2c_vm.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      The virtual machine for I2C sensors
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

#include <pios.h>
#if defined(PIOS_INCLUDE_I2C)

#include <stdint.h>	      /* uint8_t, uint32_t, etc */
#include <stdbool.h>	      /* bool */
#include "uavobjectmanager.h" /* UAVO types */
#include "i2cvm.h"	      /* UAVO that holds VM state snapshots */
#include "i2c_vm_asm.h"	      /* Minimal assembler for I2C VM */

struct i2c_vm_regs {
	bool     halted;
	bool     fault;

	uintptr_t i2c_adapter;
	uint8_t i2c_dev_addr;

	I2CVMData uavo;
};

/******************************
 *
 * VM internal helper functions
 *
 *****************************/

#define SIMM_VAL(msb,lsb) ((int16_t)((((msb) & 0xFF) << 8) | ((lsb) & 0xFF)))

/* Assign a 32-bit value into virtual machine registers
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] reg destination register where value will be stored
 * @param[in] val value to store in register
 */
static bool i2c_vm_set_reg (struct i2c_vm_regs * vm_state, uint8_t reg, uint32_t val)
{
	switch (reg) {
	case VM_R0:
		vm_state->uavo.r0 = val;
		break;
	case VM_R1:
		vm_state->uavo.r1 = val;
		break;
	case VM_R2:
		vm_state->uavo.r2 = val;
		break;
	case VM_R3:
		vm_state->uavo.r3 = val;
		break;
	case VM_R4:
		vm_state->uavo.r4 = val;
		break;
	case VM_R5:
		vm_state->uavo.r5 = val;
		break;
	case VM_R6:
		vm_state->uavo.r6 = val;
		break;
	default:
		return false;
	}

	return true;
}

/* Read a 32-bit value from virtual machine registers
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] reg register where value will be read from
 * @param[out] val pointer to hold the value of the register
 */
static bool i2c_vm_get_reg (const struct i2c_vm_regs * vm_state, uint8_t reg, uint32_t * val)
{
	switch (reg) {
	case VM_R0:
		*val = vm_state->uavo.r0;
		break;
	case VM_R1:
		*val = vm_state->uavo.r1;
		break;
	case VM_R2:
		*val = vm_state->uavo.r2;
		break;
	case VM_R3:
		*val = vm_state->uavo.r3;
		break;
	case VM_R4:
		*val = vm_state->uavo.r4;
		break;
	case VM_R5:
		*val = vm_state->uavo.r5;
		break;
	case VM_R6:
		*val = vm_state->uavo.r6;
		break;
	default:
		return false;
	}

	return true;
}

enum aluop {
	ALUOP_ADD,
	ALUOP_MUL,
	ALUOP_DIV,
	ALUOP_LSR,
	ALUOP_ASR,
	ALUOP_SL,
	ALUOP_OR,
	ALUOP_AND,
};

/* Perform an ALU operation and update the virtual machine state (rd = ra op rb)
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] operation ALU operation to be performed
 * @param[in] rd destination register for result
 * @param[in] ra first operand register
 * @param[in] rb second operand register
 */
static bool i2c_vm_aluop_reg (struct i2c_vm_regs * vm_state, enum aluop operation, uint8_t rd, uint8_t ra, uint8_t rb)
{
	uint32_t ra_val;
	uint32_t rb_val;
	uint32_t rd_val;

	/* Load ra value */
	if (!i2c_vm_get_reg(vm_state, ra, &ra_val))
		return false;

	/* Load rb value */
	if (!i2c_vm_get_reg(vm_state, rb, &rb_val))
		return false;

	/* Compute result */
	switch (operation) {
	case ALUOP_ADD:
		rd_val = (int32_t)ra_val + (int32_t)rb_val;
		break;
	case ALUOP_MUL:
		rd_val = (int32_t)ra_val * (int32_t)rb_val;
		break;
	case ALUOP_DIV:
		rd_val = (int32_t)ra_val / (int32_t)rb_val;
		break;
	case ALUOP_AND:
		rd_val = ra_val & rb_val;
		break;
	default:
		/* invalid operation, fault */
		return false;
	}

	/* Write value into destination register */
	if (!i2c_vm_set_reg(vm_state, rd, rd_val))
		return false;

	return true;
}

/* Perform an ALU operation and update the virtual machine state (rd = rd op simm)
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] operation ALU operation to be performed
 * @param[in] rd destination register for result
 * @param[in] simm_hi,simm_lo short immediate data
 */
static bool i2c_vm_aluop_imm (struct i2c_vm_regs * vm_state, enum aluop operation, uint8_t rd, uint8_t simm_hi, uint8_t simm_lo)
{
	uint32_t rd_val;
	int16_t  simm_val;

	/* Load rd value */
	if (!i2c_vm_get_reg(vm_state, rd, &rd_val))
		return false;

	/* Load short-immediate data */
	simm_val = SIMM_VAL(simm_hi, simm_lo);

	/* Compute result */
	switch (operation) {
	case ALUOP_ADD:
		rd_val += simm_val;
		break;
	case ALUOP_MUL:
		rd_val *= simm_val;
		break;
	case ALUOP_DIV:
		rd_val /= simm_val;
		break;
	case ALUOP_OR:
		rd_val |= (uint16_t)simm_val;
		break;
	case ALUOP_ASR:
		{
			/* NOTE this must be a signed integer to force the >> to be an arithmetic shift */
			int32_t rd_signed;

			rd_signed = rd_val;
			rd_signed >>= (uint8_t)(simm_val & 0x1F);
			rd_val = rd_signed;
		}
		break;
	case ALUOP_LSR:
		rd_val >>= (uint8_t)(simm_val & 0x1F);
		break;
	case ALUOP_SL:
		rd_val <<= (uint8_t)(simm_val & 0x1F);
		break;
	default:
		/* invalid operation, fault */
		return false;
	}

	/* Write value into destination register */
	if (!i2c_vm_set_reg(vm_state, rd, rd_val))
		return false;

	return true;
}

/*********************
 *
 * VM opcode execution
 *
 ********************/

/* Halt the virtual machine
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] op1,op2,op3 unused
 */
static bool i2c_vm_halt (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	vm_state->halted = true;
	return true;
}

/* Virtual machine no operation instruction
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] op1,op2,op3 unused
 */
static bool i2c_vm_nop (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	vm_state->uavo.pc++;
	return true;
}

/* Set virtual machine register
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd register be set
 * @param[in] imm_hi,imm_lo short immediate data
 */
static bool i2c_vm_set_imm (struct i2c_vm_regs * vm_state, uint8_t rd, uint8_t imm_hi, uint8_t imm_lo)
{
	if (!i2c_vm_set_reg(vm_state, rd, (uint32_t)SIMM_VAL(imm_hi, imm_lo)))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* Store virtual machine data in RAM
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] val value to be stored in RAM
 * @param[in] ram_addr address (in virtual RAM) where value will be stored
 * @param[in] op3 unused
 */
static bool i2c_vm_store (struct i2c_vm_regs * vm_state, uint8_t val, uint8_t ram_addr, uint8_t op3)
{
	if (ram_addr >= sizeof(vm_state->uavo.ram))
		return false;

	vm_state->uavo.ram[ram_addr] = val;
	vm_state->uavo.pc++;
	return true;
}


/* Load register information in Big Endian format
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] addr address (in virtual RAM) where value should be loaded from
 * @param[in] len number of bytes to load from RAM
 * @param[in] rd register to store the value into
 */
static bool i2c_vm_load_be (struct i2c_vm_regs * vm_state, uint8_t addr, uint8_t len, uint8_t rd)
{
	if ((len < 1) || (len > 4))
		return false;

	if (addr >= sizeof(vm_state->uavo.ram))
		return false;

	uint32_t val = 0;
	memcpy ((void *)((uintptr_t)&val + (4 - len)), &(vm_state->uavo.ram[addr]), len);

	/* Handle byte swapping */
	val = (((val & 0xFF000000) >> 24) |
		((val & 0x00FF0000) >> 8) |
		((val & 0x0000FF00) << 8) |
		((val & 0x000000FF) << 24));

	if (!i2c_vm_set_reg (vm_state, rd, val))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* Load register information in Little Endian format
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] addr address (in virtual RAM) where value should be loaded from
 * @param[in] len number of bytes to load from RAM
 * @param[in] rd register to store the value into
 */
static bool i2c_vm_load_le (struct i2c_vm_regs * vm_state, uint8_t addr, uint8_t len, uint8_t rd)
{
	if ((len < 1) || (len > 4))
		return false;

	if (addr >= sizeof(vm_state->uavo.ram))
		return false;

	uint32_t val = 0;
	memcpy (&val, &(vm_state->uavo.ram[addr]), len);

	if (!i2c_vm_set_reg (vm_state, rd, val))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* ADD: rd = ra + rb
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd destination register for result
 * @param[in] ra first operand register
 * @param[in] rb second operand register
 */
static bool i2c_vm_add_reg (struct i2c_vm_regs * vm_state, uint8_t rd, uint8_t ra, uint8_t rb)
{
	if (!i2c_vm_aluop_reg(vm_state, ALUOP_ADD, rd, ra, rb))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* ADD: rd += simm
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd destination register for result
 * @param[in] imm_hi,imm_lo short immediate data
 */
static bool i2c_vm_add_imm (struct i2c_vm_regs * vm_state, uint8_t rd, uint8_t imm_hi, uint8_t imm_lo)
{
	if (!i2c_vm_aluop_imm(vm_state, ALUOP_ADD, rd, imm_hi, imm_lo))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* Multiply: rd = ra * rb
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd destination register for result
 * @param[in] ra first operand register
 * @param[in] rb second operand register
 */
static bool i2c_vm_mul_reg (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	if (!i2c_vm_aluop_reg(vm_state, ALUOP_MUL, op1, op2, op3))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* Multiply: rd *= simm
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd destination register for result
 * @param[in] imm_hi,imm_lo short immediate data
 */
static bool i2c_vm_mul_imm (struct i2c_vm_regs * vm_state, uint8_t rd, uint8_t imm_hi, uint8_t imm_lo)
{
	if (!i2c_vm_aluop_imm(vm_state, ALUOP_MUL, rd, imm_hi, imm_lo))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* Divide: rd = ra / rb
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd destination register for result
 * @param[in] ra first operand register
 * @param[in] rb second operand register
 */
static bool i2c_vm_div_reg (struct i2c_vm_regs * vm_state, uint8_t rd, uint8_t ra, uint8_t rb)
{
	if (!i2c_vm_aluop_reg(vm_state, ALUOP_DIV, rd, ra, rb))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* Divide: rd /= simm
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd destination register for result
 * @param[in] imm_hi,imm_lo short immediate data
 */
static bool i2c_vm_div_imm (struct i2c_vm_regs * vm_state, uint8_t rd, uint8_t imm_hi, uint8_t imm_lo)
{
	if (!i2c_vm_aluop_imm(vm_state, ALUOP_DIV, rd, imm_hi, imm_lo))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* OR: rd |= (simm & 0xFFFF)
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd destination register for result
 * @param[in] imm_hi,imm_lo short immediate data
 */
static bool i2c_vm_or_imm (struct i2c_vm_regs * vm_state, uint8_t rd, uint8_t imm_hi, uint8_t imm_lo)
{
	if (!i2c_vm_aluop_imm(vm_state, ALUOP_OR, rd, imm_hi, imm_lo))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* AND: rd = ra & rb
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd destination register for result
 * @param[in] ra first operand register
 * @param[in] rb second operand register
 */
static bool i2c_vm_and_reg (struct i2c_vm_regs * vm_state, uint8_t rd, uint8_t ra, uint8_t rb)
{
	if (!i2c_vm_aluop_reg(vm_state, ALUOP_AND, rd, ra, rb))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* Arithmetic Shift Right (ASR): signed(rd) >>= (simm & 0x1F)
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd destination register for result
 * @param[in] imm_hi,imm_lo number of bits to shift (short immediate data)
 */
static bool i2c_vm_asr_imm (struct i2c_vm_regs * vm_state, uint8_t rd, uint8_t imm_hi, uint8_t imm_lo)
{
	if (!i2c_vm_aluop_imm(vm_state, ALUOP_ASR, rd, imm_hi, imm_lo))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* Logical Shift Right (LSR): rd >>= (simm & 0x1F)
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd destination register for result
 * @param[in] imm_hi,imm_lo number of bits to shift (short immediate data)
 */
static bool i2c_vm_lsr_imm (struct i2c_vm_regs * vm_state, uint8_t rd, uint8_t imm_hi, uint8_t imm_lo)
{
	if (!i2c_vm_aluop_imm(vm_state, ALUOP_LSR, rd, imm_hi, imm_lo))
		return false;

	vm_state->uavo.pc++;
	return true;
}


/* Logical Shift Left (SL): rd <<= (simm & 0x1F)
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] rd destination register for result
 * @param[in] imm_hi,imm_lo number of bits to shift (short immediate data)
 */
static bool i2c_vm_sl_imm (struct i2c_vm_regs * vm_state, uint8_t rd, uint8_t imm_hi, uint8_t imm_lo)
{
	if (!i2c_vm_aluop_imm(vm_state, ALUOP_SL, rd, imm_hi, imm_lo))
		return false;

	vm_state->uavo.pc++;
	return true;
}

/* Jump: pc += simm
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] op1 unused
 * @param[in] imm_hi,imm_lo offset to apply to the program counter (short immediate data)
 */
static bool i2c_vm_jump (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t imm_hi, uint8_t imm_lo)
{
	vm_state->uavo.pc += SIMM_VAL(imm_hi, imm_lo);

	return true;
}

/* Branch If Not Zero: pc += simm IFF (ra == 0)
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] ra register to compare
 * @param[in] imm_hi,imm_lo short immediate data
 */
static bool i2c_vm_bnz (struct i2c_vm_regs * vm_state, uint8_t ra, uint8_t imm_hi, uint8_t imm_lo)
{
	uint32_t ra_val;

	/* Load ra value */
	if (!i2c_vm_get_reg(vm_state, ra, &ra_val))
		return false;

	if (ra_val) {
		return (i2c_vm_jump(vm_state, 0, imm_hi, imm_lo));
	} else {
		vm_state->uavo.pc++;
	}

	return true;
}

/* Set I2C device address in virtual machine
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] i2c_dev_addr 7-bit I2C device address to be used in future I2C transfers
 * @param[in] op2,op3 unused
 */
static bool i2c_vm_set_dev_addr (struct i2c_vm_regs * vm_state, uint8_t i2c_dev_addr, uint8_t op2, uint8_t op3)
{
	vm_state->i2c_dev_addr = i2c_dev_addr;
	vm_state->uavo.pc++;
	return true;
}

/* Read I2C data into virtual machine RAM
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] ram_addr base address (in virtual RAM) where data will be stored
 * @param[in] len number of bytes to read from the i2c bus and store into virtual RAM
 * @param[in] op3 unused
 */
static bool i2c_vm_read (struct i2c_vm_regs * vm_state, uint8_t ram_addr, uint8_t len, uint8_t op3)
{
	/* Make sure our read fits in our buffer */
	if ((ram_addr + len) > sizeof(vm_state->uavo.ram)) {
		return false;
	}

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = vm_state->i2c_dev_addr,
			.rw   = PIOS_I2C_TXN_READ,
			.len  = len,
			.buf  = vm_state->uavo.ram + ram_addr,
		},
	};

	int32_t rc = PIOS_I2C_Transfer(vm_state->i2c_adapter, txn_list, NELEMENTS(txn_list));

	/* Fault the VM if the I2C transfer fails */
	if (rc < 0)
		return false;

	vm_state->uavo.pc++;

	return (true);
}

/* Write I2C data from virtual machine RAM
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] ram_addr base address (in virtual RAM) where data will be read from
 * @param[in] len number of bytes to read from virtual RAM and write to the the i2c bus
 * @param[in] op3 unused
 */
static bool i2c_vm_write (struct i2c_vm_regs * vm_state, uint8_t ram_addr, uint8_t len, uint8_t op3)
{
	if ((ram_addr + len) > sizeof(vm_state->uavo.ram)) {
		return false;
	}

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = vm_state->i2c_dev_addr,
			.rw   = PIOS_I2C_TXN_WRITE,
			.len  = len,
			.buf  = vm_state->uavo.ram + ram_addr,
		},
	};

	uint32_t rc = PIOS_I2C_Transfer(vm_state->i2c_adapter, txn_list, NELEMENTS(txn_list));

	/* Fault the VM if the I2C transfer fails */
	if (rc < 0)
		return false;

	vm_state->uavo.pc++;

	return (true);
}

/* Send UAVObject from virtual machine registers
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] op1,op2,op3 unused
 */
static bool i2c_vm_send_uavo (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	/* Push our local copy of the UAVO */
	I2CVMSet(&vm_state->uavo);

	vm_state->uavo.pc++;
	return true;
}

/* Make virtual machine wait
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] op1 unused
 * @param[in] imm_hi,imm_lo number of ms to wait (short immediate data)
 */
static bool i2c_vm_delay (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t imm_hi, uint8_t imm_lo)
{
	vTaskDelay(MS2TICKS(SIMM_VAL(imm_hi, imm_lo)));

	vm_state->uavo.pc++;
	return true;
}

/* Reboot virtual machine
 *
 * @param[in,out] vm_state virtual machine state
 * @param[in] op1,op2,op3 unused
 */
static bool i2c_vm_reboot (struct i2c_vm_regs * vm_state, uintptr_t i2c_adapter)
{
	vm_state->halted  = false;
	vm_state->fault   = false;

	/* Reset I2C configuration */
	vm_state->i2c_dev_addr = 0;
	vm_state->i2c_adapter  = i2c_adapter;

	/* Reset register state */
	vm_state->uavo.pc = 0;
	vm_state->uavo.r0 = 0;
	vm_state->uavo.r1 = 0;
	vm_state->uavo.r2 = 0;
	vm_state->uavo.r3 = 0;
	vm_state->uavo.r4 = 0;
	vm_state->uavo.r5 = 0;
	vm_state->uavo.r6 = 0;
	memset(vm_state->uavo.ram, 0, sizeof(vm_state->uavo.ram));

	return true;
}


typedef bool (*i2c_vm_inst_handler) (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3);

const i2c_vm_inst_handler i2c_vm_handlers[] = {
	/* Program flow operations */
	[I2C_VM_OP_HALT]         = i2c_vm_halt,         /* Halt */
	[I2C_VM_OP_NOP]          = i2c_vm_nop,          /* No operation */
	[I2C_VM_OP_DELAY]        = i2c_vm_delay,        /* Wait (ms) */
	[I2C_VM_OP_BNZ]          = i2c_vm_bnz,          /* Branch if register is not zero */
	[I2C_VM_OP_JUMP]         = i2c_vm_jump,         /* Jump relative */

	/* RAM operations */
	[I2C_VM_OP_STORE]        = i2c_vm_store,        /* Store value */
	[I2C_VM_OP_LOAD_BE]      = i2c_vm_load_be,      /* Load big endian */
	[I2C_VM_OP_LOAD_LE]      = i2c_vm_load_le,      /* Load little endian */

	/* Arithmetic operations */
	[I2C_VM_OP_SET_IMM]      = i2c_vm_set_imm,      /* Set register to immediate data */
	[I2C_VM_OP_ADD]          = i2c_vm_add_reg,      /* Add two registers */
	[I2C_VM_OP_ADD_IMM]      = i2c_vm_add_imm,      /* Add immediate data to register */
	[I2C_VM_OP_MUL]          = i2c_vm_mul_reg,      /* Multiply two registers */
	[I2C_VM_OP_MUL_IMM]      = i2c_vm_mul_imm,      /* Multiply register by immediate data */
	[I2C_VM_OP_DIV]          = i2c_vm_div_reg,      /* Divide two registers */
	[I2C_VM_OP_DIV_IMM]      = i2c_vm_div_imm,      /* Divide register by immediate data */

	/* Logical operations */
	[I2C_VM_OP_SL_IMM]       = i2c_vm_sl_imm,       /* Shift left */
	[I2C_VM_OP_LSR_IMM]      = i2c_vm_lsr_imm,      /* Logical Shift Right */
	[I2C_VM_OP_ASR_IMM]      = i2c_vm_asr_imm,      /* Arithmetic Shift Right */
	[I2C_VM_OP_OR_IMM]       = i2c_vm_or_imm,       /* Logical OR of register and immediate data */
	[I2C_VM_OP_AND]          = i2c_vm_and_reg,      /* Logical AND of two registers */

	/* I2C operations */
	[I2C_VM_OP_SET_DEV_ADDR] = i2c_vm_set_dev_addr, /* Set I2C device address */
	[I2C_VM_OP_READ]         = i2c_vm_read,         /* Read from I2C bus */
	[I2C_VM_OP_WRITE]        = i2c_vm_write,        /* Write to I2C bus */

	/* UAVO operations */
	[I2C_VM_OP_SEND_UAVO]    = i2c_vm_send_uavo,    /* Send UAV Object */
};

/* Run virtual machine. This is the code that loops through and interprets all the instructions
 *
 * @param[in] code pointer to program to execute
 * @param[in] code_len number of 32-bit instructions contained in the program
 * @param[in] i2c_adapter opaque I2C adapter handle to use for i2c transactions
 *
 */
bool i2c_vm_run (const uint32_t * code, uint8_t code_len, uintptr_t i2c_adapter)
{
	if (code == NULL || code_len == 0)
		return false;

	static struct i2c_vm_regs vm;

	i2c_vm_reboot (&vm, i2c_adapter);

	while (!vm.halted) {
		if (vm.uavo.pc > code_len) {
			/* PC is entirely out of range */
			vm.fault  = true;
			vm.halted = true;
			continue;
		}
		if (vm.uavo.pc == code_len) {
			/* PC is just past the end of the code, assume program is completed */
			vm.halted = true;
			continue;
		}
		/* Fetch */
		uint32_t instruction = code[vm.uavo.pc];

		/* Decode */
		uint8_t operator = (instruction & 0xFF000000) >> 24;
		uint8_t op1      = (instruction & 0x00FF0000) >> 16;
		uint8_t op2      = (instruction & 0x0000FF00) >>  8;
		uint8_t op3      = (instruction & 0x000000FF);

		if (operator > NELEMENTS(i2c_vm_handlers)) {
			vm.fault = true;
			vm.halted = true;
			continue;
		}
		i2c_vm_inst_handler f = i2c_vm_handlers[operator];

		/* Execute + Writeback */
		if (!f || !f(&vm, op1, op2, op3)) {
			vm.fault  = true;
			vm.halted = true;
			continue;
		}
	}

	return (!vm.fault);
}

#endif /* PIOS_INCLUDE_I2C */

/**
 * @}
 * @}
 */

