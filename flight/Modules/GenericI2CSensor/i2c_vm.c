/**
 ******************************************************************************
 * @file       i2c_vm.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @addtogroup I2C VirtualMachines
 * @{
 * @addtogroup 
 * @{
 * @brief Generic Programmable I2C Virtual Machine
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
#include "uavobjectmanager.h" /* UAVO types */
#include "i2cvm.h"	      /* UAVO that holds VM state snapshots */
#include "i2c_vm_asm.h"	      /* Minimal assembler for I2C VM */

#define I2C_VM_RAM_SIZE 8

struct i2c_vm_regs {
	bool     halted;
	bool     fault;
	uint8_t  pc;
	uint32_t ctr;

	uintptr_t i2c_adapter;
	uint8_t i2c_dev_addr;
	uint8_t ram[I2C_VM_RAM_SIZE];

	I2CVMData uavo;

	int32_t  r0;
	int32_t  r1;
	int32_t  r2;
	int32_t  r3;
	int32_t  r4;
	int32_t  r5;
	float    f0;
	float    f1;
	float    f2;
	float    f3;
};

/* Halt the virtual machine
 *
 * No operands
 */
static bool i2c_vm_halt (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	vm_state->halted = true;
	return (true);
}

/* Virtual machine no operation instruction
 *
 * No operands
 */
static bool i2c_vm_nop (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	vm_state->pc++;
	return (true);
}

/* Set virtual machine counter register
 *
 * op1: new value for ctr register
 */
static bool i2c_vm_set_ctr (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	vm_state->ctr = op1;
	vm_state->pc++;
	return (true);
}

/* Store virtual machine data in RAM
 *
 * op1: value to store into ram
 * op2: offset in ram where value is to be stored
 */
static bool i2c_vm_store (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	if (op2 >= sizeof(vm_state->ram)) {
		return (false);
	}

	vm_state->ram[op2] = op1;
	vm_state->pc++;
	return (true);
}


/* Load data into virtual machine registers
 *
 * val: value to store in register
 * op3: destination register where value will be stored
 */
static bool i2c_vm_load (struct i2c_vm_regs * vm_state, uint32_t val, uint8_t op3)
{
	switch (op3) {
	case VM_R0:
		vm_state->r0 = val;
		break;
	case VM_R1:
		vm_state->r1 = val;
		break;
	case VM_R2:
		vm_state->r2 = val;
		break;
	case VM_R3:
		vm_state->r3 = val;
		break;
	case VM_R4:
		vm_state->r4 = val;
		break;
	case VM_R5:
		vm_state->r5 = val;
		break;
	case VM_F0:
		vm_state->f0 = val;
		break;
	case VM_F1:
		vm_state->f1 = val;
		break;
	case VM_F2:
		vm_state->f2 = val;
		break;
	case VM_F3:
		vm_state->f3 = val;
		break;
	default:
		return (false);
	}

	vm_state->pc++;
	return (true);
}

/* Load register information in Big Endian format
 *
 * op1: offset in RAM where value is located
 * op2: length of data to read from RAM
 * op3: destination register to store the converted result into
 */
static bool i2c_vm_load_be (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	if ((op2 < 1) || (op2 > 4)) {
		return (false);
	}

	uint32_t val = 0;
	memcpy ((void *)((uintptr_t)&val + (4 - op2)), &(vm_state->ram[op1]), op2);

	/* Handle byte swapping */
	val = (((val & 0xFF000000) >> 24) |
		((val & 0x00FF0000) >> 8) |
		((val & 0x0000FF00) << 8) |
		((val & 0x000000FF) << 24));

	return (i2c_vm_load (vm_state, val, op3));
}

/* Load register information in Little Endian format
 *
 * op1: offset in RAM where value is located
 * op2: length of data to read from RAM
 * op3: destination register to store the converted result into
 */
static bool i2c_vm_load_le (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	if ((op2 < 1) || (op2 > 4)) {
		return (false);
	}

	uint32_t val = 0;
	memcpy (&val, &(vm_state->ram[op1]), op2);

	return (i2c_vm_load (vm_state, val, op3));
}

/* Subtract one from counter register, saturate at zero
 *
 * No operands
 */
static bool i2c_vm_dec_ctr (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	if (vm_state->ctr > 0) {
		vm_state->ctr--;
	}
	vm_state->pc++;
	return (true);
}

/* Virtual machine jump operation
 *
 * op1: value to add to the current PC
 *      (operand is a signed 2's complement value)
 */
static bool i2c_vm_jump (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	vm_state->pc += op1;
	return (true);
}

/* Virtual machine Branch If Not Zero operation
 *
 * op1: value to add to the current PC IFF ctr register is non-zero
 *      (operand is a signed 2's complement value)
 */
static bool i2c_vm_bnz (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	if (vm_state->ctr) {
		return (i2c_vm_jump(vm_state, op1, op2, op3));
	} else {
		vm_state->pc++;
	}
	return (true);
}

/* Set I2C device address in virtual machine
 *
 * op1: 7-bit I2C device address to be used in future I2C transfers
 */
static bool i2c_vm_set_dev_addr (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	vm_state->i2c_dev_addr = op1;
	vm_state->pc++;
	return (true);
}

/* Read I2C data into virtual machine RAM
 *
 * op1: offset into RAM to start writing
 * op2: number of bytes to read from I2C bus into RAM
 */
static bool i2c_vm_read (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	/* Make sure our read fits in our buffer */
	if ((op1 + op2) > sizeof(vm_state->ram)) {
		return false;
	}

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = vm_state->i2c_dev_addr,
			.rw   = PIOS_I2C_TXN_READ,
			.len  = op2,
			.buf  = vm_state->ram + op1,
		},
	};

	vm_state->pc++;

	int32_t rc = PIOS_I2C_Transfer(vm_state->i2c_adapter, txn_list, NELEMENTS(txn_list));
	if (rc < 0) {
		/* I2C Transfer failed, back-off so we don't consume all CPU on failed bus */
		/* TODO: should this fault the VM? */
		vTaskDelay(50 / portTICK_RATE_MS);
	}
	return (rc == 0);
}

/* Write I2C data from virtual machine RAM
 *
 * op1: offset into RAM to start reading
 * op2: number of bytes to write from RAM to the I2C bus
 */
static bool i2c_vm_write (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	if ((op1 + op2) > sizeof(vm_state->ram)) {
		return (false);
	}

	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = vm_state->i2c_dev_addr,
			.rw   = PIOS_I2C_TXN_WRITE,
			.len  = op2,
			.buf  = vm_state->ram + op1,
		},
	};

	vm_state->pc++;

	uint32_t rc = PIOS_I2C_Transfer(vm_state->i2c_adapter, txn_list, NELEMENTS(txn_list));
	if (rc < 0) {
		/* I2C Transfer failed, back-off so we don't consume all CPU on failed bus */
		/* TODO: should this fault the VM? */
		vTaskDelay(50 / portTICK_RATE_MS);
	}
	return (rc == 0);
}

/* Send UAVObject from virtual machine registers
 *
 * No operands
 */
static bool i2c_vm_send_uavo (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	/* Copy our exportable state into the uavo */
	vm_state->uavo.r0 = vm_state->r0;
	vm_state->uavo.r1 = vm_state->r1;
	vm_state->uavo.r2 = vm_state->r2;
	vm_state->uavo.r3 = vm_state->r3;
	vm_state->uavo.r4 = vm_state->r4;
	vm_state->uavo.r5 = vm_state->r5;
	vm_state->uavo.f0 = vm_state->f0;
	vm_state->uavo.f1 = vm_state->f1;
	vm_state->uavo.f2 = vm_state->f2;
	vm_state->uavo.f3 = vm_state->f3;

	I2CVMSet(&vm_state->uavo);

	vm_state->pc++;
	return (true);
}

/* Make virtual machine wait
 *
 * op1: number of ms to wait
 */
static bool i2c_vm_delay (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3)
{
	vTaskDelay(op1 / portTICK_RATE_MS);
	vm_state->pc++;
	return (true);
}

/* Reboot virtual machine
 *
 * No operands
 */
static bool i2c_vm_reboot (struct i2c_vm_regs * vm_state, uintptr_t i2c_adapter)
{
	vm_state->halted = false;
	vm_state->fault  = false;
	vm_state->pc     = 0;
	vm_state->ctr    = 0;

	vm_state->r0     = 0;
	vm_state->r1     = 0;
	vm_state->r2     = 0;
	vm_state->r3     = 0;
	vm_state->f0     = (float) 0.0;
	vm_state->f1     = (float) 0.0;
	vm_state->f2     = (float) 0.0;
	vm_state->f3     = (float) 0.0;

	vm_state->i2c_dev_addr = 0;
	memset(vm_state->ram, 0, sizeof(vm_state->ram));

	vm_state->i2c_adapter = i2c_adapter;

	return true;
}


typedef bool (*i2c_vm_inst_handler) (struct i2c_vm_regs * vm_state, uint8_t op1, uint8_t op2, uint8_t op3);

const i2c_vm_inst_handler i2c_vm_handlers[] = {
	[I2C_VM_OP_HALT]         = i2c_vm_halt,         /* Halt */
	[I2C_VM_OP_NOP]          = i2c_vm_nop,          /* No operation */
	[I2C_VM_OP_STORE]        = i2c_vm_store,        /* Store value */
	[I2C_VM_OP_LOAD_BE]      = i2c_vm_load_be,      /* Load big endian */
	[I2C_VM_OP_LOAD_LE]      = i2c_vm_load_le,      /* Load little endian */
	[I2C_VM_OP_SET_CTR]      = i2c_vm_set_ctr,      /* Set counter */
	[I2C_VM_OP_DEC_CTR]      = i2c_vm_dec_ctr,      /* Decrement counter */
	[I2C_VM_OP_BNZ]          = i2c_vm_bnz,          /* Branch if not zero */
	[I2C_VM_OP_JUMP]         = i2c_vm_jump,         /* Jump to */
	[I2C_VM_OP_SET_DEV_ADDR] = i2c_vm_set_dev_addr, /* Set I2C device address */
	[I2C_VM_OP_READ]         = i2c_vm_read,         /* Read from I2C bus */
	[I2C_VM_OP_WRITE]        = i2c_vm_write,        /* Write to I2C bus */
	[I2C_VM_OP_SEND_UAVO]    = i2c_vm_send_uavo,    /* Send UAV Object */
	[I2C_VM_OP_DELAY]        = i2c_vm_delay,        /* Wait (ms) */
};

/* Run virtual machine. This is the code that loops through and interprets all the instructions */
bool i2c_vm_run (const uint32_t * code, uint8_t code_len, uintptr_t i2c_adapter)
{
	static struct i2c_vm_regs vm;

	i2c_vm_reboot (&vm, i2c_adapter);

	while (!vm.halted) {
		if (vm.pc >= code_len) {
			vm.fault  = true;
			vm.halted = true;
			continue;
		}
		/* Fetch */
		uint32_t instruction = code[vm.pc];

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

/**
 * @}
 * @}
 */
