/**
 ******************************************************************************
 * @file       i2c_vm_asm.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012
 * @addtogroup I2C VirtualMachines
 * @{
 * @addtogroup 
 * @{
 * @brief Minimal Assembler for the Generic Programmable I2C Virtual Machine
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

#ifndef I2C_VM_ASM_H_
#define I2C_VM_ASM_H_

/* Can support up to 256 op codes */
enum i2c_vm_opcodes {
	/* Program Flow operations */
	I2C_VM_OP_HALT,         /* Halt */
	I2C_VM_OP_NOP,          /* No operation */
	I2C_VM_OP_DELAY,        /* Wait (ms) */
	I2C_VM_OP_BNZ,          /* Branch if register is not equal to zero */
	I2C_VM_OP_JUMP,         /* Jump relative */

	/* RAM operations */
	I2C_VM_OP_STORE,        /* Store value */
	I2C_VM_OP_LOAD_BE,      /* Load big endian */
	I2C_VM_OP_LOAD_LE,      /* Load little endian */

	/* Arithmetic operations */
	I2C_VM_OP_SET_IMM,      /* Set register to immediate data */
	I2C_VM_OP_ADD,          /* Add two registers */
	I2C_VM_OP_ADD_IMM,      /* Add immediate data to register */
	I2C_VM_OP_MUL,          /* Multiply two registers */
	I2C_VM_OP_MUL_IMM,      /* Multiply register by immediate data */
	I2C_VM_OP_DIV,          /* Divide two registers */
	I2C_VM_OP_DIV_IMM,      /* Divide register by immediate data */

	/* Logical operations */
	I2C_VM_OP_SL_IMM,       /* Logical Shift Left by immediate data bits */
	I2C_VM_OP_LSR_IMM,      /* Logical Shift Right by immediate data bits */
	I2C_VM_OP_ASR_IMM,      /* Arithmetic Shift Right by immediate data bits */
	I2C_VM_OP_OR_IMM,       /* OR immediate data with register */
	I2C_VM_OP_AND,          /* AND two registers */

	/* I2C operations */
	I2C_VM_OP_SET_DEV_ADDR, /* Set I2C device address */
	I2C_VM_OP_READ,         /* Read from I2C bus */
	I2C_VM_OP_WRITE,        /* Write to I2C bus */

	/* UAVO operations */
	I2C_VM_OP_SEND_UAVO,    /* Send UAV Object */
};

/* Register names */
enum i2c_vm_reg_names {
	VM_PC,

	VM_R0,
	VM_R1,
	VM_R2,
	VM_R3,
	VM_R4,
	VM_R5,
	VM_R6,
};

#define I2C_VM_ASM(operator, op1, op2, op3) (((operator & 0xFF) << 24) | \
						((op1 & 0xFF) << 16) | \
						((op2 & 0xFF) << 8) | \
						((op3 & 0xFF)))

#define I2C_VM_ASM_SIMM(operator, op1, op2) (((operator & 0xFF) << 24) | \
						((op1 & 0xFF) << 16) | \
						((op2 & 0xFFFF)))
/* Program flow operations */
#define I2C_VM_ASM_HALT()                          (I2C_VM_ASM(I2C_VM_OP_HALT, 0, 0, 0))
#define I2C_VM_ASM_NOP()                           (I2C_VM_ASM(I2C_VM_OP_NOP, 0, 0, 0))
#define I2C_VM_ASM_BNZ(reg, rel_addr)              (I2C_VM_ASM_SIMM(I2C_VM_OP_BNZ, (reg), (rel_addr)))
#define I2C_VM_ASM_DELAY(ms)                       (I2C_VM_ASM_SIMM(I2C_VM_OP_DELAY, 0, (ms)))
#define I2C_VM_ASM_JUMP(rel_addr)                  (I2C_VM_ASM_SIMM(I2C_VM_OP_JUMP, 0, (rel_addr)))

/* RAM operations */
#define I2C_VM_ASM_STORE(value, addr)              (I2C_VM_ASM(I2C_VM_OP_STORE, (value), (addr), 0))
#define I2C_VM_ASM_LOAD_BE(addr, length, dest_reg) (I2C_VM_ASM(I2C_VM_OP_LOAD_BE, (addr), (length), (dest_reg)))
#define I2C_VM_ASM_LOAD_LE(addr, length, dest_reg) (I2C_VM_ASM(I2C_VM_OP_LOAD_LE, (addr), (length), (dest_reg)))

/* Arithmetic operations */
#define I2C_VM_ASM_SET_IMM(reg, simm)              (I2C_VM_ASM_SIMM(I2C_VM_OP_SET_IMM, (reg), (simm)))
#define I2C_VM_ASM_ADD(dest_reg, reg_a, reg_b)     (I2C_VM_ASM(I2C_VM_OP_ADD, (dest_reg), (reg_a), (reg_b)))
#define I2C_VM_ASM_ADD_IMM(reg, simm)              (I2C_VM_ASM_SIMM(I2C_VM_OP_ADD_IMM, (reg), (simm)))
#define I2C_VM_ASM_MUL(dest_reg, reg_a, reg_b)     (I2C_VM_ASM(I2C_VM_OP_MUL, (dest_reg), (reg_a), (reg_b)))
#define I2C_VM_ASM_MUL_IMM(reg, simm)              (I2C_VM_ASM_SIMM(I2C_VM_OP_MUL_IMM, (reg), (simm)))
#define I2C_VM_ASM_DIV(dest_reg, reg_a, reg_b)     (I2C_VM_ASM(I2C_VM_OP_DIV, (dest_reg), (reg_a), (reg_b)))
#define I2C_VM_ASM_DIV_IMM(reg, simm)              (I2C_VM_ASM_SIMM(I2C_VM_OP_DIV_IMM, (reg), (simm)))

/* Logical operations */
#define I2C_VM_ASM_SL_IMM(reg, simm)               (I2C_VM_ASM_SIMM(I2C_VM_OP_SL_IMM, (reg), (simm)))
#define I2C_VM_ASM_LSR_IMM(reg, simm)              (I2C_VM_ASM_SIMM(I2C_VM_OP_LSR_IMM, (reg), (simm)))
#define I2C_VM_ASM_ASR_IMM(reg, simm)              (I2C_VM_ASM_SIMM(I2C_VM_OP_ASR_IMM, (reg), (simm)))
#define I2C_VM_ASM_OR_IMM(reg, simm)               (I2C_VM_ASM_SIMM(I2C_VM_OP_OR_IMM, (reg), (simm)))
#define I2C_VM_ASM_AND(dest_reg, reg_a, reg_b)     (I2C_VM_ASM(I2C_VM_OP_AND, (dest_reg), (reg_a), (reg_b)))

/* I2C operations */
#define I2C_VM_ASM_SET_DEV_ADDR(addr)              (I2C_VM_ASM(I2C_VM_OP_SET_DEV_ADDR, (addr), 0, 0))
#define I2C_VM_ASM_READ_I2C(addr, length)          (I2C_VM_ASM(I2C_VM_OP_READ, (addr), (length), 0))
#define I2C_VM_ASM_WRITE_I2C(addr, length)         (I2C_VM_ASM(I2C_VM_OP_WRITE, (addr), (length), 0))

/* UAVO operations */
#define I2C_VM_ASM_SEND_UAVO()                     (I2C_VM_ASM(I2C_VM_OP_SEND_UAVO, 0, 0, 0))

#endif /* I2C_VM_ASM_H_ */

/**
 * @}
 * @}
 */
