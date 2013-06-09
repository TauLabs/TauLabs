/**
 ******************************************************************************
 * @file       unittest.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup UnitTests
 * @{
 * @addtogroup UnitTests
 * @{
 * @brief Unit test
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

/*
 * NOTE: This program uses the Google Test infrastructure to drive the unit test
 *
 * Main site for Google Test: http://code.google.com/p/googletest/
 * Documentation and examples: http://code.google.com/p/googletest/wiki/Documentation
 */

#include "gtest/gtest.h"

#include <stdio.h>		/* printf */
#include <stdlib.h>		/* abort */
#include <string.h>		/* memset */
#include <stdint.h>		/* uint*_t */

extern "C" {

#include "i2c_vm_asm.h"
extern bool i2c_vm_run (const uint32_t * code, uint8_t code_len, uintptr_t i2c_adapter);

#include "i2cvm.h"		// uavo_data

}

#define NELEMENTS(x) (sizeof(x) / sizeof(*x))

// To use a test fixture, derive a class from testing::Test.
class I2CVMTest : public testing::Test {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(I2CVMTest, NullProgram) {
  EXPECT_FALSE(i2c_vm_run (NULL, 1, 0));
}

TEST_F(I2CVMTest, ZeroLengthProgram) {
  const uint32_t program[] = {
  };

  EXPECT_FALSE(i2c_vm_run (program, NELEMENTS(program), 0));
}

TEST_F(I2CVMTest, InvalidOpCodeProgram) {
  const uint32_t program[] = {
    0xFFFFFFFF,
  };

  EXPECT_FALSE(i2c_vm_run (program, NELEMENTS(program), 0));
}

TEST_F(I2CVMTest, NopProgram) {
  const uint32_t program[] = {
    I2C_VM_ASM_NOP(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));
}

TEST_F(I2CVMTest, SendInitialUAVO) {
  const uint32_t program[] = {
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(0, uavo_data.pc);
  EXPECT_EQ(0, uavo_data.r0);
  EXPECT_EQ(0, uavo_data.r1);
  EXPECT_EQ(0, uavo_data.r2);
  EXPECT_EQ(0, uavo_data.r3);
  EXPECT_EQ(0, uavo_data.r4);
  EXPECT_EQ(0, uavo_data.r5);
  EXPECT_EQ(0, uavo_data.r6);
}

TEST_F(I2CVMTest, RegisterIndependence) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 10),
    I2C_VM_ASM_SET_IMM(VM_R1, 11),
    I2C_VM_ASM_SET_IMM(VM_R2, 12),
    I2C_VM_ASM_SET_IMM(VM_R3, 13),
    I2C_VM_ASM_SET_IMM(VM_R4, 14),
    I2C_VM_ASM_SET_IMM(VM_R5, 15),
    I2C_VM_ASM_SET_IMM(VM_R6, 16),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(7,  uavo_data.pc);
  EXPECT_EQ(10, uavo_data.r0);
  EXPECT_EQ(11, uavo_data.r1);
  EXPECT_EQ(12, uavo_data.r2);
  EXPECT_EQ(13, uavo_data.r3);
  EXPECT_EQ(14, uavo_data.r4);
  EXPECT_EQ(15, uavo_data.r5);
  EXPECT_EQ(16, uavo_data.r6);
}

TEST_F(I2CVMTest, SetImmMax) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 32767),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(32767, uavo_data.r0);
}

TEST_F(I2CVMTest, SetImmSignExtension) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, -1),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(0xFFFFFFFFu, (uint32_t)uavo_data.r0);
}

TEST_F(I2CVMTest, SetImmMin) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, -32768),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(-32768, uavo_data.r0);
}

TEST_F(I2CVMTest, SetImmTruncatesAt16Bit) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 0x00010001),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(0x00000001, uavo_data.r0);
}

TEST_F(I2CVMTest, Store) {
  const uint32_t program[] = {
    I2C_VM_ASM_STORE(0x01, 0),
    I2C_VM_ASM_STORE(0x02, 1),
    I2C_VM_ASM_STORE(0x03, 2),
    I2C_VM_ASM_STORE(0x04, 3),
    I2C_VM_ASM_STORE(0x05, 4),
    I2C_VM_ASM_STORE(0x06, 5),
    I2C_VM_ASM_STORE(0x07, 6),
    I2C_VM_ASM_STORE(0x08, 7),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  const uint8_t ram[I2CVM_RAM_NUMELEMENTS] = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
  };

  EXPECT_EQ(0, memcmp(ram, uavo_data.ram, sizeof(ram)));
}

TEST_F(I2CVMTest, StoreBadAddress) {
  const uint32_t program[] = {
    I2C_VM_ASM_STORE(0x01, sizeof(uavo_data.ram)),
  };

  EXPECT_FALSE(i2c_vm_run (program, NELEMENTS(program), 0));
}

TEST_F(I2CVMTest, LoadEndianConversions) {
  const uint32_t program[] = {
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

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(0x00000B0Au, (uint32_t)uavo_data.r0);
  EXPECT_EQ(0x000C0B0Au, (uint32_t)uavo_data.r1);
  EXPECT_EQ(0x0D0C0B0Au, (uint32_t)uavo_data.r2);
  EXPECT_EQ(0x00000A0Bu, (uint32_t)uavo_data.r3);
  EXPECT_EQ(0x000A0B0Cu, (uint32_t)uavo_data.r4);
  EXPECT_EQ(0x0A0B0C0Du, (uint32_t)uavo_data.r5);
}

TEST_F(I2CVMTest, LoadBadAddress) {
  const uint32_t program[] = {
    I2C_VM_ASM_LOAD_LE(sizeof(uavo_data.ram), 2, VM_R0),
  };

  EXPECT_FALSE(i2c_vm_run (program, NELEMENTS(program), 0));
}

TEST_F(I2CVMTest, LoadBadLength) {
  const uint32_t program[] = {
    I2C_VM_ASM_LOAD_LE(0, sizeof(uavo_data.ram) + 1, VM_R0),
  };

  EXPECT_FALSE(i2c_vm_run (program, NELEMENTS(program), 0));
}

TEST_F(I2CVMTest, LSL_ASR_SignExtend) {
  const uint32_t program[] = {
    I2C_VM_ASM_STORE(0x80, 0),
    I2C_VM_ASM_STORE(0x81, 1),
    I2C_VM_ASM_STORE(0x82, 2),
    I2C_VM_ASM_STORE(0x83, 3),

    /* 8-bit negative */
    I2C_VM_ASM_LOAD_BE(0, 1, VM_R0),
    I2C_VM_ASM_SL_IMM(VM_R0, 24),
    I2C_VM_ASM_ASR_IMM(VM_R0, 24),

    /* 16-bit negative */
    I2C_VM_ASM_LOAD_BE(0, 2, VM_R1),
    I2C_VM_ASM_SL_IMM(VM_R1, 16),
    I2C_VM_ASM_ASR_IMM(VM_R1, 16),

    /* 24-bit negative */
    I2C_VM_ASM_LOAD_BE(0, 3, VM_R2),
    I2C_VM_ASM_SL_IMM(VM_R2, 8),
    I2C_VM_ASM_ASR_IMM(VM_R2, 8),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(0xFFFFFF80u, (uint32_t)uavo_data.r0);
  EXPECT_EQ(0xFFFF8081u, (uint32_t)uavo_data.r1);
  EXPECT_EQ(0xFF808182u, (uint32_t)uavo_data.r2);
}

TEST_F(I2CVMTest, LSL_ASR_ZeroExtend) {
  const uint32_t program[] = {
    I2C_VM_ASM_STORE(0x70, 0),
    I2C_VM_ASM_STORE(0x71, 1),
    I2C_VM_ASM_STORE(0x72, 2),
    I2C_VM_ASM_STORE(0x73, 3),

    /* 8-bit positive */
    I2C_VM_ASM_LOAD_BE(0, 1, VM_R0),
    I2C_VM_ASM_SL_IMM(VM_R0, 24),
    I2C_VM_ASM_ASR_IMM(VM_R0, 24),

    /* 16-bit positive */
    I2C_VM_ASM_LOAD_BE(0, 2, VM_R1),
    I2C_VM_ASM_SL_IMM(VM_R1, 16),
    I2C_VM_ASM_ASR_IMM(VM_R1, 16),

    /* 24-bit positive */
    I2C_VM_ASM_LOAD_BE(0, 3, VM_R2),
    I2C_VM_ASM_SL_IMM(VM_R2, 8),
    I2C_VM_ASM_ASR_IMM(VM_R2, 8),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(0x00000070u, (uint32_t)uavo_data.r0);
  EXPECT_EQ(0x00007071u, (uint32_t)uavo_data.r1);
  EXPECT_EQ(0x00707172u, (uint32_t)uavo_data.r2);
}

TEST_F(I2CVMTest, LogicalShiftLeft) {
  const uint32_t program[] = {
    /* Zero shifted left is still zero */
    I2C_VM_ASM_SET_IMM(VM_R0, 0x0000),
    I2C_VM_ASM_SL_IMM(VM_R0, 1),

    /* Any number shifted left by zero remains unchanged */
    I2C_VM_ASM_SET_IMM(VM_R1, 0x0100),
    I2C_VM_ASM_SL_IMM(VM_R1, 0),

    /* Simple shift left */
    I2C_VM_ASM_SET_IMM(VM_R2, 0x0001),
    I2C_VM_ASM_SL_IMM(VM_R2, 1),

    /* 16-bit positive number shifted left remains a positive 32-bit register value */
    I2C_VM_ASM_SET_IMM(VM_R3, 0x4000),
    I2C_VM_ASM_SL_IMM(VM_R3, 1),

    /* Sign extended negative number shifted left remains a negative 32-bit register value */
    I2C_VM_ASM_SET_IMM(VM_R4, 0xFFFF),
    I2C_VM_ASM_SL_IMM(VM_R4, 1),

    /*
     * Left shift by a negative value has an undefined result in the C spec.
     * This virtual machine treats all shifts as 5-bit unsigned values so
     * a left shift of -15 is 0x11 in two's complement which is a left shift
     * of 17 when treated as unsigned.
     */
    I2C_VM_ASM_SET_IMM(VM_R5, 0x0002),
    I2C_VM_ASM_SL_IMM(VM_R5, -15),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(0x00000000u, (uint32_t)uavo_data.r0);
  EXPECT_EQ(0x00000100u, (uint32_t)uavo_data.r1);
  EXPECT_EQ(0x00000002u, (uint32_t)uavo_data.r2);
  EXPECT_EQ(0x00008000u, (uint32_t)uavo_data.r3);
  EXPECT_EQ(0xFFFFFFFEu, (uint32_t)uavo_data.r4);
  EXPECT_EQ(0x00040000u, (uint32_t)uavo_data.r5);
}

TEST_F(I2CVMTest, LogicalShiftRight) {
  const uint32_t program[] = {
    /* Zero shifted right is still zero */
    I2C_VM_ASM_SET_IMM(VM_R0, 0x0000),
    I2C_VM_ASM_LSR_IMM(VM_R0, 1),

    /* Any number shifted right by zero remains unchanged */
    I2C_VM_ASM_SET_IMM(VM_R1, 0x0100),
    I2C_VM_ASM_LSR_IMM(VM_R1, 0),

    /* Simple shift right */
    I2C_VM_ASM_SET_IMM(VM_R2, 0x0010),
    I2C_VM_ASM_LSR_IMM(VM_R2, 1),

    /* 16-bit positive number shifted right remains a positive 32-bit register value */
    I2C_VM_ASM_SET_IMM(VM_R3, 0x4000),
    I2C_VM_ASM_LSR_IMM(VM_R3, 1),

    /* Sign extended negative number shifted right becomes a positive 32-bit register value */
    I2C_VM_ASM_SET_IMM(VM_R4, 0xFFFF),
    I2C_VM_ASM_LSR_IMM(VM_R4, 1),

    /*
     * Right shift by a negative value has an undefined result in the C spec.
     * This virtual machine treats all shifts as 5-bit unsigned values so
     * a right shift of -30 is 0x02 in two's complement which is a right shift
     * of 2 when treated as unsigned.
     */
    I2C_VM_ASM_SET_IMM(VM_R5, 0x0400),
    I2C_VM_ASM_LSR_IMM(VM_R5, -30),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(0x00000000u, (uint32_t)uavo_data.r0);
  EXPECT_EQ(0x00000100u, (uint32_t)uavo_data.r1);
  EXPECT_EQ(0x00000008u, (uint32_t)uavo_data.r2);
  EXPECT_EQ(0x00002000u, (uint32_t)uavo_data.r3);
  EXPECT_EQ(0x7FFFFFFFu, (uint32_t)uavo_data.r4);
  EXPECT_EQ(0x00000100u, (uint32_t)uavo_data.r5);
}

TEST_F(I2CVMTest, ArithmeticShiftRight) {
  const uint32_t program[] = {
    /* Zero shifted right is still zero */
    I2C_VM_ASM_SET_IMM(VM_R0, 0x0000),
    I2C_VM_ASM_ASR_IMM(VM_R0, 1),

    /* Any number shifted right by zero remains unchanged */
    I2C_VM_ASM_SET_IMM(VM_R1, 0x0100),
    I2C_VM_ASM_ASR_IMM(VM_R1, 0),

    /* Simple shift right */
    I2C_VM_ASM_SET_IMM(VM_R2, 0x0010),
    I2C_VM_ASM_ASR_IMM(VM_R2, 1),

    /* 16-bit positive number shifted right remains a positive 32-bit register value */
    I2C_VM_ASM_SET_IMM(VM_R3, 0x4000),
    I2C_VM_ASM_ASR_IMM(VM_R3, 1),

    /* Sign extended negative number shifted right remains a negative 32-bit register value */
    I2C_VM_ASM_SET_IMM(VM_R4, 0xFF00),
    I2C_VM_ASM_ASR_IMM(VM_R4, 1),

    /*
     * Right shift by a negative value has an undefined result in the C spec.
     * This virtual machine treats all shifts as 5-bit unsigned values so
     * a right shift of -30 is 0x02 in two's complement which is a right shift
     * of 2 when treated as unsigned.
     */
    I2C_VM_ASM_SET_IMM(VM_R5, 0x0400),
    I2C_VM_ASM_ASR_IMM(VM_R5, -30),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(0x00000000u, (uint32_t)uavo_data.r0);
  EXPECT_EQ(0x00000100u, (uint32_t)uavo_data.r1);
  EXPECT_EQ(0x00000008u, (uint32_t)uavo_data.r2);
  EXPECT_EQ(0x00002000u, (uint32_t)uavo_data.r3);
  EXPECT_EQ(0xFFFFFF80u, (uint32_t)uavo_data.r4);
  EXPECT_EQ(0x00000100u, (uint32_t)uavo_data.r5);
}

TEST_F(I2CVMTest, OrImm) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 0x000A),
    I2C_VM_ASM_SL_IMM(VM_R0, 8),
    I2C_VM_ASM_OR_IMM(VM_R0, 0x000B),
    I2C_VM_ASM_SL_IMM(VM_R0, 16),
    I2C_VM_ASM_OR_IMM(VM_R0, 0x0C0D),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(0x0A0B0C0Du, (uint32_t)uavo_data.r0);
}

TEST_F(I2CVMTest, And) {
  const uint32_t program[] = {
    /* R0 = 0xAAAA5555 */
    I2C_VM_ASM_SET_IMM(VM_R0, 0),
    I2C_VM_ASM_OR_IMM(VM_R0, 0xAAAA),
    I2C_VM_ASM_SL_IMM(VM_R0, 16),
    I2C_VM_ASM_OR_IMM(VM_R0, 0x5555),

    /* R1 = 0x5555AAAA */
    I2C_VM_ASM_SET_IMM(VM_R1, 0),
    I2C_VM_ASM_OR_IMM(VM_R1, 0x5555),
    I2C_VM_ASM_SL_IMM(VM_R1, 16),
    I2C_VM_ASM_OR_IMM(VM_R1, 0xAAAA),

    /* R2 = 0xA5A55A5A */
    I2C_VM_ASM_SET_IMM(VM_R2, 0),
    I2C_VM_ASM_OR_IMM(VM_R2, 0xA5A5),
    I2C_VM_ASM_SL_IMM(VM_R2, 16),
    I2C_VM_ASM_OR_IMM(VM_R2, 0x5A5A),

    /* R3 = R0 & R1 */
    I2C_VM_ASM_AND(VM_R3, VM_R0, VM_R1),

    /* R4 = R0 & R2 */
    I2C_VM_ASM_AND(VM_R4, VM_R0, VM_R2),

    /* R5 = R1 & R2 */
    I2C_VM_ASM_AND(VM_R5, VM_R1, VM_R2),

    /* R6 = R2 & R1 */
    I2C_VM_ASM_AND(VM_R6, VM_R2, VM_R1),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(0xAAAA5555u, (uint32_t)uavo_data.r0);
  EXPECT_EQ(0x5555AAAAu, (uint32_t)uavo_data.r1);
  EXPECT_EQ(0xA5A55A5Au, (uint32_t)uavo_data.r2);
  EXPECT_EQ(0x00000000u, (uint32_t)uavo_data.r3);
  EXPECT_EQ(0xA0A05050u, (uint32_t)uavo_data.r4);
  EXPECT_EQ(0x05050A0Au, (uint32_t)uavo_data.r5);
  EXPECT_EQ(0x05050A0Au, (uint32_t)uavo_data.r6);
}

TEST_F(I2CVMTest, AddImmPositive) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 1234),
    I2C_VM_ASM_ADD_IMM(VM_R0,  101),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(1335, uavo_data.r0);
}

TEST_F(I2CVMTest, AddImmNegative) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0,   101),
    I2C_VM_ASM_ADD_IMM(VM_R0, -1234),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(-1133, uavo_data.r0);
}

TEST_F(I2CVMTest, AddImmDoesNotOverflowAt16Bit) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 32767),
    I2C_VM_ASM_ADD_IMM(VM_R0,     1),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(32768, uavo_data.r0);
}

TEST_F(I2CVMTest, Add) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 1234),
    I2C_VM_ASM_SET_IMM(VM_R1,  101),
    I2C_VM_ASM_ADD(VM_R2, VM_R0, VM_R1),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(1335, uavo_data.r2);
}

TEST_F(I2CVMTest, MulPosPos) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 10233),
    I2C_VM_ASM_SET_IMM(VM_R1,   922),
    I2C_VM_ASM_MUL(VM_R2, VM_R0, VM_R1),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(9434826, uavo_data.r2);
}

TEST_F(I2CVMTest, MulPosNeg) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 10233),
    I2C_VM_ASM_SET_IMM(VM_R1,  -922),
    I2C_VM_ASM_MUL(VM_R2, VM_R0, VM_R1),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(-9434826, uavo_data.r2);
}

TEST_F(I2CVMTest, MulNegNeg) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, -10233),
    I2C_VM_ASM_SET_IMM(VM_R1,   -922),
    I2C_VM_ASM_MUL(VM_R2, VM_R0, VM_R1),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(9434826, uavo_data.r2);
}

TEST_F(I2CVMTest, MulImm) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 10233),
    I2C_VM_ASM_MUL_IMM(VM_R0,   922),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(9434826, uavo_data.r0);
}

TEST_F(I2CVMTest, DivPosPos) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 10233),
    I2C_VM_ASM_SET_IMM(VM_R1,    22),
    I2C_VM_ASM_DIV(VM_R2, VM_R0, VM_R1),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(465, uavo_data.r2);
}

TEST_F(I2CVMTest, DivPosNeg) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 10233),
    I2C_VM_ASM_SET_IMM(VM_R1,   -22),
    I2C_VM_ASM_DIV(VM_R2, VM_R0, VM_R1),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(-465, uavo_data.r2);
}

TEST_F(I2CVMTest, DivNegNeg) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, -10233),
    I2C_VM_ASM_SET_IMM(VM_R1,    -22),
    I2C_VM_ASM_DIV(VM_R2, VM_R0, VM_R1),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(465, uavo_data.r2);
}

TEST_F(I2CVMTest, DivImm) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 10233),
    I2C_VM_ASM_DIV_IMM(VM_R0,    22),
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(465, uavo_data.r0);
}

TEST_F(I2CVMTest, JumpForward) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 1),
    I2C_VM_ASM_JUMP(2),

    /* Never reached */
    I2C_VM_ASM_SET_IMM(VM_R0, 2),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(1, uavo_data.r0);
}

TEST_F(I2CVMTest, JumpBackward) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 1),

    /* Skip forward over the next block */
    I2C_VM_ASM_JUMP(3),

    I2C_VM_ASM_SET_IMM(VM_R0, 2),
    I2C_VM_ASM_JUMP(2),

    /* Jump back into the previous block */
    I2C_VM_ASM_JUMP(-2),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(2, uavo_data.r0);
}

TEST_F(I2CVMTest, BNZNotTaken) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 1),
    I2C_VM_ASM_SET_IMM(VM_R1, 0),

    /* Skip forward over the next block */
    I2C_VM_ASM_JUMP(3),

    I2C_VM_ASM_SET_IMM(VM_R0, 2),
    I2C_VM_ASM_JUMP(2),

    /* Jump back into the previous block IFF R1 != 0 */
    I2C_VM_ASM_BNZ(VM_R1, -2),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(1, uavo_data.r0);
}

TEST_F(I2CVMTest, BNZTaken) {
  const uint32_t program[] = {
    I2C_VM_ASM_SET_IMM(VM_R0, 1),
    I2C_VM_ASM_SET_IMM(VM_R1, 1),

    /* Skip forward over the next block */
    I2C_VM_ASM_JUMP(3),

    I2C_VM_ASM_SET_IMM(VM_R0, 2),
    I2C_VM_ASM_JUMP(2),

    /* Jump back into the previous block IFF R1 != 0 */
    I2C_VM_ASM_BNZ(VM_R1, -2),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(2, uavo_data.r0);
}

TEST_F(I2CVMTest, CleanReboot) {
  /* Run a program to scribble all over the machine state */
  const uint32_t program[] = {
    I2C_VM_ASM_STORE(0x01, 0),
    I2C_VM_ASM_STORE(0x02, 1),
    I2C_VM_ASM_STORE(0x03, 2),
    I2C_VM_ASM_STORE(0x04, 3),
    I2C_VM_ASM_STORE(0x05, 4),
    I2C_VM_ASM_STORE(0x06, 5),
    I2C_VM_ASM_STORE(0x07, 6),
    I2C_VM_ASM_STORE(0x08, 7),

    I2C_VM_ASM_SET_IMM(VM_R0, 10),
    I2C_VM_ASM_SET_IMM(VM_R1, 11),
    I2C_VM_ASM_SET_IMM(VM_R2, 12),
    I2C_VM_ASM_SET_IMM(VM_R3, 13),
    I2C_VM_ASM_SET_IMM(VM_R4, 14),
    I2C_VM_ASM_SET_IMM(VM_R5, 15),
    I2C_VM_ASM_SET_IMM(VM_R6, 16),

    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program, NELEMENTS(program), 0));

  EXPECT_EQ(15, uavo_data.pc);
  EXPECT_EQ(10, uavo_data.r0);
  EXPECT_EQ(11, uavo_data.r1);
  EXPECT_EQ(12, uavo_data.r2);
  EXPECT_EQ(13, uavo_data.r3);
  EXPECT_EQ(14, uavo_data.r4);
  EXPECT_EQ(15, uavo_data.r5);
  EXPECT_EQ(16, uavo_data.r6);

  const uint8_t ram[I2CVM_RAM_NUMELEMENTS] = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
  };

  EXPECT_EQ(0, memcmp(ram, uavo_data.ram, sizeof(ram)));

  /* Run a new program and expect that the machine state has been entirely reset */

  const uint32_t program2[] = {
    I2C_VM_ASM_SEND_UAVO(),
  };

  EXPECT_TRUE(i2c_vm_run (program2, NELEMENTS(program2), 0));

  EXPECT_EQ(0, uavo_data.pc);
  EXPECT_EQ(0, uavo_data.r0);
  EXPECT_EQ(0, uavo_data.r1);
  EXPECT_EQ(0, uavo_data.r2);
  EXPECT_EQ(0, uavo_data.r3);
  EXPECT_EQ(0, uavo_data.r4);
  EXPECT_EQ(0, uavo_data.r5);
  EXPECT_EQ(0, uavo_data.r6);

  const uint8_t ram2[I2CVM_RAM_NUMELEMENTS] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };

  EXPECT_EQ(0, memcmp(ram2, uavo_data.ram, sizeof(ram)));
}
