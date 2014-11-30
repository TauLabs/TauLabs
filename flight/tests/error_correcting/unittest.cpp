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

#include <ecc.h>

}

#include <math.h>   /* fabs() */



// To use a test fixture, derive a class from testing::Test.
class EncodeDecode : public testing::Test {
protected:
  virtual void SetUp() {
    initialize_ecc();
  }

  virtual void TearDown() {
  }
};

TEST_F(EncodeDecode, EmptyEncode) {
  char p[4] = {};
  encode_data((unsigned char *)p, 0, (unsigned char *)p);
  EXPECT_EQ(0, p[0]);
  EXPECT_EQ(0, p[1]);
  EXPECT_EQ(0, p[2]);
  EXPECT_EQ(0, p[3]);
};

TEST_F(EncodeDecode, CorrectEncode) {
  unsigned char p[10] = {'a', 'b', 'c', 'd', 'e', 'f'};
  encode_data(p, 6, p);
  //fprintf(stdout, "%d %d %d %d\n", p[6], p[7], p[8], p[9]);
  EXPECT_EQ(0x1f, p[6]);
  EXPECT_EQ(0xa3, p[7]);
  EXPECT_EQ(0x9a, p[8]);
  EXPECT_EQ(0x3b, p[9]);
};

TEST_F(EncodeDecode, PassEncode) {
  unsigned char p[10] = {'a', 'b', 'c', 'd', 'e', 'f'};
  encode_data(p, 6, p);
  decode_data(p, 6 + RS_ECC_NPARITY);
  EXPECT_EQ(0, check_syndrome());
};

TEST_F(EncodeDecode, Recover) {
  unsigned char p[10] = {'a', 'b', 'c', 'd', 'e', 'f'};
  encode_data(p, 6, p);
  unsigned char p2[10];
  for (int i = 0; i < 10; i++)
    p2[i] = p[i];
  p2[4] = 30;

  // verify it flags the error
  decode_data(p2, 6 + RS_ECC_NPARITY);
  EXPECT_EQ(1, check_syndrome());

  // verify it is corrected
  EXPECT_EQ(1, correct_errors_erasures(p2, 10, 0, 0));

  for (int i = 0; i < 6; i++)
    EXPECT_EQ(p[i], p2[i]);

};
