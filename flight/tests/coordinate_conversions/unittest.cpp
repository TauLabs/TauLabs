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

#include "coordinate_conversions.h" /* API for coordinate_conversions functions */

}

#include <math.h>		/* fabs() */

// To use a test fixture, derive a class from testing::Test.
class CoordConversion : public testing::Test {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

// Test fixture for bound_min_max()
class RneFromLLATest : public CoordConversion {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(RneFromLLATest, Equator) {
  float LLA[] = { 0, 0, 0 };
  float Rne[3][3];

  RneFromLLA(LLA, Rne);

  float eps = 0.000001f;
  ASSERT_NEAR(0, Rne[0][0], eps);
  ASSERT_NEAR(0, Rne[0][1], eps);
  ASSERT_NEAR(1, Rne[0][2], eps);

  ASSERT_NEAR(0, Rne[1][0], eps);
  ASSERT_NEAR(1, Rne[1][1], eps);
  ASSERT_NEAR(0, Rne[1][2], eps);

  ASSERT_NEAR(-1, Rne[2][0], eps);
  ASSERT_NEAR(0, Rne[2][1], eps);
  ASSERT_NEAR(0, Rne[2][2], eps);
};
