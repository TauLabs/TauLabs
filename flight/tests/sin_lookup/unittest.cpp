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

#include "sin_lookup.h"		/* API for sin_lookup functions */

}

#include <math.h>		/* sinf/cosf/tanf */

// To use a test fixture, derive a class from testing::Test.
class SinLookup : public testing::Test {
protected:
  virtual void SetUp() {
    EXPECT_EQ(0, sin_lookup_initialize());
  }

  virtual void TearDown() {
  }

  float deg2rad(float deg)
  {
    return (deg * M_PI / 180);
  }
};

TEST_F(SinLookup, SinDegSweep) {
  float eps = 0.02f;

  for (float x = 0; x <= 720.0; x += .001) {
    ASSERT_NEAR(sinf(deg2rad(x)), sin_lookup_deg(x), eps);
  }
}

TEST_F(SinLookup, SinRadSweep) {
  float eps = 0.02f;

  for (float x = 0; x <= 4 * M_PI; x += .0001) {
    ASSERT_NEAR(sinf(x), sin_lookup_rad(x), eps);
  }
}

TEST_F(SinLookup, CosDegSweep) {
  float eps = 0.02f;

  for (float x = 0; x <= 720.0; x += .001) {
    ASSERT_NEAR(cosf(deg2rad(x)), cos_lookup_deg(x), eps);
  }
}

TEST_F(SinLookup, CosRadSweep) {
  float eps = 0.02f;

  for (float x = 0; x <= 4 * M_PI; x += .0001) {
    ASSERT_NEAR(cosf(x), cos_lookup_rad(x), eps);
  }
}
