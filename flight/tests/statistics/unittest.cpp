/**
 ******************************************************************************
 * @file       unittest.cpp
 * @author     Kenn Sebesta, Copyright (C) 2016
 * @addtogroup UnitTests
 * @{
 * @addtogroup UnitTests
 * @{
 * @brief Unit test
 *****************************************************************************/

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

#include "statistics.h"		/* API for statistics functions */

}

#include <math.h>		/* fabs() */

#define TEST_ARRAY_LENGTH_SMALL 6

// To use a test fixture, derive a class from testing::Test.
class LinearStatistics : public testing::Test {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }

  // Test arrays. These are global to all instances
  static const float test_array0[TEST_ARRAY_LENGTH_SMALL];
  static const float test_array1[TEST_ARRAY_LENGTH_SMALL];
  static const float test_array2[TEST_ARRAY_LENGTH_SMALL];

  // Test structures. These are unique to each instance
  struct linear_mean_and_std_dev test_struct_0;
  struct linear_mean_and_std_dev test_struct_1;
  struct linear_mean_and_std_dev test_struct_2;

  static const double eps;
  static const float newest_y;
};

// Initialize static constants
const double LinearStatistics::eps = 1e-6;
const float LinearStatistics::newest_y = 6;
const float LinearStatistics::test_array0[] = {0, 1, 2, 3, 4, 5};
const float LinearStatistics::test_array1[] = {-2, -1, 0, 0, 1, 2};
const float LinearStatistics::test_array2[] = {0, 0, 0, 0, 0, 0};

// Test fixture for incremental_update_linear_sums()
class InitializeLinear : public LinearStatistics {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }

};


/**
 * @brief TEST_F Test that we are properly initializing the structures
 */
TEST_F(InitializeLinear, InitializeMeanAndStandardDeviation) {
  initialize_linear_sums(&test_struct_0,
                         TEST_ARRAY_LENGTH_SMALL,
                         TEST_ARRAY_LENGTH_SMALL,
                         test_array0);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_0.T0);
  EXPECT_NEAR(15, test_struct_0.T1, eps);
  EXPECT_NEAR(55, test_struct_0.T2, eps);

  initialize_linear_sums(&test_struct_1,
                         TEST_ARRAY_LENGTH_SMALL,
                         TEST_ARRAY_LENGTH_SMALL,
                         test_array1);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_1.T0);
  EXPECT_NEAR(0, test_struct_1.T1, eps);
  EXPECT_NEAR(10, test_struct_1.T2, eps);

  initialize_linear_sums(&test_struct_2,
                         TEST_ARRAY_LENGTH_SMALL,
                         TEST_ARRAY_LENGTH_SMALL,
                         test_array2);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_2.T0);
  EXPECT_NEAR(0, test_struct_2.T1, eps);
  EXPECT_NEAR(0, test_struct_2.T2, eps);
};


// Test fixture for incremental_update_linear_sums()
class IncrementalLinearUpdate : public LinearStatistics {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};


TEST_F(IncrementalLinearUpdate, IncrementalLinearUpdateMeanAndStandardDeviation) {
  initialize_linear_sums(&test_struct_0,
                         TEST_ARRAY_LENGTH_SMALL,
                         TEST_ARRAY_LENGTH_SMALL,
                         test_array0);
  incremental_update_linear_sums(&test_struct_0,
                                 test_array0[0],
                                 newest_y);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_0.T0);
  EXPECT_NEAR(21, test_struct_0.T1, eps);
  EXPECT_NEAR(91, test_struct_0.T2, eps);

  initialize_linear_sums(&test_struct_1,
                         TEST_ARRAY_LENGTH_SMALL,
                         TEST_ARRAY_LENGTH_SMALL,
                         test_array1);
  incremental_update_linear_sums(&test_struct_1,
                                 test_array1[0],
                                 newest_y);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_1.T0);
  EXPECT_NEAR(8, test_struct_1.T1, eps);
  EXPECT_NEAR(42, test_struct_1.T2, eps);

  initialize_linear_sums(&test_struct_2,
                         TEST_ARRAY_LENGTH_SMALL,
                         TEST_ARRAY_LENGTH_SMALL,
                         test_array2);
  incremental_update_linear_sums(&test_struct_2,
                                 test_array2[0],
                                 newest_y);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_2.T0);
  EXPECT_NEAR(6, test_struct_2.T1, eps);
  EXPECT_NEAR(36, test_struct_2.T2, eps);
};


/**
 * @brief TEST_F This tests that a structure with a partially full data
 * array grows correctly
 */
TEST_F(IncrementalLinearUpdate, IncrementallyGrowMeanAndStandardDeviation) {
  initialize_linear_sums(&test_struct_0,
                         TEST_ARRAY_LENGTH_SMALL,
                         TEST_ARRAY_LENGTH_SMALL - 2,
                         test_array0);
  incremental_update_linear_sums(&test_struct_0,
                                 test_array0[0],
                                 test_array0[TEST_ARRAY_LENGTH_SMALL - 2]);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL - 1, test_struct_0.T0);
  EXPECT_NEAR(10, test_struct_0.T1, eps);
  EXPECT_NEAR(30, test_struct_0.T2, eps);

  initialize_linear_sums(&test_struct_1,
                         TEST_ARRAY_LENGTH_SMALL,
                         TEST_ARRAY_LENGTH_SMALL-2,
                         test_array1);
  incremental_update_linear_sums(&test_struct_1,
                                 test_array1[0],
                                 test_array1[TEST_ARRAY_LENGTH_SMALL - 2]);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL - 1, test_struct_1.T0);
  EXPECT_NEAR(-2, test_struct_1.T1, eps);
  EXPECT_NEAR(6, test_struct_1.T2, eps);

  initialize_linear_sums(&test_struct_2,
                         TEST_ARRAY_LENGTH_SMALL,
                         TEST_ARRAY_LENGTH_SMALL-2,
                         test_array2);
  incremental_update_linear_sums(&test_struct_2,
                                 test_array2[0],
                                 test_array2[TEST_ARRAY_LENGTH_SMALL - 2]);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL - 1, test_struct_2.T0);
  EXPECT_NEAR(0, test_struct_2.T1, eps);
  EXPECT_NEAR(0, test_struct_2.T2, eps);
};


// Test fixture for incremental_update_linear_sums()
class CalculateLinearUpdate : public LinearStatistics {
protected:
  virtual void SetUp() {
    initialize_linear_sums(&test_struct_0,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL,
                           test_array0);
    initialize_linear_sums(&test_struct_1,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL,
                           test_array1);
    initialize_linear_sums(&test_struct_2,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL,
                           test_array2);
   }

  virtual void TearDown() {
  }
};

TEST_F(CalculateLinearUpdate, Mean) {
  float mean0 = get_linear_mean(&test_struct_0);
  EXPECT_NEAR(2.5, mean0, eps);

  float mean1 = get_linear_mean(&test_struct_1);
  EXPECT_NEAR(0, mean1, eps);

  float mean2 = get_linear_mean(&test_struct_2);
  EXPECT_NEAR(0, mean2, eps);
};


TEST_F(CalculateLinearUpdate, StandardDeviation) {
  float std_dev0 = get_linear_standard_deviation(&test_struct_0);
  EXPECT_NEAR(1.70782512765993, std_dev0, eps);

  float std_dev1 = get_linear_standard_deviation(&test_struct_1);
  EXPECT_NEAR(1.29099444873581, std_dev1, eps);

  float std_dev2 = get_linear_standard_deviation(&test_struct_2);
  EXPECT_NEAR(0, std_dev2, eps);
};

