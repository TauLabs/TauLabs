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
#define TEST_ARRAY_LENGTH_LARGE 100

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
  static const float test_array3[TEST_ARRAY_LENGTH_LARGE];

  // Test structures. These are unique to each instance
  struct linear_mean_and_std_dev test_struct_0;
  struct linear_mean_and_std_dev test_struct_1;
  struct linear_mean_and_std_dev test_struct_2;
  struct linear_mean_and_std_dev test_struct_3;

  static const double eps;
  static const float newest_y;
};

// Initialize static constants
const double LinearStatistics::eps = 1e-6;
const float LinearStatistics::newest_y = 6;
const float LinearStatistics::test_array0[] = {0, 1, 2, 3, 4, 5};
const float LinearStatistics::test_array1[] = {-2, -1, 0, 0, 1, 2};
const float LinearStatistics::test_array2[] = {0, 0, 0, 0, 0, 0};
const float LinearStatistics::test_array3[] = { // Generated with MATLAB: `rng default; rand(10,10)`
   0.8147, 0.1576, 0.6557, 0.7060, 0.4387, 0.2760, 0.7513, 0.8407, 0.3517, 0.0759,
   0.9058, 0.9706, 0.0357, 0.0318, 0.3816, 0.6797, 0.2551, 0.2543, 0.8308, 0.0540,
   0.1270, 0.9572, 0.8491, 0.2769, 0.7655, 0.6551, 0.5060, 0.8143, 0.5853, 0.5308,
   0.9134, 0.4854, 0.9340, 0.0462, 0.7952, 0.1626, 0.6991, 0.2435, 0.5497, 0.7792,
   0.6324, 0.8003, 0.6787, 0.0971, 0.1869, 0.1190, 0.8909, 0.9293, 0.9172, 0.9340,
   0.0975, 0.1419, 0.7577, 0.8235, 0.4898, 0.4984, 0.9593, 0.3500, 0.2858, 0.1299,
   0.2785, 0.4218, 0.7431, 0.6948, 0.4456, 0.9597, 0.5472, 0.1966, 0.7572, 0.5688,
   0.5469, 0.9157, 0.3922, 0.3171, 0.6463, 0.3404, 0.1386, 0.2511, 0.7537, 0.4694,
   0.9575, 0.7922, 0.6555, 0.9502, 0.7094, 0.5853, 0.1493, 0.6160, 0.3804, 0.0119,
   0.9649, 0.9595, 0.1712, 0.0344, 0.7547, 0.2238, 0.2575, 0.4733, 0.5678, 0.3371};

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

  initialize_linear_sums(&test_struct_3,
                         TEST_ARRAY_LENGTH_LARGE,
                         TEST_ARRAY_LENGTH_LARGE,
                         test_array3);
  EXPECT_EQ(TEST_ARRAY_LENGTH_LARGE, test_struct_3.T0);
  EXPECT_NEAR(52.7994, test_struct_3.T1, eps);
  EXPECT_NEAR(36.61145552, test_struct_3.T2, eps);
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

  initialize_linear_sums(&test_struct_3,
                         TEST_ARRAY_LENGTH_LARGE,
                         TEST_ARRAY_LENGTH_LARGE,
                         test_array3);
  incremental_update_linear_sums(&test_struct_3,
                                 test_array3[0],
                                 newest_y);
  EXPECT_EQ(TEST_ARRAY_LENGTH_LARGE, test_struct_3.T0);
  EXPECT_NEAR(57.9847, test_struct_3.T1, eps);
  EXPECT_NEAR(71.94771943, test_struct_3.T2, eps);
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

  initialize_linear_sums(&test_struct_3,
                         TEST_ARRAY_LENGTH_LARGE,
                         TEST_ARRAY_LENGTH_LARGE-2,
                         test_array3);
  incremental_update_linear_sums(&test_struct_3,
                                 test_array3[0],
                                 test_array3[TEST_ARRAY_LENGTH_LARGE - 2]);
  EXPECT_EQ(TEST_ARRAY_LENGTH_LARGE - 1, test_struct_3.T0);
  EXPECT_NEAR(52.4623, test_struct_3.T1, eps);
  EXPECT_NEAR(36.49781911, test_struct_3.T2, eps);
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
    initialize_linear_sums(&test_struct_3,
                           TEST_ARRAY_LENGTH_LARGE,
                           TEST_ARRAY_LENGTH_LARGE,
                           test_array3);
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

  float mean3 = get_linear_mean(&test_struct_3);
  EXPECT_NEAR(0.527994466938442, mean3, eps);
};


TEST_F(CalculateLinearUpdate, StandardDeviation) {
  float std_dev0 = get_linear_standard_deviation(&test_struct_0);
  EXPECT_NEAR(1.70782512765993, std_dev0, eps);

  float std_dev1 = get_linear_standard_deviation(&test_struct_1);
  EXPECT_NEAR(1.29099444873581, std_dev1, eps);

  float std_dev2 = get_linear_standard_deviation(&test_struct_2);
  EXPECT_NEAR(0, std_dev2, eps);

  float std_dev3 = get_linear_standard_deviation(&test_struct_3);
  EXPECT_NEAR(0.295528156296486, std_dev3, eps);
};


//--------------
// To use a test fixture, derive a class from testing::Test.
class CircularStatistics : public testing::Test {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }

  // Test arrays. These are global to all instances
  static const float test_array0[TEST_ARRAY_LENGTH_SMALL];
  static const float test_array1[TEST_ARRAY_LENGTH_SMALL];
  static const float test_array2[TEST_ARRAY_LENGTH_SMALL];
  static const float test_array3[TEST_ARRAY_LENGTH_LARGE];

  // Test structures. These are unique to each instance
  struct circular_mean_and_std_dev test_struct_0;
  struct circular_mean_and_std_dev test_struct_1;
  struct circular_mean_and_std_dev test_struct_2;
  struct circular_mean_and_std_dev test_struct_3;

  static const double eps;
  static const float newest_y;
  static const double pi;
};

// Initialize static constants
const double CircularStatistics::eps = 1e-6;
const float CircularStatistics::newest_y = 6;
const double CircularStatistics::pi = 3.14159265358979323846264338327950288;
const float CircularStatistics::test_array0[] = {0, 1, 2, 3, 4, 5};
const float CircularStatistics::test_array1[] = {-2*pi, -1*pi, 0*pi, 0*pi, 1*pi, 2*pi};
const float CircularStatistics::test_array2[] = {0, 0, 0, 0, 0, 0};
const float CircularStatistics::test_array3[] = { // Generated with MATLAB: `rng default; rand(10,10)`
   0.8147, 0.1576, 0.6557, 0.7060, 0.4387, 0.2760, 0.7513, 0.8407, 0.3517, 0.0759,
   0.9058, 0.9706, 0.0357, 0.0318, 0.3816, 0.6797, 0.2551, 0.2543, 0.8308, 0.0540,
   0.1270, 0.9572, 0.8491, 0.2769, 0.7655, 0.6551, 0.5060, 0.8143, 0.5853, 0.5308,
   0.9134, 0.4854, 0.9340, 0.0462, 0.7952, 0.1626, 0.6991, 0.2435, 0.5497, 0.7792,
   0.6324, 0.8003, 0.6787, 0.0971, 0.1869, 0.1190, 0.8909, 0.9293, 0.9172, 0.9340,
   0.0975, 0.1419, 0.7577, 0.8235, 0.4898, 0.4984, 0.9593, 0.3500, 0.2858, 0.1299,
   0.2785, 0.4218, 0.7431, 0.6948, 0.4456, 0.9597, 0.5472, 0.1966, 0.7572, 0.5688,
   0.5469, 0.9157, 0.3922, 0.3171, 0.6463, 0.3404, 0.1386, 0.2511, 0.7537, 0.4694,
   0.9575, 0.7922, 0.6555, 0.9502, 0.7094, 0.5853, 0.1493, 0.6160, 0.3804, 0.0119,
   0.9649, 0.9595, 0.1712, 0.0344, 0.7547, 0.2238, 0.2575, 0.4733, 0.5678, 0.3371};

// Test fixture for incremental_update_linear_sums()
class Initialize : public CircularStatistics {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }

};


TEST_F(Initialize, InitializeMeanAndStandardDeviation) {
  initialize_circular_sums(&test_struct_0,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL,
                           test_array0);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_0.T0);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL*TEST_ARRAY_LENGTH_SMALL, (int)test_struct_0.T0_2);
  EXPECT_NEAR(0.176161649722379, test_struct_0.S1, eps);
  EXPECT_NEAR(-0.235818462679834, test_struct_0.C1, eps);

  initialize_circular_sums(&test_struct_1,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL,
                           test_array1);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_1.T0);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL*TEST_ARRAY_LENGTH_SMALL, (int)test_struct_1.T0_2);
  EXPECT_NEAR(0, test_struct_1.S1, eps);
  EXPECT_NEAR(2, test_struct_1.C1, eps);

  initialize_circular_sums(&test_struct_2,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL,
                           test_array2);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_2.T0);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL*TEST_ARRAY_LENGTH_SMALL, (int)test_struct_2.T0_2);
  EXPECT_NEAR(0, test_struct_2.S1, eps);
  EXPECT_NEAR(6, test_struct_2.C1, eps);

  initialize_circular_sums(&test_struct_3,
                           TEST_ARRAY_LENGTH_LARGE,
                           TEST_ARRAY_LENGTH_LARGE,
                           test_array3);
  EXPECT_EQ(TEST_ARRAY_LENGTH_LARGE, test_struct_3.T0);
  EXPECT_EQ(TEST_ARRAY_LENGTH_LARGE*TEST_ARRAY_LENGTH_LARGE, (int)test_struct_3.T0_2);
  EXPECT_NEAR(48.2528200811188, test_struct_3.S1, eps);
  EXPECT_NEAR(82.6300334798971, test_struct_3.C1, eps);
};


// Test fixture for incremental_update_circular_sums()
class IncrementalCircularUpdate : public CircularStatistics {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};


TEST_F(IncrementalCircularUpdate, IncrementalCircularUpdateMeanAndStandardDeviation) {
  initialize_circular_sums(&test_struct_0,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL,
                           test_array0);
  incremental_update_circular_sums(&test_struct_0,
                                   test_array0[0],
                                   newest_y);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_0.T0);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL*TEST_ARRAY_LENGTH_SMALL, (int)test_struct_0.T0_2);
  EXPECT_NEAR(-0.103253848476547, test_struct_0.S1, eps);
  EXPECT_NEAR(-0.275648176029468, test_struct_0.C1, eps);

  initialize_circular_sums(&test_struct_1,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL,
                           test_array1);
  incremental_update_circular_sums(&test_struct_1,
                                   test_array1[0],
                                   newest_y);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_1.T0);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL*TEST_ARRAY_LENGTH_SMALL, (int)test_struct_1.T0_2);
  EXPECT_NEAR(-0.279415498198926, test_struct_1.S1, eps);
  EXPECT_NEAR(1.96017028665037, test_struct_1.C1, eps);

  initialize_circular_sums(&test_struct_2,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL,
                           test_array2);
  incremental_update_circular_sums(&test_struct_2,
                                   test_array2[0],
                                   newest_y);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL, test_struct_2.T0);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL*TEST_ARRAY_LENGTH_SMALL, (int)test_struct_2.T0_2);
  EXPECT_NEAR(-0.279415498198926, test_struct_2.S1, eps);
  EXPECT_NEAR(5.96017028665037, test_struct_2.C1, eps);

  initialize_circular_sums(&test_struct_3,
                           TEST_ARRAY_LENGTH_LARGE,
                           TEST_ARRAY_LENGTH_LARGE,
                           test_array3);
  incremental_update_circular_sums(&test_struct_3,
                                   test_array3[0],
                                   newest_y);
  EXPECT_EQ(TEST_ARRAY_LENGTH_LARGE, test_struct_3.T0);
  EXPECT_EQ(TEST_ARRAY_LENGTH_LARGE*TEST_ARRAY_LENGTH_LARGE, (int)test_struct_3.T0_2);
  EXPECT_NEAR(47.2458847775829, test_struct_3.S1, eps);
  EXPECT_NEAR(82.9041170862785, test_struct_3.C1, eps);

};


/**
 * @brief TEST_F This tests that a structure with a partially full data
 * array grows correctly
 */
TEST_F(IncrementalCircularUpdate, IncrementallyGrowMeanAndStandardDeviation) {
  initialize_circular_sums(&test_struct_0,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL - 2,
                           test_array0);
  incremental_update_circular_sums(&test_struct_0,
                                   test_array0[0],
                                   test_array0[TEST_ARRAY_LENGTH_SMALL - 2]);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL - 1, test_struct_0.T0);
  EXPECT_EQ((TEST_ARRAY_LENGTH_SMALL-1) * (TEST_ARRAY_LENGTH_SMALL-1), (int)test_struct_0.T0_2);
  EXPECT_NEAR(1.13508592438552, test_struct_0.S1, eps);
  EXPECT_NEAR(-0.51948064814306, test_struct_0.C1, eps);

  initialize_circular_sums(&test_struct_1,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL-2,
                           test_array1);
  incremental_update_circular_sums(&test_struct_1,
                                   test_array1[0],
                                   test_array1[TEST_ARRAY_LENGTH_SMALL - 2]);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL - 1, test_struct_1.T0);
  EXPECT_EQ((TEST_ARRAY_LENGTH_SMALL-1) * (TEST_ARRAY_LENGTH_SMALL-1), (int)test_struct_1.T0_2);
  EXPECT_NEAR(0, test_struct_1.S1, eps);
  EXPECT_NEAR(1, test_struct_1.C1, eps);

  initialize_circular_sums(&test_struct_2,
                           TEST_ARRAY_LENGTH_SMALL,
                           TEST_ARRAY_LENGTH_SMALL-2,
                           test_array2);
  incremental_update_circular_sums(&test_struct_2,
                                   test_array2[0],
                                   test_array2[TEST_ARRAY_LENGTH_SMALL - 2]);
  EXPECT_EQ(TEST_ARRAY_LENGTH_SMALL - 1, test_struct_2.T0);
  EXPECT_EQ((TEST_ARRAY_LENGTH_SMALL-1) * (TEST_ARRAY_LENGTH_SMALL-1), (int)test_struct_2.T0_2);
  EXPECT_NEAR(0, test_struct_2.S1, eps);
  EXPECT_NEAR(5, test_struct_2.C1, eps);

  initialize_circular_sums(&test_struct_3,
                           TEST_ARRAY_LENGTH_LARGE,
                           TEST_ARRAY_LENGTH_LARGE-2,
                           test_array3);
  incremental_update_circular_sums(&test_struct_3,
                                   test_array3[0],
                                   test_array3[TEST_ARRAY_LENGTH_LARGE - 2]);
  EXPECT_EQ(TEST_ARRAY_LENGTH_LARGE - 1, test_struct_3.T0);
  EXPECT_EQ((TEST_ARRAY_LENGTH_LARGE-1) * (TEST_ARRAY_LENGTH_LARGE-1), (int)test_struct_3.T0_2);
  EXPECT_NEAR(47.9220683759881, test_struct_3.S1, eps);
  EXPECT_NEAR(81.6863156674377, test_struct_3.C1, eps);
};


// Test fixture for incremental_update_circular_sums()
class CalculateCircularStatistics : public CircularStatistics {
protected:
  virtual void SetUp() {
    initialize_circular_sums(&test_struct_0,
                             TEST_ARRAY_LENGTH_SMALL,
                             TEST_ARRAY_LENGTH_SMALL,
                             test_array0);
    initialize_circular_sums(&test_struct_1,
                             TEST_ARRAY_LENGTH_SMALL,
                             TEST_ARRAY_LENGTH_SMALL,
                             test_array1);
    initialize_circular_sums(&test_struct_2,
                             TEST_ARRAY_LENGTH_SMALL,
                             TEST_ARRAY_LENGTH_SMALL,
                             test_array2);
    initialize_circular_sums(&test_struct_3,
                             TEST_ARRAY_LENGTH_LARGE,
                             TEST_ARRAY_LENGTH_LARGE,
                             test_array3);
   }

  virtual void TearDown() {
  }
};

TEST_F(CalculateCircularStatistics, Mean) {
  float mean0 = get_circular_mean(&test_struct_0);
  EXPECT_NEAR(2.5, mean0, eps);

  float mean1 = get_circular_mean(&test_struct_1);
  EXPECT_NEAR(0, mean1, eps);

  float mean2 = get_circular_mean(&test_struct_2);
  EXPECT_NEAR(0, mean2, eps);

  float mean3 = get_circular_mean(&test_struct_3);
  EXPECT_NEAR(0.528543561815535, mean3, eps);
};


TEST_F(CalculateCircularStatistics, StandardDeviation) {
  float std_dev0 = get_circular_standard_deviation(&test_struct_0);
  EXPECT_NEAR(2.45549889531754, std_dev0, eps);

  float std_dev1 = get_circular_standard_deviation(&test_struct_1);
  EXPECT_NEAR(1.48230380736751, std_dev1, eps);

  float std_dev2 = get_circular_standard_deviation(&test_struct_2);
  EXPECT_NEAR(0, std_dev2, eps);

  float std_dev3 = get_circular_standard_deviation(&test_struct_3);
  EXPECT_NEAR(0.296933421670676, std_dev3, eps);
};

TEST_F(CalculateCircularStatistics, AngularDeviation) {
  float angular_deviation0 = get_angular_deviation(&test_struct_0);
  EXPECT_NEAR(1.3790875853232, angular_deviation0, eps);

  float angular_deviation1 = get_angular_deviation(&test_struct_1);
  EXPECT_NEAR(1.15470053837925, angular_deviation1, eps);

  float angular_deviation2 = get_angular_deviation(&test_struct_2);
  EXPECT_NEAR(0, angular_deviation2, eps);

  float angular_deviation3 = get_angular_deviation(&test_struct_3);
  EXPECT_NEAR(0.293690722768234, angular_deviation3, eps);
};
