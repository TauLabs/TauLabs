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
#include "physical_constants.h"

#include <stdio.h>		/* printf */
#include <stdlib.h>		/* abort */
#include <string.h>		/* memset */
#include <stdint.h>		/* uint*_t */

extern "C" {

#include "misc_math.h"		/* API for misc_math functions */

}

#include <math.h>		/* fabs() */

// To use a test fixture, derive a class from testing::Test.
class MiscMath : public testing::Test {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};


// Test fixture for bound_min_max()
class BoundMinMax : public MiscMath {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(BoundMinMax, ValBelowZeroRange) {
  // Test lower bounding when min = max with (val < min)
  EXPECT_EQ(-1.0f, bound_min_max(-10.0f, -1.0f, -1.0f));
  EXPECT_EQ(0.0f, bound_min_max(-10.0f, 0.0f, 0.0f));
  EXPECT_EQ(1.0f, bound_min_max(-10.0f, 1.0f, 1.0f));
};

TEST_F(BoundMinMax, ValWithinZeroRange) {
  // Test bounding when min = max = val
  EXPECT_EQ(-1.0f, bound_min_max(-1.0f, -1.0f, -1.0f));
  EXPECT_EQ(0.0f, bound_min_max(0.0f, 0.0f, 0.0f));
  EXPECT_EQ(1.0f, bound_min_max(1.0f, 1.0f, 1.0f));
};

TEST_F(BoundMinMax, ValAboveZeroRange) {
  // Test upper bounding when min = max with (val > max)
  EXPECT_EQ(-1.0f, bound_min_max(10.0f, -1.0f, -1.0f));
  EXPECT_EQ(0.0f, bound_min_max(10.0f, 0.0f, 0.0f));
  EXPECT_EQ(1.0f, bound_min_max(10.0f, 1.0f, 1.0f));
}

TEST_F(BoundMinMax, PositiveMinMax) {
  float min = 1.0f;
  float max = 10.0f;

  // Below Lower Bound
  EXPECT_EQ(min, bound_min_max(min - 1.0f, min, max));
  // At Lower Bound
  EXPECT_EQ(min, bound_min_max(min, min, max));
  // In Bounds
  EXPECT_EQ(2.0f, bound_min_max(2.0f, min, max));
  // At Upper Bound
  EXPECT_EQ(max, bound_min_max(max, min, max));
  // Above Upper Bound
  EXPECT_EQ(max, bound_min_max(max + 1.0f, min, max));
}

TEST_F(BoundMinMax, NegativeMinMax) {
  float min = -10.0f;
  float max = -1.0f;

  // Below Lower Bound
  EXPECT_EQ(min, bound_min_max(min - 1.0f, min, max));
  // At Lower Bound
  EXPECT_EQ(min, bound_min_max(min, min, max));
  // In Bounds
  EXPECT_EQ(-2.0f, bound_min_max(-2.0f, min, max));
  // At Upper Bound
  EXPECT_EQ(max, bound_min_max(max, min, max));
  // Above Upper Bound
  EXPECT_EQ(max, bound_min_max(max + 1.0f, min, max));
}

TEST_F(BoundMinMax, StraddleZeroMinMax) {
  float min = -10.0f;
  float max = 10.0f;

  // Below Lower Bound
  EXPECT_EQ(min, bound_min_max(min - 1.0f, min, max));
  // At Lower Bound
  EXPECT_EQ(min, bound_min_max(min, min, max));
  // In Bounds
  EXPECT_EQ(0.0f, bound_min_max(0.0f, min, max));
  // At Upper Bound
  EXPECT_EQ(max, bound_min_max(max, min, max));
  // Above Upper Bound
  EXPECT_EQ(max, bound_min_max(max + 1.0f, min, max));
}

// Test fixture for bound_sym()
class BoundSym : public MiscMath {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(BoundSym, ZeroRange) {
  float range = 0.0f;

  // Below Lower Bound
  EXPECT_EQ(-range, bound_sym(-range - 1.0f, range));
  // At Lower Bound
  EXPECT_EQ(-range, bound_sym(-range, range));
  // In Bounds
  EXPECT_EQ(0.0f, bound_sym(0.0f, range));
  // At Upper Bound
  EXPECT_EQ(range, bound_sym(range, range));
  // Above Upper Bound
  EXPECT_EQ(range, bound_sym(range + 1.0f, range));
};

TEST_F(BoundSym, NonZeroRange) {
  float range = 10.0f;

  // Below Lower Bound
  EXPECT_EQ(-range, bound_sym(-range - 1.0f, range));
  // At Lower Bound
  EXPECT_EQ(-range, bound_sym(-range, range));
  // In Bounds
  EXPECT_EQ(0.0f, bound_sym(0.0f, range));
  // At Upper Bound
  EXPECT_EQ(range, bound_sym(range, range));
  // Above Upper Bound
  EXPECT_EQ(range, bound_sym(range + 1.0f, range));
};


// Test fixture for circular_modulus_deg()
class CircularModulus : public MiscMath {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(CircularModulus, NullError) {
  float eps = 0.005;
  float error = 0.0f;
  uint8_t num_test_inputs = 9;

  float test_inputs[num_test_inputs];
  test_inputs[0] = error - 3600000;
  test_inputs[1] = error - 1080;
  test_inputs[2] = error - 720;
  test_inputs[3] = error - 360;
  test_inputs[4] = error;
  test_inputs[5] = error + 360;
  test_inputs[6] = error + 720;
  test_inputs[7] = error + 1080;
  test_inputs[8] = error + 3600000;
  for (uint8_t i=0; i < num_test_inputs; i++){
    EXPECT_EQ(-error, circular_modulus_deg(test_inputs[i]));
    ASSERT_NEAR(-error * DEG2RAD, circular_modulus_rad(test_inputs[i] * DEG2RAD), eps);
  }
};

TEST_F(CircularModulus, MaxPosError) {
  // Use fabs() for +/-180.0 to accept either -180.0 or +180.0 as valid and correct
  float eps = 0.005;
  float error = 180.0f;
  uint8_t num_test_inputs = 9;

  float test_inputs[num_test_inputs];
  test_inputs[0] = error - 3600000;
  test_inputs[1] = error - 1080;
  test_inputs[2] = error - 720;
  test_inputs[3] = error - 360;
  test_inputs[4] = error;
  test_inputs[5] = error + 360;
  test_inputs[6] = error + 720;
  test_inputs[7] = error + 1080;
  test_inputs[8] = error + 3600000;
  for (uint8_t i=0; i < num_test_inputs; i++){
    EXPECT_EQ(error, fabsf(circular_modulus_deg(test_inputs[i])));
    ASSERT_NEAR(error * DEG2RAD, fabsf(circular_modulus_rad(test_inputs[i] * DEG2RAD)), eps);
  }
};

TEST_F(CircularModulus, MaxNegError) {
  // Use fabs() for +/-180.0 to accept either -180.0 or +180.0 as valid and correct
  float eps = 0.005;
  float error = 180.0f;
  uint8_t num_test_inputs = 9;

  float test_inputs[num_test_inputs];
  test_inputs[0] = -error - 3600000;
  test_inputs[1] = -error - 1080;
  test_inputs[2] = -error - 720;
  test_inputs[3] = -error - 360;
  test_inputs[4] = -error;
  test_inputs[5] = -error + 360;
  test_inputs[6] = -error + 720;
  test_inputs[7] = -error + 1080;
  test_inputs[8] = -error + 3600000;
  for (uint8_t i=0; i < num_test_inputs; i++){
    EXPECT_EQ(error, fabsf(circular_modulus_deg(test_inputs[i])));
    ASSERT_NEAR(error * DEG2RAD, fabsf(circular_modulus_rad(test_inputs[i] * DEG2RAD)), eps);
  }
};

TEST_F(CircularModulus, SweepError) {
  float eps = 0.0001f;
  uint8_t num_test_inputs = 7;

  for (float error = -179.9f; error < 179.9f; error += 0.001f) {
    float test_inputs[num_test_inputs];
    test_inputs[0] = error - 1080;
    test_inputs[1] = error - 720;
    test_inputs[2] = error - 360;
    test_inputs[3] = error;
    test_inputs[4] = error + 360;
    test_inputs[5] = error + 720;
    test_inputs[6] = error + 1080;
    for (uint8_t i=0; i < num_test_inputs; i++){
      ASSERT_NEAR(error, circular_modulus_deg(test_inputs[i]), eps);
      ASSERT_NEAR(error * DEG2RAD, circular_modulus_rad(test_inputs[i] * DEG2RAD), eps);
    }
  }
};


// Test fixture for find_arc_center()
class FindArcCenter : public MiscMath {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(FindArcCenter, ArcCenterExists) {
  float eps = 0.0001f;
  float center[2];
  enum arc_center_results ret;

  for (int i=0; i< 10; i++){
    float radius = 1.1;
    float center0[2] = {pow(1.1, i), pow(-0.9, i)}; // pseudo-random center
	float theta[2] = {0, 2*PI/11};
	bool arc_minor = i%2;
	bool arc_sense = i%2;
    float start_point[2] = {center0[0] + radius*cosf(theta[0]), center0[1] + radius*sinf(theta[0])};
    float end_point[2]   = {center0[0] + radius*cosf(theta[1]), center0[1] + radius*sinf(theta[1])};
    ret = find_arc_center(start_point, end_point, radius, arc_sense, arc_minor, center);

    // Test lower bounding when min = max with (val < min)
    EXPECT_EQ(ARC_CENTER_FOUND, ret);
    ASSERT_NEAR(center0[0], center[0], eps);
    ASSERT_NEAR(center0[1], center[1], eps);
  }
};

TEST_F(FindArcCenter, ArcCenterBarelyExists) {
  // Test a .5% error in radius
  float radius_scale_error = 1.005;
  float center[2];
  enum arc_center_results ret;

  for (int i=0; i< 10; i++){
    float radius = 1.1;
    float center0[2] = {pow(1.1, i), pow(-0.9, i)}; // pseudo-random center
	float theta[2] = {0, 2*PI/11};
	bool arc_minor = i%2;
	bool arc_sense = i%2;
    float start_point[2] = {center0[0] + radius_scale_error*radius*cosf(theta[0]), center0[1] + radius_scale_error*radius*sinf(theta[0])};
    float end_point[2]   = {center0[0] + radius_scale_error*radius*cosf(theta[1]), center0[1] + radius_scale_error*radius*sinf(theta[1])};
	ret = find_arc_center(start_point, end_point, radius, arc_sense, arc_minor, center);

    // Test lower bounding when min = max with (val < min)
    EXPECT_EQ(ARC_CENTER_FOUND, ret);
    ASSERT_NEAR(center0[0], center[0], radius*(radius_scale_error - 1+.001));
    ASSERT_NEAR(center0[1], center[1], radius*(radius_scale_error - 1+.001));
  }
};

TEST_F(FindArcCenter, ArcCenterDoesNotExist) {
  // Test a 1.5% error in radius
  float radius_scale_error = 1.15;
  float center[2];
  enum arc_center_results ret;

  for (int i=0; i< 10; i++){
	float radius = 1.1;
    float center0[2] = {pow(1.1, i), pow(-0.9, i)}; // pseudo-random center
	float theta[2] = {0, 2*PI/11};
	bool arc_minor = i%2;
	bool arc_sense = i%2;
    float start_point[2] = {center0[0] + radius_scale_error*radius*cosf(theta[0]), center0[1] + radius_scale_error*radius*sinf(theta[0])};
    float end_point[2]   = {center0[0] + radius_scale_error*radius*cosf(theta[0]+PI), center0[1] + radius_scale_error*radius*sinf(theta[0]+PI)};
	ret = find_arc_center(start_point, end_point, radius, arc_sense, arc_minor, center);

    // Test lower bounding when min = max with (val < min)
    EXPECT_EQ(ARC_INSUFFICIENT_RADIUS, ret);
  }
}

TEST_F(FindArcCenter, CoincidentInputs) {
  float center[2];
  enum arc_center_results ret;

  for (int i=0; i< 10; i++){
    float radius = 1.1;
    float center0[2] = {pow(1.1, i), pow(-0.9, i)}; // pseudo-random center
    float theta = expf(i); // pseudo-random angles
    float start_point[2] = {center0[0] + radius*cosf(theta), center0[1] + radius*sinf(theta)};
    ret = find_arc_center(start_point, start_point, radius, i%2, i%2, center);

    // Test lower bounding when min = max with (val < min)
    EXPECT_EQ(ARC_COINCIDENT_POINTS, ret);
  }
}


// Test fixture for measure_arc_rad()
class MeasureArcRad : public MiscMath {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(MeasureArcRad, SeparatedPoints) {
  float eps = 0.005f;
  float phi;
  for (int i=0; i< 10; i++){
    float radius = exp(i);
    float center0[2] = {pow(1.1, i), pow(-0.9, i)}; // pseudo-random center
    float theta[2] = {expf(i), expf(i+1)}; // pseudo-random angles
    float start_point[2] = {center0[0] + radius*cosf(theta[0]), center0[1] + radius*sinf(theta[0])};
    float end_point[2]   = {center0[0] + radius*cosf(theta[1]), center0[1] + radius*sinf(theta[1])};
    phi = measure_arc_rad(start_point, end_point, center0);

    // Test lower bounding when min = max with (val < min)
    ASSERT_NEAR(circular_modulus_rad(theta[1]-theta[0]), circular_modulus_rad(phi), eps);
  }
};

TEST_F(MeasureArcRad, CoincidentPoints) {
  float eps = 0.0001f;
  float phi;
  for (int i=0; i< 10; i++){
    float radius = exp(i);
    float center0[2] = {pow(1.1, i), pow(-0.9, i)}; // pseudo-random center
    float theta = expf(i); // pseudo-random angles
    float start_point[2] = {center0[0] + radius*cosf(theta), center0[1] + radius*sinf(theta)};
    phi = measure_arc_rad(start_point, start_point, center0);

    // Test lower bounding when min = max with (val < min)
    ASSERT_NEAR(0, phi, eps);
  }
};


// Test fixture for angle_between_2d_vectors()
class AngleBetween2dVectors : public MiscMath {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(AngleBetween2dVectors, DivergentVectors) {
  float eps = 0.005f;
  float phi;
  for (int i=0; i< 10; i++){
    float theta[2] = {expf(i), expf(i+1)}; // pseudo-random angles
    float mag_a = expf(i);
    float mag_b = pow(i+1, 1.1);
    float a[2] = {mag_a * cosf(theta[0]), mag_a * sinf(theta[0])};
    float b[2] = {mag_b * cosf(theta[1]), mag_b * sinf(theta[1])};
    phi = angle_between_2d_vectors(a, b);

    // Test lower bounding when min = max with (val < min)
    ASSERT_NEAR(circular_modulus_rad(theta[1]-theta[0]), circular_modulus_rad(phi), eps);
  }
};

TEST_F(AngleBetween2dVectors, ParallelVectors) {
  float eps = 0.0001f;
  float phi;
  for (int i=0; i< 10; i++){
    float theta = expf(i); // pseudo-random angles
    float mag_a = expf(i);
    float a[2] = {mag_a * cosf(theta), mag_a * sinf(theta)};
    phi = angle_between_2d_vectors(a, a);

    // Test lower bounding when min = max with (val < min)
    ASSERT_NEAR(0, phi, eps);
  }
};

