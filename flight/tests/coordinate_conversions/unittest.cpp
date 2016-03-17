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

// Test fixture for RneFromLLA()
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


//// Test fixture for LLA lineraization()
class LLALineraization : public CoordConversion {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

//Following tests compare the ratio between expected and returned values, and not absolute differences.
TEST_F(LLALineraization, Equator_float) {
  // Test location is Equator
  float homeLLA_D[] = {0, 0, 0};
  float dN = 0.0111319490793274;
  float dE = 0.0111319490793274;
  float dD = -1;

  float linearized_conversion_factor_f[3];
  LLA2NED_linearization_float(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_f);

  float eps = 0.000001f;

  ASSERT_NEAR(dN/linearized_conversion_factor_f[0], 1, eps);
  ASSERT_NEAR(dE/linearized_conversion_factor_f[1], 1, eps);
  ASSERT_NEAR(dD/linearized_conversion_factor_f[2], 1, eps);

};

TEST_F(LLALineraization, Aconagua_float) {
  // Test location is Equator
  float homeLLA_D[] = {-32.653431*1e7, -70.011083*1e7, 6961};
  float dN = 0.0111440983163;
  float dE = 0.00938276913926;
  float dD = -1;

  float linearized_conversion_factor_f[3];
  LLA2NED_linearization_float(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_f);

  float eps = 0.000001f;

  ASSERT_NEAR(dN/linearized_conversion_factor_f[0], 1, eps);
  ASSERT_NEAR(dE/linearized_conversion_factor_f[1], 1, eps);
  ASSERT_NEAR(dD/linearized_conversion_factor_f[2], 1, eps);

};

TEST_F(LLALineraization, ChallengerDeep_float) {
  // Test location is Equator
  float homeLLA_D[] = {11.4*1e7, 143.266667*1e7, -10898};
  float dN = .0111129284811;
  float dE = .0108936834557;
  float dD = -1;

  float linearized_conversion_factor_f[3];
  LLA2NED_linearization_float(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_f);

  float eps = 0.000001f;

  ASSERT_NEAR(dN/linearized_conversion_factor_f[0], 1, eps);
  ASSERT_NEAR(dE/linearized_conversion_factor_f[1], 1, eps);
  ASSERT_NEAR(dD/linearized_conversion_factor_f[2], 1, eps);

};


TEST_F(LLALineraization, Boston_float) {
  // Test location is Boston (According to wikipedia)
  float homeLLA_D[] = {42.37*1e7, -71.03*1e7, 10};
  float dN = 0.0111319665326199;
  float dE = 0.00822438930622829;
  float dD = -1;

  float linearized_conversion_factor_f[3];
  LLA2NED_linearization_float(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_f);

  float eps = 0.000001f;

  ASSERT_NEAR(dN/linearized_conversion_factor_f[0], 1, eps);
  ASSERT_NEAR(dE/linearized_conversion_factor_f[1], 1, eps);
  ASSERT_NEAR(dD/linearized_conversion_factor_f[2], 1, eps);

};

TEST_F(LLALineraization, Sydney_float) {
  // Test location is Sydney, AUS (According to wikipedia)
  float homeLLA_D[] = {-33.86*1e7, 151.2*1e7, 10};
  float dN = 0.0111319665326199;
  float dE = 0.00924400128736662;
  float dD = -1;

  float linearized_conversion_factor_f[3];
  LLA2NED_linearization_float(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_f);

  float eps = 0.000001f;

  ASSERT_NEAR(dN/linearized_conversion_factor_f[0], 1, eps);
  ASSERT_NEAR(dE/linearized_conversion_factor_f[1], 1, eps);
  ASSERT_NEAR(dD/linearized_conversion_factor_f[2], 1, eps);

};

TEST_F(LLALineraization, Equator_double) {
  // Test location is Equator
  double homeLLA_D[] = {0, 0, 0};
  double dN = 0.0111319490793274;
  double dE = 0.0111319490793274;
  double dD = -1;

  double linearized_conversion_factor_d[3];
  LLA2NED_linearization_double(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_d);

  float eps = 0.0000001f;
  ASSERT_NEAR(dN/linearized_conversion_factor_d[0], 1, eps);
  ASSERT_NEAR(dE/linearized_conversion_factor_d[1], 1, eps);
  ASSERT_NEAR(dD/linearized_conversion_factor_d[2], 1, eps);

};

TEST_F(LLALineraization, Boston_double) {
  // Test location is Boston (According to wikipedia)
  double homeLLA_D[] = {42.37*1e7, -71.03*1e7, 10};
  double dN = 0.0111319665326199;
  double dE = 0.00822438930622829;
  double dD = -1;

  double linearized_conversion_factor_d[3];
  LLA2NED_linearization_double(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_d);

  float eps = 0.0000001f;
  ASSERT_NEAR(dN/linearized_conversion_factor_d[0], 1, eps);
  ASSERT_NEAR(dE/linearized_conversion_factor_d[1], 1, eps);
  ASSERT_NEAR(dD/linearized_conversion_factor_d[2], 1, eps);

};

TEST_F(LLALineraization, Sydney_double) {
  // Test location is Sydney, AUS (According to wikipedia)
  float homeLLA_D[] = {-33.86*1e7, 151.2*1e7, 10};
  float dN = 0.0111319665326199;
  float dE = 0.00924400128736662;
  float dD = -1;

  float linearized_conversion_factor_f[3];
  LLA2NED_linearization_float(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_f);

  float eps = 0.000001f;
  ASSERT_NEAR(dN/linearized_conversion_factor_f[0], 1, eps);
  ASSERT_NEAR(dE/linearized_conversion_factor_f[1], 1, eps);
  ASSERT_NEAR(dD/linearized_conversion_factor_f[2], 1, eps);

};

TEST_F(LLALineraization, Equator_on_ground) {
  // Test location is Equator and Prime Meridian (According to wikipedia). This is just checking that everything works if
  // there is no position change
  float homeLLA_D[] = {0, 0, 0};
  float currentLLA_D[] = {0, 0, 0};

  float linearized_conversion_factor_f[3];
  LLA2NED_linearization_float(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_f);

  float NED[3];
  get_linearized_3D_transformation(currentLLA_D[0], currentLLA_D[1], currentLLA_D[2],
                                   homeLLA_D[0], homeLLA_D[1], homeLLA_D[2],
                                   linearized_conversion_factor_f, NED);
  float eps = 0.000001f;
  ASSERT_NEAR(0, NED[0], eps);
  ASSERT_NEAR(0, NED[1], eps);
  ASSERT_NEAR(0, NED[2], eps);
};


TEST_F(LLALineraization, Equator_to_Takeoff) {
  // Test location is Equator and Prime Meridian (According to wikipedia)
  double homeLLA_D[] = {0, 0, 0};
  float currentLLA_D[] = {0, 0, 20};


  float linearized_conversion_factor_f[3];
  LLA2NED_linearization_float(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_f);

  float NED[3];
  get_linearized_3D_transformation(currentLLA_D[0], currentLLA_D[1], currentLLA_D[2],
                                   homeLLA_D[0], homeLLA_D[1], homeLLA_D[2],
                                   linearized_conversion_factor_f, NED);
  float eps = 0.000001f;
  ASSERT_NEAR(0, NED[0], eps);
  ASSERT_NEAR(0, NED[1], eps);
  ASSERT_NEAR(-20, NED[2], eps);

};

TEST_F(LLALineraization, Boston) {
  // Test location is Boston, position change is smallest distance system can calculate. (true for following Aconagua
  // and Challenger Deep tests
  float homeLLA_D[] = {42.37*1e7, -71.03*1e7, 0};

  int32_t dLat = 1;
  int32_t dLon = 1;
  float dAlt = 1;

  int32_t currentLat = (int32_t)homeLLA_D[0] + dLat;
  int32_t currentLon = (int32_t)homeLLA_D[1] + dLon;
  float currentAlt   = homeLLA_D[2] + dAlt;

  float linearized_conversion_factor_f[3];
  LLA2NED_linearization_float(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_f);

  float NED[3];
  get_linearized_3D_transformation(currentLat, currentLon, currentAlt,
                                   homeLLA_D[0], homeLLA_D[1], homeLLA_D[2],
                                   linearized_conversion_factor_f, NED);

  float eps = 0.003f;

  double N_expected = 0.01110805;
  double E_expected = 0.00823691;
  double D_expected = -1;

  ASSERT_NEAR(N_expected / NED[0], 1, eps);
  ASSERT_NEAR(E_expected / NED[1], 1, eps);
  ASSERT_NEAR(D_expected / NED[2], 1, eps);

};


TEST_F(LLALineraization, Aconagua) {
  // Test location is Mt. Aconagua, mountain in Southern Hemisphere
  float homeLLA_D[] = {-32.653431*1e7, -70.011083*1e7, 6961};

  int32_t dLat = 1;
  int32_t dLon = 1;
  float dAlt = 1;

  int32_t currentLat = (int32_t)homeLLA_D[0] + dLat;
  int32_t currentLon = (int32_t)homeLLA_D[1] + dLon;
  float currentAlt   = homeLLA_D[2] + dAlt;

  float linearized_conversion_factor_f[3];
  LLA2NED_linearization_float(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_f);

  float NED[3];
  get_linearized_3D_transformation(currentLat, currentLon, currentAlt,
                                   homeLLA_D[0], homeLLA_D[1], homeLLA_D[2],
                                   linearized_conversion_factor_f, NED);

  float eps = 0.004f;

  double N_expected = .01110198;
  double E_expected = .00939192;
  double D_expected = -1;

  ASSERT_NEAR(N_expected, NED[0], eps);

  ASSERT_NEAR(N_expected / NED[0], 1, eps);
  ASSERT_NEAR(E_expected / NED[1], 1, eps);
  ASSERT_NEAR(D_expected / NED[2], 1, eps);

};

TEST_F(LLALineraization, ChallengerDeep) {
  // Test location is Challenger Deep, deepest point in ocean (northern hemisphere)
  float homeLLA_D[] = {11.4*1e7, 143.266667*1e7, -10898};

  int32_t dLat = 1;
  int32_t dLon = 1;
  float dAlt = 1;

  int32_t currentLat = (int32_t)homeLLA_D[0] + dLat;
  int32_t currentLon = (int32_t)homeLLA_D[1] + dLon;
  float currentAlt   = homeLLA_D[2] + dAlt;

  float linearized_conversion_factor_f[3];
  LLA2NED_linearization_float(homeLLA_D[0], homeLLA_D[2], linearized_conversion_factor_f);

  float NED[3];
  get_linearized_3D_transformation(currentLat, currentLon, currentAlt,
                                   homeLLA_D[0], homeLLA_D[1], homeLLA_D[2],
                                   linearized_conversion_factor_f, NED);

  float eps = 0.007f;

  double N_expected = .01104275;
  double E_expected = .01089511;
  double D_expected = -1;

  ASSERT_NEAR(N_expected, NED[0], eps);

  ASSERT_NEAR(N_expected / NED[0], 1, eps);
  ASSERT_NEAR(E_expected / NED[1], 1, eps);
  ASSERT_NEAR(D_expected / NED[2], 1, eps);



};

