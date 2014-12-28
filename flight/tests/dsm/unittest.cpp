/**
 ******************************************************************************
 * @file       unittest.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
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

#include "dsm.h"

}

// example data can be found at http://wiki.paparazziuav.org/wiki/DSM
// format described at https://bitbucket.org/PhracturedBlue/deviation/src/92e1705cf895b415ab16f6e1d7df93ee11d55afe/doc/DSM.txt?at=default

// To use a test fixture, derive a class from testing::Test.
class DsmTest : public testing::Test {
protected:
  virtual void SetUp() {
    PIOS_DSM_Reset(&dev);
    state = &dev.state;
  }

  virtual void TearDown() {
 }
 void pack_channels_10bit(uint16_t channels[DSM_CHANNELS_PER_FRAME], struct pios_dsm_state *state, bool frame);
 void pack_channels_11bit(uint16_t channels[DSM_CHANNELS_PER_FRAME], struct pios_dsm_state *state, bool frame);
 struct pios_dsm_state *state;
 struct pios_dsm_dev dev;
};

//! pack data into DSM2 10 bit packets
void DsmTest::pack_channels_10bit(uint16_t channels[DSM_CHANNELS_PER_FRAME], struct pios_dsm_state *state, bool frame)
{
  for (int i = 0; i < DSM_CHANNELS_PER_FRAME; i++) {
    uint16_t word = (frame ? 0x8000 : 0) | ((i & 0x000F) << 10) | (channels[i] & 0x03FF);
    state->received_data[2 + i * 2 + 1] = word & 0x00FF;
    state->received_data[2 + i * 2] = (word >> 8) & 0x00FF;
  }
}

//! pack data into DSM2 11 bit packets
void DsmTest::pack_channels_11bit(uint16_t channels[DSM_CHANNELS_PER_FRAME], struct pios_dsm_state *state, bool frame)
{
  for (int i = 0; i < DSM_CHANNELS_PER_FRAME; i++) {
    uint16_t word = (frame ? 0x8000 : 0) | ((i & 0x000F) << 11) | (channels[i] & 0x07FF);
    state->received_data[2 + i * 2 + 1] = word & 0x00FF;
    state->received_data[2 + i * 2] = (word >> 8) & 0x00FF;
  }
}

void verify_channels(uint16_t *c1, uint16_t *c2)
{
  for (int i = 0; i < DSM_CHANNELS_PER_FRAME; i++) {
    EXPECT_EQ(c1[i], c2[i]);
  }
}

TEST_F(DsmTest, Invalid) {
  uint16_t channels[DSM_CHANNELS_PER_FRAME] = {512, 513, 514, 515, 516, 517, 518};
  pack_channels_10bit(channels, state, false);
  for (int i = 0; i < DSM_FRAME_LENGTH; i++)
    state->received_data[i] = 0;
  EXPECT_EQ(-2, PIOS_DSM_UnrollChannels(&dev));
}

TEST_F(DsmTest, DSM2_10BIT) {
  uint16_t channels[DSM_CHANNELS_PER_FRAME] = {512, 513, 514, 515, 516, 517, 518};
  pack_channels_10bit(channels, state, false);
  EXPECT_EQ(0, PIOS_DSM_UnrollChannels(&dev));

  EXPECT_EQ(10, PIOS_DSM_GetResolution(&dev));
  verify_channels(channels, state->channel_data);
}

TEST_F(DsmTest, DSM2_11BIT) {
  uint16_t channels[DSM_CHANNELS_PER_FRAME] = {512, 513, 514, 515, 516, 517, 518};
  pack_channels_11bit(channels, state, false);
  EXPECT_EQ(0, PIOS_DSM_UnrollChannels(&dev));

  EXPECT_EQ(11, PIOS_DSM_GetResolution(&dev));
  verify_channels(channels, state->channel_data);
}

TEST_F(DsmTest, DSM2_10BIT_MISMATCH) {
  // Once the resolution is detected as 11 bit, processing
  // 10 bit frames should fail
  dev.resolution = DSM_11BIT;

  uint16_t channels[DSM_CHANNELS_PER_FRAME] = {512, 513, 514, 515, 516, 517, 518};
  pack_channels_10bit(channels, state, false);
  EXPECT_EQ(-1, PIOS_DSM_UnrollChannels(&dev));
}

TEST_F(DsmTest, DSM2_11BIT_MISMATCH) {
  // Once the resolution is detected as 10 bit, processing
  // 11 bit frames should fail
  dev.resolution = DSM_10BIT;

  uint16_t channels[DSM_CHANNELS_PER_FRAME] = {512, 513, 514, 515, 516, 517, 518};
  pack_channels_11bit(channels, state, false);
  EXPECT_EQ(-1, PIOS_DSM_UnrollChannels(&dev));
}

