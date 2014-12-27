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
 int validate_file(const char *fn, int resolution, int channels);
 int get_packet(FILE *fid, uint8_t *buf);
 struct pios_dsm_state *state;
 struct pios_dsm_dev dev;
};

const int idx[] = {1,5,2,3,0,7,6,1,5,2,3,4,8,9};

//! pack data into DSM2 10 bit packets
void DsmTest::pack_channels_10bit(uint16_t channels[DSM_CHANNELS_PER_FRAME], struct pios_dsm_state *state, bool frame)
{
  for (int i = 0; i < DSM_CHANNELS_PER_FRAME; i++) {
    uint16_t j = idx[i + DSM_CHANNELS_PER_FRAME*frame];
    uint16_t val = channels[j];
    uint16_t word = ((frame & (i == 0)) ? 0x8000 : 0) | ((j & 0x000F) << 10) | (val & 0x03FF);
    state->received_data[2 + i * 2 + 1] = word & 0x00FF;
    state->received_data[2 + i * 2] = (word >> 8) & 0x00FF;
  }
}

//! pack data into DSM2 11 bit packets
void DsmTest::pack_channels_11bit(uint16_t channels[DSM_CHANNELS_PER_FRAME], struct pios_dsm_state *state, bool frame)
{
  for (int i = 0; i < DSM_CHANNELS_PER_FRAME; i++) {
    uint16_t j = idx[i + DSM_CHANNELS_PER_FRAME*frame];
    uint16_t val = channels[j];
    uint16_t word = ((frame  & (i == 0)) ? 0x8000 : 0) | ((j & 0x000F) << 11) | (val & 0x07FF);
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

TEST_F(DsmTest, DSM_10BIT) {
  uint16_t channels[PIOS_DSM_NUM_INPUTS] = {512, 513, 514, 515, 516, 517, 518, 0, 0, 0, 0, 0};
  pack_channels_10bit(channels, state, false);
  EXPECT_EQ(0, PIOS_DSM_UnrollChannels(&dev));
  pack_channels_10bit(channels, state, true);
  EXPECT_EQ(0, PIOS_DSM_UnrollChannels(&dev));

  EXPECT_EQ(10, PIOS_DSM_GetResolution(&dev));
  verify_channels(channels, state->channel_data);
}

TEST_F(DsmTest, DSM_11BIT) {
  uint16_t channels[PIOS_DSM_NUM_INPUTS] = {512, 513, 514, 515, 516, 517, 518, 0, 0, 0, 0, 0};
  pack_channels_11bit(channels, state, false);
  EXPECT_EQ(0, PIOS_DSM_UnrollChannels(&dev));
  pack_channels_11bit(channels, state, true);
  EXPECT_EQ(0, PIOS_DSM_UnrollChannels(&dev));

  EXPECT_EQ(11, PIOS_DSM_GetResolution(&dev));
  verify_channels(channels, state->channel_data);
}

int DsmTest::get_packet(FILE *fid, uint8_t *buf)
{
  double t;
  uint8_t val;

  for (int i = 0; i < DSM_FRAME_LENGTH; i++) {
    if (fscanf(fid, "%lf,%x,,", &t, (unsigned int*) &val) == 2) {
      buf[i] = val;
    } else
      return -1;
  }
  return 0;
}


int DsmTest::validate_file(const char *fn, int resolution, int channels)
{
  FILE *fid = fopen(fn, "r");
  char *line = NULL;
  size_t len = 0;

  // throwaway intro line
  getline(&line, &len, fid);

  const int MIN = (resolution == 11) ? 340 : 150;
  const int MAX = (resolution == 11) ? 2048 : 1024;
  if (resolution == 11)
  // warm up parser
  get_packet(fid, state->received_data);
  PIOS_DSM_UnrollChannels(&dev);
  //EXPECT_EQ(0, PIOS_DSM_UnrollChannels(&dev));

  get_packet(fid, state->received_data);
  PIOS_DSM_UnrollChannels(&dev);
  //EXPECT_EQ(0, PIOS_DSM_UnrollChannels(&dev));

  while(get_packet(fid, state->received_data) == 0) {
    EXPECT_EQ(0, PIOS_DSM_UnrollChannels(&dev));
    EXPECT_EQ(resolution, PIOS_DSM_GetResolution(&dev));

    bool valid[PIOS_DSM_NUM_INPUTS];
    for (int i = 0; i < PIOS_DSM_NUM_INPUTS; i++) {
      // this file only has 7 channels
      valid[i] = ((i >= channels) && dev.state.channel_data[i] == 0) ||
              ((i < channels) && ((dev.state.channel_data[i] > MIN) && (dev.state.channel_data[i] <= MAX)));
      //fprintf(stdout, "%d %d %d %d\r\n", i, valid[i], dev.state.channel_data[i], channels);
      EXPECT_TRUE(valid[i]);

      if (!valid[i]) {
        for (int i = 0; i < PIOS_DSM_NUM_INPUTS; i++) {
          fprintf(stdout, "%d, ", dev.state.channel_data[i]);
        }
        fprintf(stdout, "\r\n");

        fclose(fid);
        return 0;
      }
    }

    /*for (int i = 0; i < PIOS_DSM_NUM_INPUTS; i++) {
      fprintf(stdout, "%d, ", dev.state.channel_data[i]);
    }
    fprintf(stdout, "\r\n");*/
  }

  fclose(fid);

  return 0;
}

TEST_F(DsmTest, DX7_DSM2) {

  PIOS_DSM_Reset(&dev);
  validate_file("DX7_11msDSM2.txt",11,8);

  PIOS_DSM_Reset(&dev);
  validate_file("DX7_22msDSM2.txt",11,8);
}

TEST_F(DsmTest, DX7_DSMX) {
  PIOS_DSM_Reset(&dev);
  validate_file("DX7_11msDSMX.txt",11,8);

  PIOS_DSM_Reset(&dev);
  validate_file("DX7_22msDSMX.txt",11,8);
}

TEST_F(DsmTest, DX18_DSM2) {
  PIOS_DSM_Reset(&dev);
  validate_file("DX18_11msDSM2_2048res.txt",11,10);

  // Note: even though this file is named like it has 10 bit
  // resolution, it appears to differ in having 12 channels
  PIOS_DSM_Reset(&dev);
  validate_file("DX18_22msDSM2_1024res.txt",11,12);

  // Note: we do not decode the XPlus channels but it is
  // important to make sure they are handled appropriately
  PIOS_DSM_Reset(&dev);
  validate_file("DX18_22msDSM2_XPlus_1024res.txt",11,12);
}

TEST_F(DsmTest, DX18_DSMX) {
  validate_file("DX18_22msDSMX.txt",11,12);

  PIOS_DSM_Reset(&dev);
  validate_file("DX18_11msDSMX.txt",11,10);

  PIOS_DSM_Reset(&dev);
  validate_file("DX18_22msDSMX_XPlus.txt",11,12);
}
