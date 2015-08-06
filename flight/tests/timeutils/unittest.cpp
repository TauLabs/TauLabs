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

#include <time.h>       /* time functions */
#include <stdio.h>		/* printf */
#include <stdlib.h>		/* abort */
#include <string.h>		/* memset */
#include <stdint.h>		/* uint*_t */

extern "C" {
#include "timeutils.h"		/* API for time conversion */
}



// To use a test fixture, derive a class from testing::Test.
class Time : public testing::Test {
protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(Time, DateSweep) {
    struct tm *datetime1;
    DateTimeT datetime2;
    time_t timestamp = 0;
 
    // test every day from 1970 to 2100
    struct tm tm;
    memset((void*)&tm, 0, sizeof(tm));
    tm.tm_year = 130;
    time_t end_time = mktime(&tm);

    while (timestamp < end_time){
        datetime1 = gmtime(&timestamp);
        date_from_timestamp((uint32_t)timestamp, &datetime2);
        EXPECT_EQ(datetime1->tm_year, datetime2.year);
        EXPECT_EQ(datetime1->tm_mon, datetime2.mon);
        EXPECT_EQ(datetime1->tm_mday, datetime2.mday);
        EXPECT_EQ(datetime1->tm_hour, datetime2.hour);
        EXPECT_EQ(datetime1->tm_min, datetime2.min);
        EXPECT_EQ(datetime1->tm_sec, datetime2.sec);
        EXPECT_EQ(datetime1->tm_wday, datetime2.wday);
        timestamp += 86400 + 3600 + 42; // advance by one day, one hour, 42 seconds
    }
}


