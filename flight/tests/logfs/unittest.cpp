/**
 ******************************************************************************
 * @file       unittest.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012
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

#include "pios_flash.h"		/* PIOS_FLASH_* API */

#include "pios_flash_priv.h"	/* struct pios_flash_partition */

extern const struct pios_flash_partition pios_flash_partition_table[];
extern uint32_t pios_flash_partition_table_size;

#include "pios_flash_posix_priv.h"

extern uintptr_t pios_posix_flash_id;
extern struct pios_flash_posix_cfg flash_config;

#include "pios_flashfs_logfs_priv.h"

extern struct flashfs_logfs_cfg flashfs_config_settings;
extern struct flashfs_logfs_cfg flashfs_config_waypoints;

#include "pios_flashfs.h"	/* PIOS_FLASHFS_* */

}

#define OBJ0_ID 0xAA55AA55

#define OBJ1_ID 0x12345678
#define OBJ1_SIZE 76

#define OBJ2_ID 0xABCDEFAB
#define OBJ2_SIZE 123

#define OBJ3_ID 0x99999999
#define OBJ3_SIZE (256 - 12) // leave room for the slot header

#define OBJ4_ID 0x90901111
#define OBJ4_SIZE (768)		// only fits in partition b slots

// To use a test fixture, derive a class from testing::Test.
class LogfsTestRaw : public testing::Test {
protected:
  virtual void SetUp() {
    /* create an empty, appropriately sized flash filesystem */
    FILE * theflash = fopen("theflash.bin", "w");
    uint8_t sector[flash_config.size_of_sector];
    memset(sector, 0xFF, sizeof(sector));
    for (uint32_t i = 0; i < flash_config.size_of_flash / flash_config.size_of_sector; i++) {
      fwrite(sector, sizeof(sector), 1, theflash);
    }
    fclose(theflash);

    /* Set up obj1 */
    for (uint32_t i = 0; i < sizeof(obj1); i++) {
      obj1[i] = 0x10 + (i % 10);
    }

    /* Set up a second version of obj1 with different data */
    for (uint32_t i = 0; i < sizeof(obj1_alt); i++) {
      obj1_alt[i] = 0xA0 + (i % 10);
    }

    /* Set up obj2 */
    for (uint32_t i = 0; i < sizeof(obj2); i++) {
      obj2[i] = 0x20 + (i % 10);
    }

    /* Set up obj3 */
    for (uint32_t i = 0; i < sizeof(obj3); i++) {
      obj3[i] = 0x30 + (i % 10);
    }

    /* Set up obj4 */
    for (uint32_t i = 0; i < sizeof(obj4); i++) {
      obj4[i] = 0x40 + (i % 10);
    }
  }

  virtual void TearDown() {
    unlink("theflash.bin");
  }

  unsigned char obj1[OBJ1_SIZE];
  unsigned char obj1_alt[OBJ1_SIZE];
  unsigned char obj2[OBJ2_SIZE];
  unsigned char obj3[OBJ3_SIZE];
  unsigned char obj4[OBJ4_SIZE];
};

TEST_F(LogfsTestRaw, FlashInit) {
  EXPECT_EQ(0, PIOS_Flash_Posix_Init(&pios_posix_flash_id, &flash_config));

  PIOS_Flash_Posix_Destroy(pios_posix_flash_id);
}

TEST_F(LogfsTestRaw, LogfsInit) {
  EXPECT_EQ(0, PIOS_Flash_Posix_Init(&pios_posix_flash_id, &flash_config));

  /* Register the partition table */
  PIOS_FLASH_register_partition_table(pios_flash_partition_table, pios_flash_partition_table_size);

  uintptr_t fs_id;
  EXPECT_EQ(0, PIOS_FLASHFS_Logfs_Init(&fs_id, &flashfs_config_settings, FLASH_PARTITION_LABEL_SETTINGS));

  PIOS_FLASHFS_Logfs_Destroy(fs_id);
  PIOS_Flash_Posix_Destroy(pios_posix_flash_id);
}

class LogfsTestCooked : public LogfsTestRaw {
protected:
  virtual void SetUp() {
    /* First, we need to set up the super fixture (LogfsTestRaw) */
    LogfsTestRaw::SetUp();

    /* Init the flash and the flashfs so we don't need to repeat this in every test */
    EXPECT_EQ(0, PIOS_Flash_Posix_Init(&pios_posix_flash_id, &flash_config));
    EXPECT_EQ(0, PIOS_FLASHFS_Logfs_Init(&fs_id, &flashfs_config_settings, FLASH_PARTITION_LABEL_SETTINGS));
  }

  virtual void TearDown() {
    PIOS_FLASHFS_Logfs_Destroy(fs_id);
    PIOS_Flash_Posix_Destroy(pios_posix_flash_id);
  }

  uintptr_t fs_id;
};

TEST_F(LogfsTestCooked, BadIdLogfsFormat) {
  EXPECT_EQ(-1, PIOS_FLASHFS_Format(fs_id + 1));
}

TEST_F(LogfsTestCooked, BadIdSave) {
  EXPECT_EQ(-1, PIOS_FLASHFS_ObjSave(fs_id + 1, OBJ1_ID, 0, obj1, sizeof(obj1)));
}

TEST_F(LogfsTestCooked, BadIdLoad) {
  unsigned char obj1_check[OBJ1_SIZE];
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(-1, PIOS_FLASHFS_ObjLoad(fs_id + 1, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));
}

TEST_F(LogfsTestCooked, LogfsFormat) {
  EXPECT_EQ(0, PIOS_FLASHFS_Format(fs_id));
}

TEST_F(LogfsTestCooked, WriteOne) {
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ1_ID, 0, obj1, sizeof(obj1)));
}

TEST_F(LogfsTestCooked, WriteVerifyOne) {
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ1_ID, 0, obj1, sizeof(obj1)));

  unsigned char obj1_check[OBJ1_SIZE];
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));
  EXPECT_EQ(0, memcmp(obj1, obj1_check, sizeof(obj1)));
}

TEST_F(LogfsTestCooked, WriteVerifyDeleteVerifyOne) {
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ1_ID, 0, obj1, sizeof(obj1)));

  unsigned char obj1_check[OBJ1_SIZE];
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));
  EXPECT_EQ(0, memcmp(obj1, obj1_check, sizeof(obj1)));

  EXPECT_EQ(0, PIOS_FLASHFS_ObjDelete(fs_id, OBJ1_ID, 0));

  EXPECT_EQ(-3, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));
}

TEST_F(LogfsTestCooked, WriteTwoVerifyOneA) {
  /* Write obj1 then obj2 */
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ1_ID, 0, obj1, sizeof(obj1)));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ2_ID, 0, obj2, sizeof(obj2)));

  /* Read back obj1 */
  unsigned char obj1_check[OBJ1_SIZE];
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));
  EXPECT_EQ(0, memcmp(obj1, obj1_check, sizeof(obj1)));
}

TEST_F(LogfsTestCooked, WriteTwoVerifyOneB) {
  /* Write obj2 then obj1 */
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ2_ID, 0, obj2, sizeof(obj2)));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ1_ID, 0, obj1, sizeof(obj1)));

  /* Read back obj1 */
  unsigned char obj1_check[OBJ1_SIZE];
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));
  EXPECT_EQ(0, memcmp(obj1, obj1_check, sizeof(obj1)));
}

TEST_F(LogfsTestCooked, WriteZeroSize) {
  /* Write a zero length object */
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ0_ID, 0, NULL, 0));
}

TEST_F(LogfsTestCooked, WriteVerifyZeroLength) {
  /* Write a zero length object */
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ0_ID, 0, NULL, 0));

  /* Read back a zero length object -- effectively an existence check */
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ0_ID, 0, NULL, 0));
}

TEST_F(LogfsTestCooked, WriteMaxSize) {
  /* Write a max length object */
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ3_ID, 0, obj3, sizeof(obj3)));
}

TEST_F(LogfsTestCooked, ReadNonexistent) {
  /* Read back a zero length object -- basically an existence check */
  unsigned char obj1_check[OBJ1_SIZE];
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(-3, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));
}

TEST_F(LogfsTestCooked, WriteVerifyMultiInstance) {
  /* Write instance zero */
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ1_ID, 0, obj1, sizeof(obj1)));

  /* Write a non-zero instance ID */
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ1_ID, 123, obj1_alt, sizeof(obj1_alt)));

  unsigned char obj1_check[OBJ1_SIZE];

  /* Read back instance 123 */
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 123, obj1_check, sizeof(obj1_check)));
  EXPECT_EQ(0, memcmp(obj1_alt, obj1_check, sizeof(obj1_alt)));

  /* Read back instance 0 */
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));
  EXPECT_EQ(0, memcmp(obj1, obj1_check, sizeof(obj1)));
}

TEST_F(LogfsTestCooked, FillFilesystemAndGarbageCollect) {
  /* Fill up the entire filesystem with multiple instances of obj1 */
  for (uint32_t i = 0; i < (flashfs_config_settings.arena_size / flashfs_config_settings.slot_size) - 1; i++) {
    EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ1_ID, i, obj1, sizeof(obj1)));
  }

  /* Should fail to add a new object since the filesystem is full */
  EXPECT_EQ(-4, PIOS_FLASHFS_ObjSave(fs_id, OBJ2_ID, 0, obj2, sizeof(obj2)));

  /* Now save a new version of an existing object which should trigger gc and succeed */
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ1_ID, 0, obj1_alt, sizeof(obj1_alt)));

  /* Read back one of the original obj1 instances */
  unsigned char obj1_check[OBJ1_SIZE];
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 1, obj1_check, sizeof(obj1_check)));
  EXPECT_EQ(0, memcmp(obj1, obj1_check, sizeof(obj1)));

  /* Read back the new version of obj1 written after gc */
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));
  EXPECT_EQ(0, memcmp(obj1_alt, obj1_check, sizeof(obj1_alt)));
}

TEST_F(LogfsTestCooked, WriteManyVerify) {
  for (uint32_t i = 0; i < 10000; i++) {
    /* Write a collection of objects */
    EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ0_ID, 0, NULL, 0));
    EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ1_ID, 0, obj1, sizeof(obj1)));
    EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ1_ID, 123, obj1_alt, sizeof(obj1_alt)));
    EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ2_ID, 0, obj2, sizeof(obj2)));
    EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id, OBJ3_ID, 0, obj3, sizeof(obj3)));
  }

  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ0_ID, 0, NULL, 0));

  unsigned char obj1_check[OBJ1_SIZE];
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));
  EXPECT_EQ(0, memcmp(obj1, obj1_check, sizeof(obj1)));

  unsigned char obj1_alt_check[OBJ1_SIZE];
  memset(obj1_alt_check, 0, sizeof(obj1_alt_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ1_ID, 123, obj1_alt_check, sizeof(obj1_alt_check)));
  EXPECT_EQ(0, memcmp(obj1_alt, obj1_alt_check, sizeof(obj1_alt)));

  unsigned char obj2_check[OBJ2_SIZE];
  memset(obj2_check, 0, sizeof(obj2_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ2_ID, 0, obj2_check, sizeof(obj2_check)));
  EXPECT_EQ(0, memcmp(obj2, obj2_check, sizeof(obj2)));

  unsigned char obj3_check[OBJ3_SIZE];
  memset(obj3_check, 0, sizeof(obj3_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id, OBJ3_ID, 0, obj3_check, sizeof(obj3_check)));
  EXPECT_EQ(0, memcmp(obj3, obj3_check, sizeof(obj3)));
}

class LogfsTestCookedMultiPart : public LogfsTestRaw {
protected:
  virtual void SetUp() {
    /* First, we need to set up the super fixture (LogfsTestRaw) */
    LogfsTestRaw::SetUp();

    /* Init the flash and the flashfs so we don't need to repeat this in every test */
    EXPECT_EQ(0, PIOS_Flash_Posix_Init(&pios_posix_flash_id, &flash_config));
    EXPECT_EQ(0, PIOS_FLASHFS_Logfs_Init(&fs_id_a, &flashfs_config_settings, FLASH_PARTITION_LABEL_SETTINGS));
    EXPECT_EQ(0, PIOS_FLASHFS_Logfs_Init(&fs_id_b, &flashfs_config_waypoints, FLASH_PARTITION_LABEL_WAYPOINTS));
  }

  virtual void TearDown() {
    PIOS_FLASHFS_Logfs_Destroy(fs_id_b);
    PIOS_FLASHFS_Logfs_Destroy(fs_id_a);
    PIOS_Flash_Posix_Destroy(pios_posix_flash_id);
  }

  uintptr_t fs_id_a;
  uintptr_t fs_id_b;
};

TEST_F(LogfsTestCookedMultiPart, WriteOneWriteOneVerify) {
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id_a, OBJ1_ID, 0, obj1, sizeof(obj1)));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id_b, OBJ2_ID, 0, obj2, sizeof(obj2)));

  /* OBJ1 found in A */
  unsigned char obj1_check[OBJ1_SIZE];
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id_a, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));
  EXPECT_EQ(0, memcmp(obj1, obj1_check, sizeof(obj1)));

  /* OBJ2 found in B */
  unsigned char obj2_check[OBJ2_SIZE];
  memset(obj2_check, 0, sizeof(obj2_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id_b, OBJ2_ID, 0, obj2_check, sizeof(obj2_check)));
  EXPECT_EQ(0, memcmp(obj2, obj2_check, sizeof(obj2)));
}

TEST_F(LogfsTestCookedMultiPart, WriteOneWriteOneFormatOneVerify) {
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id_a, OBJ1_ID, 0, obj1, sizeof(obj1)));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id_b, OBJ2_ID, 0, obj2, sizeof(obj2)));

  EXPECT_EQ(0, PIOS_FLASHFS_Format(fs_id_a));

  /* OBJ2 still found in B */
  unsigned char obj2_check[OBJ2_SIZE];
  memset(obj2_check, 0, sizeof(obj2_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id_b, OBJ2_ID, 0, obj2_check, sizeof(obj2_check)));
  EXPECT_EQ(0, memcmp(obj2, obj2_check, sizeof(obj2)));
}

TEST_F(LogfsTestCookedMultiPart, WriteOneWriteOneNoCross) {
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id_a, OBJ1_ID, 0, obj1, sizeof(obj1)));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id_b, OBJ2_ID, 0, obj2, sizeof(obj2)));

  /* OBJ1 not found in B */
  unsigned char obj1_check[OBJ1_SIZE];
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(-3, PIOS_FLASHFS_ObjLoad(fs_id_b, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));

  /* OBJ2 not found in A */
  unsigned char obj2_check[OBJ2_SIZE];
  memset(obj2_check, 0, sizeof(obj2_check));
  EXPECT_EQ(-3, PIOS_FLASHFS_ObjLoad(fs_id_a, OBJ2_ID, 0, obj2_check, sizeof(obj2_check)));
}

TEST_F(LogfsTestCookedMultiPart, WriteOneWriteOneDeleteOne) {
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id_a, OBJ1_ID, 0, obj1, sizeof(obj1)));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id_b, OBJ2_ID, 0, obj2, sizeof(obj2)));

  EXPECT_EQ(0, PIOS_FLASHFS_ObjDelete(fs_id_a, OBJ1_ID, 0));

  /* OBJ1 not found in A */
  unsigned char obj1_check[OBJ1_SIZE];
  memset(obj1_check, 0, sizeof(obj1_check));
  EXPECT_EQ(-3, PIOS_FLASHFS_ObjLoad(fs_id_a, OBJ1_ID, 0, obj1_check, sizeof(obj1_check)));

  /* OBJ2 still found in B */
  unsigned char obj2_check[OBJ2_SIZE];
  memset(obj2_check, 0, sizeof(obj2_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id_b, OBJ2_ID, 0, obj2_check, sizeof(obj2_check)));
}

TEST_F(LogfsTestCookedMultiPart, WriteLarge) {
  EXPECT_EQ(0, PIOS_FLASHFS_ObjSave(fs_id_b, OBJ4_ID, 0, obj4, sizeof(obj4)));

  /* OBJ4 found in B */
  unsigned char obj4_check[OBJ4_SIZE];
  memset(obj4_check, 0, sizeof(obj4_check));
  EXPECT_EQ(0, PIOS_FLASHFS_ObjLoad(fs_id_b, OBJ4_ID, 0, obj4_check, sizeof(obj4_check)));
}
