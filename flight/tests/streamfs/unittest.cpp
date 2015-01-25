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

#include "pios_flash.h"		/* PIOS_FLASH_* API */
#include "pios_com.h"
#include "pios_com_priv.h"
#include "pios_flash_priv.h"	/* struct pios_flash_partition */

extern const struct pios_flash_partition pios_flash_partition_table[];
extern uint32_t pios_flash_partition_table_size;

#include "pios_flash_posix_priv.h"

extern uintptr_t pios_posix_flash_id;
extern struct pios_flash_posix_cfg flash_config;

#include "pios_streamfs_priv.h"
#include "pios_streamfs.h"

extern struct streamfs_cfg streamfs_settings;

// Methods to use for testing
int32_t PIOS_STREAMFS_Testing_Write(uintptr_t fs_id, uint8_t *data, uint32_t len);
int32_t PIOS_STREAMFS_Testing_Read(uintptr_t fs_id, uint8_t *data, uint32_t len);

// Define this method just to prevent warnings
int32_t PIOS_DELAY_WaitmS(uint32_t mS) {
  return mS;
}

}

// To use a test fixture, derive a class from testing::Test.
class StreamfsTestRaw : public testing::Test {
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
  }

  virtual void TearDown() {
    unlink("theflash.bin");
  }
};

TEST_F(StreamfsTestRaw, FlashInit) {
  EXPECT_EQ(0, PIOS_Flash_Posix_Init(&pios_posix_flash_id, &flash_config));

  PIOS_Flash_Posix_Destroy(pios_posix_flash_id);
}

TEST_F(StreamfsTestRaw, StreamfsInit) {
  EXPECT_EQ(0, PIOS_Flash_Posix_Init(&pios_posix_flash_id, &flash_config));

  /* Register the partition table */
  PIOS_FLASH_register_partition_table(pios_flash_partition_table, pios_flash_partition_table_size);

  uintptr_t fs_id;
  EXPECT_EQ(0, PIOS_STREAMFS_Init(&fs_id, &streamfs_settings, FLASH_PARTITION_LABEL_SETTINGS));

  PIOS_STREAMFS_Destroy(fs_id);
  PIOS_Flash_Posix_Destroy(pios_posix_flash_id);
}

TEST_F(StreamfsTestRaw, StreamfsFormat) {
  EXPECT_EQ(0, PIOS_Flash_Posix_Init(&pios_posix_flash_id, &flash_config));

  /* Register the partition table */
  PIOS_FLASH_register_partition_table(pios_flash_partition_table, pios_flash_partition_table_size);

  uintptr_t fs_id;
  EXPECT_EQ(0, PIOS_STREAMFS_Init(&fs_id, &streamfs_settings, FLASH_PARTITION_LABEL_SETTINGS));

  EXPECT_EQ(0, PIOS_STREAMFS_Format(fs_id));

  EXPECT_EQ(-1, PIOS_STREAMFS_MaxFileId(fs_id));
  EXPECT_EQ(-1, PIOS_STREAMFS_MinFileId(fs_id));

  PIOS_STREAMFS_Destroy(fs_id);
  PIOS_Flash_Posix_Destroy(pios_posix_flash_id);
}

#define DATA_LEN 100000

class StreamfsTestCooked : public StreamfsTestRaw {
protected:
  virtual void SetUp() {
    /* First, we need to set up the super fixture (StreamfsTestRaw) */
    StreamfsTestRaw::SetUp();

    /* Init the flash and the flashfs so we don't need to repeat this in every test */
    EXPECT_EQ(0, PIOS_Flash_Posix_Init(&pios_posix_flash_id, &flash_config));

    /* Register the partition table */
    PIOS_FLASH_register_partition_table(pios_flash_partition_table, pios_flash_partition_table_size);

    EXPECT_EQ(0, PIOS_STREAMFS_Init(&fs_id, &streamfs_settings, FLASH_PARTITION_LABEL_SETTINGS));

    for (unsigned long i = 0; i < DATA_LEN; i++) {
      data1[i] = i ^ 0x37;
      data2[i] = i ^ 0x19;
    }
  }

  virtual void TearDown() {
    PIOS_STREAMFS_Destroy(fs_id);
    PIOS_Flash_Posix_Destroy(pios_posix_flash_id);
  }

  void CompareArray(uint8_t *a, uint8_t *b, int32_t size) {
    for (int32_t i = 0; i < size; i++) {
      EXPECT_EQ(a[i], b[i]);
      if (a[i] != b[i]) {
        fprintf(stderr, "Mismatch on element %d\r\n", i);
        break;
      }
    }
  }

  uintptr_t fs_id;
  uint8_t data1[DATA_LEN];
  uint8_t data2[DATA_LEN];
};

TEST_F(StreamfsTestCooked, StreamfsOpenClose) {
  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
}


TEST_F(StreamfsTestCooked, StreamfsWriting) {
  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_MaxFileId(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_MinFileId(fs_id));
}


TEST_F(StreamfsTestCooked, StreamfsReading) {
  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_MaxFileId(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_MinFileId(fs_id));

  uint8_t data_read[DATA_LEN];

  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,0));
  EXPECT_EQ(DATA_LEN, PIOS_STREAMFS_Testing_Read(fs_id, data_read, DATA_LEN));
  for (int32_t i = 0; i < DATA_LEN; i++)
    EXPECT_EQ(data1[i], data_read[i]);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
}

TEST_F(StreamfsTestCooked, StreamfsWritingMulti) {
  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(2, PIOS_STREAMFS_MaxFileId(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_MinFileId(fs_id));
}

TEST_F(StreamfsTestCooked, StreamfsReadingMulti) {
  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data2, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(2, PIOS_STREAMFS_MaxFileId(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_MinFileId(fs_id));

  uint8_t data_read[DATA_LEN];

  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,0));
  EXPECT_EQ(DATA_LEN, PIOS_STREAMFS_Testing_Read(fs_id, data_read, DATA_LEN));
  CompareArray(data1, data_read, DATA_LEN);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,1));
  EXPECT_EQ(DATA_LEN, PIOS_STREAMFS_Testing_Read(fs_id, data_read, DATA_LEN));
  CompareArray(data2, data_read, DATA_LEN);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,2));
  EXPECT_EQ(DATA_LEN, PIOS_STREAMFS_Testing_Read(fs_id, data_read, DATA_LEN));
  CompareArray(data1, data_read, DATA_LEN);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
}


class StreamfsTestUsed : public StreamfsTestCooked {
protected:
  virtual void SetUp() {
    /* First, we need to set up the super fixture (StreamfsTestRaw) */
    StreamfsTestCooked::SetUp();

    EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
    EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
    EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

    EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
    EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data2, DATA_LEN));
    EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

    EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
    EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
    EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
  }

  virtual void TearDown() {
    StreamfsTestCooked::TearDown();
  }

};

TEST_F(StreamfsTestUsed, OpenInvalid) {
  EXPECT_TRUE(PIOS_STREAMFS_OpenRead(fs_id,20) != 0);
}

TEST_F(StreamfsTestUsed, OpenMultiple) {
  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,0));
  EXPECT_TRUE(PIOS_STREAMFS_OpenRead(fs_id,1) != 0);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
  
  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,1));
  EXPECT_TRUE(PIOS_STREAMFS_OpenWrite(fs_id) != 0);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_TRUE(PIOS_STREAMFS_OpenWrite(fs_id) != 0);
  EXPECT_TRUE(PIOS_STREAMFS_OpenRead(fs_id,1) != 0);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
}

TEST_F(StreamfsTestUsed, WriteNull) {
  int32_t file_id = PIOS_STREAMFS_MaxFileId(fs_id);

  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(file_id, PIOS_STREAMFS_MaxFileId(fs_id));
}

TEST_F(StreamfsTestUsed, WriteNew) {

  EXPECT_EQ(2, PIOS_STREAMFS_MaxFileId(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  uint8_t data_read[DATA_LEN];
  int32_t file_id = PIOS_STREAMFS_MaxFileId(fs_id);
  EXPECT_EQ(3, file_id);

  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,file_id));
  EXPECT_EQ(DATA_LEN, PIOS_STREAMFS_Testing_Read(fs_id, data_read, DATA_LEN));
  CompareArray(data1, data_read, DATA_LEN);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
}


TEST_F(StreamfsTestUsed, WriteReadWriteRead) {

  EXPECT_EQ(2, PIOS_STREAMFS_MaxFileId(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  uint8_t data_read[DATA_LEN];
  int32_t file_id = PIOS_STREAMFS_MaxFileId(fs_id);
  EXPECT_EQ(3, file_id);

  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,file_id));
  EXPECT_EQ(DATA_LEN, PIOS_STREAMFS_Testing_Read(fs_id, data_read, DATA_LEN));
  CompareArray(data1, data_read, DATA_LEN);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data2, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,file_id+1));
  EXPECT_EQ(DATA_LEN, PIOS_STREAMFS_Testing_Read(fs_id, data_read, DATA_LEN));
  CompareArray(data2, data_read, DATA_LEN);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,file_id));
  EXPECT_EQ(DATA_LEN, PIOS_STREAMFS_Testing_Read(fs_id, data_read, DATA_LEN));
  CompareArray(data1, data_read, DATA_LEN);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
}

TEST_F(StreamfsTestUsed, ReadChunk) {
  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,1));
  int32_t total_read = 0;
  uint8_t data_read[DATA_LEN];
  while(total_read < DATA_LEN) {
    EXPECT_EQ(100, PIOS_STREAMFS_Testing_Read(fs_id, &data_read[total_read], 100));
    total_read += 100;
  }
  CompareArray(data2, data_read, DATA_LEN);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
}

TEST_F(StreamfsTestUsed, WriteChunk) {
  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  int32_t total_write = 0;
  while(total_write < DATA_LEN) {
    EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, &data2[total_write], 100));
    total_write += 100;
  }
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  int32_t file_id = PIOS_STREAMFS_MaxFileId(fs_id);
  uint8_t data_read[DATA_LEN];

  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,file_id));
  EXPECT_EQ(DATA_LEN, PIOS_STREAMFS_Testing_Read(fs_id, data_read, DATA_LEN));
  CompareArray(data2, data_read, DATA_LEN);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
}

TEST_F(StreamfsTestUsed, ReadLong) {
  uint8_t data_read[DATA_LEN * 2];
  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,0));
  EXPECT_EQ(DATA_LEN, PIOS_STREAMFS_Testing_Read(fs_id, data_read, DATA_LEN * 2));
  CompareArray(data1, data_read, DATA_LEN);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
}

#define BUF_LEN 50
class StreamfsComTest : public StreamfsTestCooked {
protected:
  virtual void SetUp() {
    /* First, we need to set up the super fixture (StreamfsTestRaw) */
    StreamfsTestCooked::SetUp();

    PIOS_COM_Init(&com_id, &pios_streamfs_com_driver, fs_id,
            rx_buffer, BUF_LEN,
            tx_buffer, BUF_LEN);
  }

  virtual void TearDown() {
    StreamfsTestCooked::TearDown();
  }

  uintptr_t com_id;
  uint8_t rx_buffer[BUF_LEN];
  uint8_t tx_buffer[BUF_LEN];
};

TEST_F(StreamfsComTest, ComWriteClosed) {
  uint8_t data_read[DATA_LEN];
  EXPECT_EQ(10, PIOS_COM_SendBufferNonBlocking(com_id, data_read, 10));
  EXPECT_EQ(-2, PIOS_COM_SendBufferNonBlocking(com_id, data_read, (uint16_t)DATA_LEN));
  EXPECT_EQ(-2, PIOS_COM_SendBufferNonBlocking(com_id, data_read, (uint16_t)DATA_LEN));
}

TEST_F(StreamfsComTest, ComWrite) {
  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  int32_t total_write = 0;
  while(total_write < DATA_LEN) {
    EXPECT_EQ(100, PIOS_COM_SendBuffer(com_id, &data2[total_write], 100));
    total_write += 100;
  }
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  int32_t file_id = PIOS_STREAMFS_MaxFileId(fs_id);
  uint8_t data_read[DATA_LEN];
  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,file_id));
  EXPECT_EQ(DATA_LEN, PIOS_STREAMFS_Testing_Read(fs_id, data_read, DATA_LEN));
  CompareArray(data2, data_read, DATA_LEN);
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
}

TEST_F(StreamfsComTest, ComReadClosed) {
  uint8_t data_read[DATA_LEN];
  EXPECT_EQ(0, PIOS_COM_ReceiveBuffer(com_id, data_read, (uint16_t) DATA_LEN, 0));
  EXPECT_EQ(0, PIOS_COM_ReceiveBuffer(com_id, data_read, (uint16_t)DATA_LEN, 0));
}

TEST_F(StreamfsComTest, ComRead) {
  EXPECT_EQ(0, PIOS_STREAMFS_OpenWrite(fs_id));
  EXPECT_EQ(0, PIOS_STREAMFS_Testing_Write(fs_id, data1, DATA_LEN));
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));

  uint8_t data_read[DATA_LEN];
  int32_t file_id = PIOS_STREAMFS_MaxFileId(fs_id);

  EXPECT_EQ(0, PIOS_STREAMFS_OpenRead(fs_id,file_id));
  int32_t total_read = 0;
  int32_t read = 1;
  while(read) {
    read = PIOS_COM_ReceiveBuffer(com_id, &data_read[total_read], 10, 2); 
    EXPECT_TRUE(read >= 0);
    total_read += read;
  }
  EXPECT_EQ(0, PIOS_STREAMFS_Close(fs_id));
  CompareArray(data1, data_read, DATA_LEN);
}
