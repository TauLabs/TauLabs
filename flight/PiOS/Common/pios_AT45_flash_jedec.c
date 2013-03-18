/**
 ******************************************************************************
 *
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_FLASH Flash device handler
 * @{
 *
 * @file       pios_AT45_flash_jedec.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Virtual Robotix Network Team, http://www.virtualrobotix.com Copyright (C) 2013.
 * @brief      Driver for talking to AT45DB161D flash chip (and most JEDEC chips)
 * @see        The GNU Public License (GPL) Version 3
 *
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
#include "pios.h"

// AT45DB161D Commands (from Datasheet)
#define JEDEC_PAGE_TO_BUFFER_1   0x53
#define JEDEC_PAGE_TO_BUFFER_2   0x55
#define JEDEC_BUFFER_1_TO_PAGE_WITH_ERASE   0x83
#define JEDEC_BUFFER_2_TO_PAGE_WITH_ERASE   0x86
#define JEDEC_READ_STATUS            0xD7 //
#define JEDEC_DEVICE_ID              0x9F //
#define JEDEC_PAGE_WRITE             0x02
#define JEDEC_BUFFER1_WRITE          0x84 //
#define JEDEC_BUFFER2_WRITE          0x87 //
#define JEDEC_BUFFER1_READ           0xD4 //
#define JEDEC_BUFFER2_READ           0xD6 //
#define JEDEC_BUFFER1_MM_PROGRAM     0x83 //
#define JEDEC_BUFFER2_MM_PROGRAM     0x86 //

#define AT45DB161D_CHIP_ERASE_0   	0xC7 //
#define AT45DB161D_CHIP_ERASE_1   	0x94 //
#define AT45DB161D_CHIP_ERASE_2   	0x80 //
#define AT45DB161D_CHIP_ERASE_3   	0x9A //

#define JEDEC_STATUS_BUSY            0x80 //

static uint8_t device_type;

enum pios_jedec_dev_magic {
	PIOS_JEDEC_DEV_MAGIC = 0xcb55aa55,
};

//! Device handle structure
struct jedec_flash_dev {
	uint32_t spi_id;
	uint32_t slave_num;
	bool claimed;
	uint32_t device_type;
	uint32_t capacity;
	const struct pios_flash_jedec_cfg * cfg;
#if defined(FLASH_FREERTOS)
	xSemaphoreHandle transaction_lock;
#endif
	enum pios_jedec_dev_magic magic;
};

//! Global structure for this flash device
struct jedec_flash_dev * flash_dev;

//! Private functions
static int32_t PIOS_Flash_Jedec_Validate(struct jedec_flash_dev * dev);
static struct jedec_flash_dev * PIOS_Flash_Jedec_alloc(void);
static int32_t PIOS_Flash_Jedec_ClaimBus();
static int32_t PIOS_Flash_Jedec_ReleaseBus();
static int32_t PIOS_Flash_Jedec_Busy() ;


/**
 * @brief Allocate a new device
 */
static struct jedec_flash_dev * PIOS_Flash_Jedec_alloc(void)
{
	struct jedec_flash_dev * jedec_dev;
	
	jedec_dev = (struct jedec_flash_dev *)pvPortMalloc(sizeof(*jedec_dev));
	if (!jedec_dev) return (NULL);
	
	jedec_dev->claimed = false;
	jedec_dev->magic = PIOS_JEDEC_DEV_MAGIC;
#if defined(FLASH_FREERTOS)
	jedec_dev->transaction_lock = xSemaphoreCreateMutex();
#endif
	return(jedec_dev);
}

/**
 * @brief Validate the handle to the spi device
 */
static int32_t PIOS_Flash_Jedec_Validate(struct jedec_flash_dev * dev) {
	if (dev == NULL) 
		return -1;
	if (dev->magic != PIOS_JEDEC_DEV_MAGIC)
		return -2;
	if (dev->spi_id == 0)
		return -3;
	return 0;
}

/**
 * @brief Claim the SPI bus for flash use and assert CS pin
 * @return 0 for sucess, -1 for failure to get semaphore
 */
static int32_t PIOS_Flash_Jedec_ClaimBus()
{
	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if(PIOS_SPI_ClaimBus(flash_dev->spi_id) < 0)
		return -1;
		
	PIOS_SPI_RC_PinSet(flash_dev->spi_id, flash_dev->slave_num, 0);
	flash_dev->claimed = true;
	
	return 0;
}

/**
 * @brief Release the SPI bus sempahore and ensure flash chip not using bus
 */
static int32_t PIOS_Flash_Jedec_ReleaseBus()
{
	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;
	PIOS_SPI_RC_PinSet(flash_dev->spi_id, flash_dev->slave_num, 1);
	PIOS_SPI_ReleaseBus(flash_dev->spi_id);
	flash_dev->claimed = false;
	return 0;
}

/**
 * @brief Returns if the flash chip is busy
 * @returns -1 for failure, 0 for not busy, 1 for busy
 */
/*****************************************************************************/
/*Status Register Format:                                   */
/* ------------------------------------------------------------------------- */
/* | bit7   | bit6   | bit5   | bit4   | bit3   | bit2   | bit1   | bit0   | */
/* |--------|--------|--------|--------|--------|--------|--------|--------| */
/* |RDY/BUSY| COMP   |         device density            |   X    |   X    | */
/* ------------------------------------------------------------------------- */
/* 0:busy   |        |        AT45DB041:0111             | protect|page size */
/* 1:ready  |        |        AT45DB161:1011             |                   */
/* --------------------------------------------------------------------------*/
static int32_t PIOS_Flash_Jedec_Busy()
{
	int32_t status = PIOS_Flash_Jedec_ReadStatus();
	if (status < 0)
		return -1;
	return status & JEDEC_STATUS_BUSY;  // 0x80
}

/**
 * @brief Initialize the flash device and enable write access
 */
int32_t PIOS_Flash_Jedec_Init(uint32_t spi_id, uint32_t slave_num, const struct pios_flash_jedec_cfg * cfg)
{
	flash_dev = PIOS_Flash_Jedec_alloc();
	if(flash_dev == NULL)
		return -1;

	flash_dev->spi_id = spi_id;
	flash_dev->slave_num = slave_num;
	flash_dev->cfg = cfg;

	device_type = PIOS_Flash_Jedec_ReadID(); // 1f 26 00 00

	  // get page size: 512 or 528
//	  df_PageSize=PIOS_Flash_Jedec_PageSize();
	return 0;
}

/**
 * @brief Grab the semaphore to perform a transaction
 * @return 0 for success, -1 for timeout
 */
int32_t PIOS_Flash_Jedec_StartTransaction()
{
#if defined(FLASH_FREERTOS)
	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if(xSemaphoreTake(flash_dev->transaction_lock, portMAX_DELAY) != pdTRUE)
		return -1;
#endif
	return 0;
}

/**
 * @brief Release the semaphore to perform a transaction
 * @return 0 for success, -1 for timeout
 */
int32_t PIOS_Flash_Jedec_EndTransaction()
{
#if defined(FLASH_FREERTOS)
	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if(xSemaphoreGive(flash_dev->transaction_lock) != pdTRUE)
		return -1;
#endif
	return 0;
}

/**
 * @brief Read the status register from flash chip and return it
 */
/*****************************************************************************/
/*Status Register Format:                                   */
/* ------------------------------------------------------------------------- */
/* | bit7   | bit6   | bit5   | bit4   | bit3   | bit2   | bit1   | bit0   | */
/* |--------|--------|--------|--------|--------|--------|--------|--------| */
/* |RDY/BUSY| COMP   |         device density            |   X    |   X    | */
/* ------------------------------------------------------------------------- */
/* 0:busy   |        |        AT45DB041:0111             | protect|page size */
/* 1:ready  |        |        AT45DB161:1011             |                   */
/* --------------------------------------------------------------------------*/
int32_t PIOS_Flash_Jedec_ReadStatus()
{
	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	uint8_t out[] = {JEDEC_READ_STATUS, 0x00};  //0xd7
	uint8_t in[] = {0,0};
	if(PIOS_Flash_Jedec_ClaimBus() < 0)
		return -2;

	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out,in,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -3;
	}

	PIOS_Flash_Jedec_ReleaseBus();
	return in[1];
}

/**
 * @brief Read 4 byte the flash ID and return Manufacturer
 */
int32_t PIOS_Flash_Jedec_ReadID()
{
	uint8_t out[] = {JEDEC_DEVICE_ID}; // 0x9f
	uint8_t in[4];
	if (PIOS_Flash_Jedec_ClaimBus() < 0) 
		return -1;
	
	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -2;
	}

	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,NULL,in,sizeof(in),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -2;
	}
	PIOS_Flash_Jedec_ReleaseBus();
	
	flash_dev->device_type = in[0];  // 0x1f - Manufacturer ID Atmel
	flash_dev->capacity = in[1];	 // 0x26 00100110b (001-dataflash  00110 - Density 16mB)
	return in[0];
}


/**
 * @brief Erase a block on the flash chip 4kb
 * @param[in] add Address of flash to erase
 * @returns 0 if successful
 * @retval -1 if unable to claim bus
 * @retval
 */
int32_t PIOS_Flash_Jedec_ErasePage(uint32_t addr)
{
	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	//AT45_page_earse
	uint8_t out[] = {0x81, (uint8_t)(addr >> 6) , (uint8_t)(addr << 2) , 0x00 };


	if(PIOS_Flash_Jedec_ClaimBus() != 0)
		return -1;
	
	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -2;
	}
	
	PIOS_Flash_Jedec_ReleaseBus();

	// Keep polling when bus is busy too
	while(!PIOS_Flash_Jedec_Busy()) {
#if defined(FLASH_FREERTOS)
		vTaskDelay(1);
#endif
	}
	return 0;
}

/**
 * @brief Execute the whole chip
 * @returns 0 if successful, -1 if unable to claim bus
 */
int32_t PIOS_Flash_Jedec_EraseChip()
{

	PIOS_LED_On(PIOS_LED_YELLOW);

	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	uint8_t out[] = {AT45DB161D_CHIP_ERASE_0, AT45DB161D_CHIP_ERASE_1, AT45DB161D_CHIP_ERASE_2, AT45DB161D_CHIP_ERASE_3 };

	if(PIOS_Flash_Jedec_ClaimBus() != 0)
		return -1;
	
	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -2;
	}
	
	PIOS_Flash_Jedec_ReleaseBus();

	// Keep polling when bus is busy too
	int i = 0;
	while(!PIOS_Flash_Jedec_Busy()) {
#if defined(FLASH_FREERTOS)
		vTaskDelay(1);
		if ((i++) % 100 == 0)
#else
		if ((i++) % 10000 == 0)
#endif
		PIOS_LED_Toggle(PIOS_LED_ALARM);
	}

	PIOS_LED_Off(PIOS_LED_YELLOW);
	return 0;
}

/*
int32_t PIOS_Flash_Jedec_WriteData(uint32_t addr, uint8_t * data, uint16_t len)
{
// AT45_RamTo_buf
	PIOS_Flash_Jedec_BufferWrite (1, 0, data, len);

// AT45_bufTo_ROM
	PIOS_Flash_Jedec_BufferToPage(1, addr);

	return 0;
}
*/

/**
 * @brief Write one page of data (up to 512 bytes) aligned to a page start
 * @param[in] addr Address in flash to write to
 * @param[in] data Pointer to data to write to flash
 * @param[in] len Length of data to write (max 512 bytes)
 * @return Zero if success or error code
 * @retval -1 Unable to claim SPI bus
 */

int32_t PIOS_Flash_Jedec_WritePage(uint32_t page, uint16_t offset, uint8_t * data, uint16_t len)
{
	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if(PIOS_Flash_Jedec_ClaimBus() != 0)
		return -1;

	uint8_t out[] = {0x82, (uint8_t)(page >> 6) , (uint8_t)((page << 2) | (offset >> 8)), (uint8_t)(offset)};

	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -1;
	}

	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,data,NULL,len,NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -1;
	}

	PIOS_Flash_Jedec_ReleaseBus();

// Keep polling when bus is busy too
	while(!PIOS_Flash_Jedec_Busy()) {
#if defined(FLASH_FREERTOS)
		vTaskDelay(1);
#endif
	}
	return 0;
}
/**
 * @brief Write multiple chunks of data in one transaction
 * @param[in] addr Address in flash to write to
 * @param[in] data Pointer to data to write to flash
 * @param[in] len Length of data to write (max 512 bytes)
 * @return Zero if success or error code
 * @retval -1 Unable to claim SPI bus
 * @retval -2 Size exceeds 512 bytes
 */
int32_t PIOS_Flash_Jedec_WriteChunks(uint32_t addr, struct pios_flash_chunk * p_chunk, uint32_t num)
{

	/* Can only write one page at a time */
	uint32_t len = 0;
	for(uint32_t i = 0; i < num; i++)
		len += p_chunk[i].len;

	if(len > 0x200)
		return -2;
	
	/* Ensure number of bytes fits after starting address before end of page */
	if(((addr & 0x1ff) + len) > 0x200)
		return -3;

	uint32_t buffOffset = 0;
	for(uint32_t i = 0; i < num; i++) {
		struct pios_flash_chunk * chunk = &p_chunk[i];

		/* Clock out data to buffer */
		PIOS_Flash_Jedec_BufferWrite (1, buffOffset  , chunk->addr, chunk->len);
		// next buffer address
		buffOffset += chunk->len;
	}

	// AT45 write buffer(x) to Page
	PIOS_Flash_Jedec_BufferToPage(1, addr);  // write buffer 1 to page(addr)

	return 0;
}

/**
 * @brief Read data from a location in flash memory
 * @param[in] addr Address in flash to write to
 * @param[in] data Pointer to data to write from flash
 * @param[in] len Length of data to write (max 256 bytes)
 * @return Zero if success or error code
 * @retval -1 Unable to claim SPI bus
 */
int32_t PIOS_Flash_Jedec_ReadData(uint32_t addr, uint8_t * data, uint16_t len)
{
	uint8_t out1[5];

	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if(PIOS_Flash_Jedec_ClaimBus() == -1)
		return -1;

	uint8_t out[] = {JEDEC_PAGE_TO_BUFFER_1, (addr >> 6) & 0xff, (addr << 2) & 0xff , 0x00};

	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -2;
	}
	PIOS_Flash_Jedec_ReleaseBus();

// Keep polling when bus is busy too
while(!PIOS_Flash_Jedec_Busy()) {
#if defined(FLASH_FREERTOS)
	vTaskDelay(1);
#endif
}
	// Buffer Read
	if(PIOS_Flash_Jedec_ClaimBus() == -1)
		return -1;

	  out1[0] = JEDEC_BUFFER1_READ;  // 0xd4
	  out1[1] = 0x00;
	  out1[2] = 0x00;
	  out1[3] = 0x00;
	  out1[4] = 0x00;

		if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out1,NULL,sizeof(out1),NULL) < 0) {
			PIOS_Flash_Jedec_ReleaseBus();
			return -2;
		}

		if(PIOS_SPI_TransferBlock(flash_dev->spi_id,NULL,data,len,NULL) < 0) {
			PIOS_Flash_Jedec_ReleaseBus();
			return -1;
		}

		PIOS_Flash_Jedec_ReleaseBus();
		// End Buffer READ

	return 0;
}

int32_t PIOS_Flash_Jedec_ReadPage(uint32_t page, uint16_t offset, uint8_t * data, uint16_t len)
{
	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0){
		PIOS_LED_On(PIOS_LED_ALARM);
		return -1;
}
	if(PIOS_Flash_Jedec_ClaimBus() == -1)
		return -1;

	uint8_t out[] = {0xD2, (uint8_t)(page >> 6) , (uint8_t)((page << 2) | (offset >> 8)), (uint8_t)(offset), 0x00, 0x00, 0x00, 0x00};
	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -2;
	}

// Read page
	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,NULL,data,len,NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -3;
	}

	PIOS_Flash_Jedec_ReleaseBus();
	// End Buffer READ
	return 0;
}

int32_t PIOS_Flash_Jedec_PageSize()
{
  return(528-((PIOS_Flash_Jedec_ReadStatus() & 0x01)<<4));  // if first bit 1 trhen 512 else 528 bytes
}

//
int32_t PIOS_Flash_Jedec_PageToBuffer(unsigned char BufferNum, uint16_t PageAdr)
{
	uint8_t out[4];

	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if(PIOS_Flash_Jedec_ClaimBus() == -1)
		return -1;

  if (BufferNum==1)
	  out[0] = JEDEC_PAGE_TO_BUFFER_1;
  else
	  out[0] = JEDEC_PAGE_TO_BUFFER_2;

  out[1] = (unsigned char)(PageAdr >> 6);
  out[2] = (unsigned char)(PageAdr << 2);
  out[3] = 0x00;

	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -2;
	}
	PIOS_Flash_Jedec_ReleaseBus();

	// Keep polling when bus is busy too
	while(!PIOS_Flash_Jedec_Busy()) {
#if defined(FLASH_FREERTOS)
		vTaskDelay(1);
#endif
	}

	return 0;
}
// Write buffer 1/2 (512-528) to Page 0-4095
int32_t PIOS_Flash_Jedec_BufferToPage(unsigned char BufferNum, uint16_t PageAdr)
{
	uint8_t out[4];

	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if(PIOS_Flash_Jedec_ClaimBus() == -1)
		return -1;

  if (BufferNum==1)
	  out[0] = JEDEC_BUFFER_1_TO_PAGE_WITH_ERASE;
  else
	  out[0] = JEDEC_BUFFER_2_TO_PAGE_WITH_ERASE;

/*
  if(df_PageSize==512){
	  out[1] = (unsigned char)(PageAdr >> 7);
	  out[2] = (unsigned char)(PageAdr << 1);
  }else{
	  out[1] = (unsigned char)(PageAdr >> 6);
	  out[2] = (unsigned char)(PageAdr << 2);
  }
  */
  out[1] = (unsigned char)(PageAdr >> 6);
  out[2] = (unsigned char)(PageAdr << 2);
  out[3] = 0x00;

	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -2;
	}
	PIOS_Flash_Jedec_ReleaseBus();

	// Keep polling when bus is busy too
	while(!PIOS_Flash_Jedec_Busy()) {
#if defined(FLASH_FREERTOS)
		vTaskDelay(1);
#endif
	}
	return 0;
}

int32_t PIOS_Flash_Jedec_BufferWrite (unsigned char BufferNum, uint16_t IntPageAdr,  uint8_t * data, uint16_t len)
{
	uint8_t out[4];

	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if(PIOS_Flash_Jedec_ClaimBus() == -1)
		return -1;

  if (BufferNum==1)
	  out[0] = JEDEC_BUFFER1_WRITE;
  else
	  out[0] = JEDEC_BUFFER2_WRITE;

  	  out[1] = 0x00;
	  out[2] = (unsigned char)(IntPageAdr >> 8);
	  out[3] = (unsigned char)(IntPageAdr );


	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -2;
	}

	/* Clock out data to flash */
	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,data,NULL,len,NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -3;
	}

	PIOS_Flash_Jedec_ReleaseBus();

	// Keep polling when bus is busy too
	while(!PIOS_Flash_Jedec_Busy()) {
#if defined(FLASH_FREERTOS)
		vTaskDelay(1);
#endif
	}
	return 0;
}

int32_t PIOS_Flash_Jedec_BufferRead (unsigned char BufferNum, uint16_t IntBufferAdr,  uint8_t * data, uint16_t len)
{
	uint8_t out[5];

	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if(PIOS_Flash_Jedec_ClaimBus() == -1)
		return -1;

  if (BufferNum==1)
	  out[0] = JEDEC_BUFFER1_READ;  // 0xd4
  else
	  out[0] = JEDEC_BUFFER2_READ;

  	  out[1] = 0x00;
	  out[2] = (unsigned char)(IntBufferAdr >> 8);
	  out[3] = (unsigned char)(IntBufferAdr );
	  out[4] = 0x00;

	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -2;
	}

	/* Clock out data to flash */
	if(PIOS_SPI_TransferBlock(flash_dev->spi_id,NULL,data,len,NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus();
		return -1;
	}

	PIOS_Flash_Jedec_ReleaseBus();
	return 0;
}


