/**
 ******************************************************************************
 *
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_FLASH Flash device handler
 * @{
 *
 * @file       pios_flash_w25x.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Driver for talking to W25X flash chip (and most JEDEC chips)
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

#if defined(PIOS_INCLUDE_FLASH_JEDEC)

#include "pios_flash_jedec_priv.h"
#include "pios_semaphore.h"

#define JEDEC_WRITE_ENABLE           0x06
#define JEDEC_WRITE_DISABLE          0x04
#define JEDEC_READ_STATUS            0x05
#define JEDEC_WRITE_STATUS           0x01
#define JEDEC_READ_DATA              0x03
#define JEDEC_FAST_READ              0x0b
#define JEDEC_DEVICE_ID              0x9F
#define JEDEC_PAGE_WRITE             0x02

#define JEDEC_STATUS_BUSY            0x01
#define JEDEC_STATUS_WRITEPROTECT    0x02
#define JEDEC_STATUS_BP0             0x04
#define JEDEC_STATUS_BP1             0x08
#define JEDEC_STATUS_BP2             0x10
#define JEDEC_STATUS_TP              0x20
#define JEDEC_STATUS_SEC             0x40
#define JEDEC_STATUS_SRP0            0x80

enum pios_jedec_dev_magic {
	PIOS_JEDEC_DEV_MAGIC = 0xcb55aa55,
};

//! Device handle structure
struct jedec_flash_dev {
	uint32_t spi_id;
	uint32_t slave_num;

	uint8_t manufacturer;
	uint8_t memorytype;
	uint8_t capacity;

	const struct pios_flash_jedec_cfg *cfg;
	struct pios_semaphore *transaction_lock;
	enum pios_jedec_dev_magic magic;
};

//! Private functions
static int32_t PIOS_Flash_Jedec_Validate(struct jedec_flash_dev *flash_dev);
static struct jedec_flash_dev *PIOS_Flash_Jedec_alloc(void);

static int32_t PIOS_Flash_Jedec_ReadID(struct jedec_flash_dev *flash_dev);
static int32_t PIOS_Flash_Jedec_ReadStatus(struct jedec_flash_dev *flash_dev);
static int32_t PIOS_Flash_Jedec_ClaimBus(struct jedec_flash_dev *flash_dev);
static int32_t PIOS_Flash_Jedec_ReleaseBus(struct jedec_flash_dev *flash_dev);
static int32_t PIOS_Flash_Jedec_WriteEnable(struct jedec_flash_dev *flash_dev);
static int32_t PIOS_Flash_Jedec_Busy(struct jedec_flash_dev *flash_dev);

/**
 * @brief Allocate a new device
 */
static struct jedec_flash_dev *PIOS_Flash_Jedec_alloc(void)
{
	struct jedec_flash_dev *flash_dev;

	flash_dev = (struct jedec_flash_dev *)PIOS_malloc(sizeof(*flash_dev));
	if (!flash_dev) return (NULL);

	flash_dev->magic = PIOS_JEDEC_DEV_MAGIC;

	return(flash_dev);
}

/**
 * @brief Validate the handle to the spi device
 */
static int32_t PIOS_Flash_Jedec_Validate(struct jedec_flash_dev *flash_dev) {
	if (flash_dev == NULL)
		return -1;
	if (flash_dev->magic != PIOS_JEDEC_DEV_MAGIC)
		return -2;
	if (flash_dev->spi_id == 0)
		return -3;
	return 0;
}

/**
 * @brief Initialize the flash device and enable write access
 */
int32_t PIOS_Flash_Jedec_Init(uintptr_t *chip_id, uint32_t spi_id, uint32_t slave_num, const struct pios_flash_jedec_cfg *cfg)
{
	struct jedec_flash_dev *flash_dev = PIOS_Flash_Jedec_alloc();
	if (flash_dev == NULL)
		return -1;

	flash_dev->transaction_lock = PIOS_Semaphore_Create();
	if (flash_dev->transaction_lock == NULL)
		return -1;

	flash_dev->spi_id = spi_id;
	flash_dev->slave_num = slave_num;
	flash_dev->cfg = cfg;

	(void) PIOS_Flash_Jedec_ReadID(flash_dev);
	if ((flash_dev->manufacturer != flash_dev->cfg->expect_manufacturer) ||
		(flash_dev->memorytype != flash_dev->cfg->expect_memorytype) ||
		(flash_dev->capacity != flash_dev->cfg->expect_capacity)) {
		/* Mismatched device has been discovered */
		return -1;
	}

	/* Give back a handle to this flash device */
	*chip_id = (uintptr_t) flash_dev;

	return 0;
}


/**
 * @brief Claim the SPI bus for flash use and assert CS pin
 * @return 0 for sucess, -1 for failure to get semaphore
 */
static int32_t PIOS_Flash_Jedec_ClaimBus(struct jedec_flash_dev *flash_dev)
{
	if (PIOS_SPI_ClaimBus(flash_dev->spi_id) < 0)
		return -1;

	PIOS_SPI_RC_PinSet(flash_dev->spi_id, flash_dev->slave_num, 0);

	return 0;
}

/**
 * @brief Release the SPI bus sempahore and ensure flash chip not using bus
 */
static int32_t PIOS_Flash_Jedec_ReleaseBus(struct jedec_flash_dev *flash_dev)
{
	PIOS_SPI_RC_PinSet(flash_dev->spi_id, flash_dev->slave_num, 1);
	PIOS_SPI_ReleaseBus(flash_dev->spi_id);
	return 0;
}

/**
 * @brief Returns if the flash chip is busy
 * @returns -1 for failure, 0 for not busy, 1 for busy
 */
static int32_t PIOS_Flash_Jedec_Busy(struct jedec_flash_dev *flash_dev)
{
	int32_t status = PIOS_Flash_Jedec_ReadStatus(flash_dev);
	if (status < 0)
		return -1;
	return status & JEDEC_STATUS_BUSY;
}

/**
 * @brief Execute the write enable instruction and returns the status
 * @returns 0 if successful, -1 if unable to claim bus
 */
static int32_t PIOS_Flash_Jedec_WriteEnable(struct jedec_flash_dev *flash_dev)
{
	if (PIOS_Flash_Jedec_ClaimBus(flash_dev) != 0)
		return -1;

	uint8_t out[] = {
		JEDEC_WRITE_ENABLE,
	};
	PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL);
	PIOS_Flash_Jedec_ReleaseBus(flash_dev);

	return 0;
}


/**
 * @brief Read the status register from flash chip and return it
 */
static int32_t PIOS_Flash_Jedec_ReadStatus(struct jedec_flash_dev *flash_dev)
{
	if (PIOS_Flash_Jedec_ClaimBus(flash_dev) < 0)
		return -1;

	uint8_t out[2] = {
		JEDEC_READ_STATUS,
		0,
	};
	uint8_t in[2] = {0,0};
	if (PIOS_SPI_TransferBlock(flash_dev->spi_id,out,in,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus(flash_dev);
		return -2;
	}

	PIOS_Flash_Jedec_ReleaseBus(flash_dev);

	return in[1];
}

/**
 * @brief Read the status register from flash chip and return it
 */
static int32_t PIOS_Flash_Jedec_ReadID(struct jedec_flash_dev *flash_dev)
{
	if (PIOS_Flash_Jedec_ClaimBus(flash_dev) < 0)
		return -2;

	uint8_t out[] = {
		JEDEC_DEVICE_ID,
		0,
		0,
		0,
	};
	uint8_t in[4];
	if (PIOS_SPI_TransferBlock(flash_dev->spi_id,out,in,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus(flash_dev);
		return -3;
	}

	PIOS_Flash_Jedec_ReleaseBus(flash_dev);

	flash_dev->manufacturer = in[1];
	flash_dev->memorytype   = in[2];
	flash_dev->capacity     = in[3];

	return flash_dev->manufacturer;
}

/**********************************
 *
 * Provide a PIOS flash driver API
 *
 *********************************/
#include "pios_flash_priv.h"

/**
 * @brief Grab the semaphore to perform a transaction
 * @param[in] chip_id the opaque handle for the chip that this operation should be applied to
 * @return 0 for success, -1 for timeout
 */
static int32_t PIOS_Flash_Jedec_StartTransaction(uintptr_t chip_id)
{
	struct jedec_flash_dev *flash_dev = (struct jedec_flash_dev *)chip_id;

	if (PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if (PIOS_Semaphore_Take(flash_dev->transaction_lock, PIOS_SEMAPHORE_TIMEOUT_MAX) != true)
		return -2;

	return 0;
}

/**
 * @brief Release the semaphore to perform a transaction
 * @param[in] chip_id the opaque handle for the chip that this operation should be applied to
 * @return 0 for success, -1 for timeout
 */
static int32_t PIOS_Flash_Jedec_EndTransaction(uintptr_t chip_id)
{
	struct jedec_flash_dev *flash_dev = (struct jedec_flash_dev *)chip_id;

	if (PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if (PIOS_Semaphore_Give(flash_dev->transaction_lock) != true)
		return -2;

	return 0;
}

/**
 * @brief Erase a sector on the flash chip
 * @param[in] chip_id the opaque handle for the chip that this operation should be applied to
 * @param[in] chip_sector Sector number of flash to erase
 * @param[in] chip_offset Address within flash to erase
 * @returns 0 if successful
 * @retval -1 if unable to claim bus
 * @retval
 */
static int32_t PIOS_Flash_Jedec_EraseSector(uintptr_t chip_id, uint32_t chip_sector, uint32_t chip_offset)
{
	struct jedec_flash_dev *flash_dev = (struct jedec_flash_dev *)chip_id;

	if (PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	uint8_t ret;
	uint8_t out[] = {
		flash_dev->cfg->sector_erase,
		(chip_offset >> 16) & 0xff,
		(chip_offset >>  8) & 0xff,
		(chip_offset >>  0) & 0xff,
	};

	if ((ret = PIOS_Flash_Jedec_WriteEnable(flash_dev)) != 0)
		return ret;

	if (PIOS_Flash_Jedec_ClaimBus(flash_dev) != 0)
		return -1;

	if (PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus(flash_dev);
		return -2;
	}

	PIOS_Flash_Jedec_ReleaseBus(flash_dev);

	// Keep polling when bus is busy too
	while (PIOS_Flash_Jedec_Busy(flash_dev) != 0) {
#if defined(FLASH_FREERTOS)
		vTaskDelay(1);
#endif
	}

	return 0;
}

/**
 * @brief Write one page of data (up to 256 bytes) aligned to a page start
 * @param[in] chip_id the opaque handle for the chip that this operation should be applied to
 * @param[in] chip_offset Address within flash to write to
 * @param[in] data Pointer to data to write to flash
 * @param[in] len Length of data to write (max 256 bytes)
 * @return Zero if success or error code
 * @retval -1 Unable to claim SPI bus
 * @retval -2 Size exceeds 256 bytes
 * @retval -3 Length to write would wrap around page boundary
 */
static int32_t PIOS_Flash_Jedec_WriteData(uintptr_t chip_id, uint32_t chip_offset, const uint8_t *data, uint16_t len)
{
	struct jedec_flash_dev *flash_dev = (struct jedec_flash_dev *)chip_id;

	if(PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	uint8_t ret;
	uint8_t out[4] = {
		JEDEC_PAGE_WRITE,
		(chip_offset >> 16) & 0xff,
		(chip_offset >>  8) & 0xff,
		(chip_offset >>  0) & 0xff,
	};

	/* Can only write one page at a time */
	if (len > 0x100)
		return -2;

	/* Ensure number of bytes fits after starting address before end of page */
	if (((chip_offset & 0xff) + len) > 0x100)
		return -3;

	if ((ret = PIOS_Flash_Jedec_WriteEnable(flash_dev)) != 0)
		return ret;

	/* Execute write page command and clock in address.  Keep CS asserted */
	if (PIOS_Flash_Jedec_ClaimBus(flash_dev) != 0)
		return -1;

	if (PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus(flash_dev);
		return -1;
	}

	/* Clock out data to flash */
	if (PIOS_SPI_TransferBlock(flash_dev->spi_id,data,NULL,len,NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus(flash_dev);
		return -1;
	}

	PIOS_Flash_Jedec_ReleaseBus(flash_dev);

	// Keep polling when bus is busy too
#if defined(FLASH_FREERTOS)
	while (PIOS_Flash_Jedec_Busy(flash_dev) != 0) {
		vTaskDelay(1);
	}
#else

	// Query status this way to prevent accel chip locking us out
	if (PIOS_Flash_Jedec_ClaimBus(flash_dev) < 0)
		return -1;

	PIOS_SPI_TransferByte(flash_dev->spi_id, JEDEC_READ_STATUS);
	while (PIOS_SPI_TransferByte(flash_dev->spi_id, JEDEC_READ_STATUS) & JEDEC_STATUS_BUSY);

	PIOS_Flash_Jedec_ReleaseBus(flash_dev);

#endif
	return 0;
}

/**
 * @brief Read data from a location in flash memory
 * @param[in] chip_id the opaque handle for the chip that this operation should be applied to
 * @param[in] chip_offset Address within flash to write to
 * @param[in] data Pointer to data to write from flash
 * @param[in] len Length of data to write (max 256 bytes)
 * @return Zero if success or error code
 * @retval -1 Unable to claim SPI bus
 */
static int32_t PIOS_Flash_Jedec_ReadData(uintptr_t chip_id, uint32_t chip_offset, uint8_t *data, uint16_t len)
{
	struct jedec_flash_dev *flash_dev = (struct jedec_flash_dev *)chip_id;

	if (PIOS_Flash_Jedec_Validate(flash_dev) != 0)
		return -1;

	if (PIOS_Flash_Jedec_ClaimBus(flash_dev) == -1)
		return -1;

	/* Execute read command and clock in address.  Keep CS asserted */
	uint8_t out[] = {
		JEDEC_READ_DATA,
		(chip_offset >> 16) & 0xff,
		(chip_offset >>  8) & 0xff,
		(chip_offset >>  0) & 0xff,
	};

	if (PIOS_SPI_TransferBlock(flash_dev->spi_id,out,NULL,sizeof(out),NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus(flash_dev);
		return -2;
	}

	/* Copy the transfer data to the buffer */
	if (PIOS_SPI_TransferBlock(flash_dev->spi_id,NULL,data,len,NULL) < 0) {
		PIOS_Flash_Jedec_ReleaseBus(flash_dev);
		return -3;
	}

	PIOS_Flash_Jedec_ReleaseBus(flash_dev);

	return 0;
}

/* Provide a flash driver to external drivers */
const struct pios_flash_driver pios_jedec_flash_driver = {
	.start_transaction = PIOS_Flash_Jedec_StartTransaction,
	.end_transaction   = PIOS_Flash_Jedec_EndTransaction,
	.erase_sector      = PIOS_Flash_Jedec_EraseSector,
	.write_data        = PIOS_Flash_Jedec_WriteData,
	.read_data         = PIOS_Flash_Jedec_ReadData,
};

#endif	/* PIOS_INCLUDE_FLASH_JEDEC */
