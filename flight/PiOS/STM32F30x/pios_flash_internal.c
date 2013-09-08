/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PiosFlashInternal PIOS Flash internal flash driver
 * @{
 *
 * @file       pios_flash_internal.c  
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief Provides a flash driver for the STM32 internal flash sectors
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

#if defined(PIOS_INCLUDE_FLASH_INTERNAL)

#include "stm32f30x_flash.h"
#include "pios_flash_internal_priv.h"
#include "pios_wdg.h"
#include "pios_semaphore.h"
#include <stdbool.h>

#define STM32F30X_FLASH_SECTOR_SIZE 2048

enum pios_internal_flash_dev_magic {
	PIOS_INTERNAL_FLASH_DEV_MAGIC = 0xae82c65d,
};

struct pios_internal_flash_dev {
	enum pios_internal_flash_dev_magic magic;

	const struct pios_flash_internal_cfg *cfg;

	struct pios_semaphore *transaction_lock;
};

static bool PIOS_Flash_Internal_Validate(struct pios_internal_flash_dev *flash_dev) {
	return (flash_dev && (flash_dev->magic == PIOS_INTERNAL_FLASH_DEV_MAGIC));
}

static struct pios_internal_flash_dev *PIOS_Flash_Internal_alloc(void)
{
	struct pios_internal_flash_dev *flash_dev;

	flash_dev = (struct pios_internal_flash_dev *)PIOS_malloc(sizeof(*flash_dev));
	if (!flash_dev) return (NULL);

	flash_dev->magic = PIOS_INTERNAL_FLASH_DEV_MAGIC;

	return(flash_dev);
}

int32_t PIOS_Flash_Internal_Init(uintptr_t *chip_id, const struct pios_flash_internal_cfg *cfg)
{
	struct pios_internal_flash_dev *flash_dev;

	flash_dev = PIOS_Flash_Internal_alloc();
	if (flash_dev == NULL)
		return -1;

	flash_dev->transaction_lock = PIOS_Semaphore_Create();

	flash_dev->cfg = cfg;

	*chip_id = (uintptr_t) flash_dev;

	return 0;
}

/**********************************
 *
 * Provide a PIOS flash driver API
 *
 *********************************/
#include "pios_flash_priv.h"

static int32_t PIOS_Flash_Internal_StartTransaction(uintptr_t chip_id)
{
	struct pios_internal_flash_dev *flash_dev = (struct pios_internal_flash_dev *)chip_id;

	if (!PIOS_Flash_Internal_Validate(flash_dev))
		return -1;

	if (PIOS_Semaphore_Take(flash_dev->transaction_lock, PIOS_SEMAPHORE_TIMEOUT_MAX) != true)
		return -2;

	/* Unlock the internal flash so we can write to it */
	FLASH_Unlock();

	return 0;
}

static int32_t PIOS_Flash_Internal_EndTransaction(uintptr_t chip_id)
{
	struct pios_internal_flash_dev *flash_dev = (struct pios_internal_flash_dev *)chip_id;

	if (!PIOS_Flash_Internal_Validate(flash_dev))
		return -1;

	if (PIOS_Semaphore_Give(flash_dev->transaction_lock) != true)
		return -2;

	/* Lock the internal flash again so we can no longer write to it */
	FLASH_Lock();

	return 0;
}

static int32_t PIOS_Flash_Internal_EraseSector(uintptr_t chip_id, uint32_t chip_sector, uint32_t chip_offset)
{
	struct pios_internal_flash_dev *flash_dev = (struct pios_internal_flash_dev *)chip_id;

	if (!PIOS_Flash_Internal_Validate(flash_dev))
		return -1;

#if defined(PIOS_INCLUDE_WDG)
	PIOS_WDG_Clear();
#endif

	FLASH_Status ret = FLASH_ErasePage(FLASH_BASE + chip_offset);
	if (ret != FLASH_COMPLETE)
		return -3;

	return 0;
}

static int32_t PIOS_Flash_Internal_ReadData(uintptr_t chip_id, uint32_t chip_offset, uint8_t *data, uint16_t len)
{
	PIOS_Assert(data);

	struct pios_internal_flash_dev *flash_dev = (struct pios_internal_flash_dev *)chip_id;

	if (!PIOS_Flash_Internal_Validate(flash_dev))
		return -1;

	/* Read the data into the buffer directly */
	memcpy(data, (void *)(FLASH_BASE + chip_offset), len);

	return 0;
}

static int32_t PIOS_Flash_Internal_WriteData(uintptr_t chip_id, uint32_t chip_offset, const uint8_t *data, uint16_t len)
{
	PIOS_Assert(data);
	PIOS_Assert((chip_offset & 0x0001) == 0)

	struct pios_internal_flash_dev *flash_dev = (struct pios_internal_flash_dev *)chip_id;

	if (!PIOS_Flash_Internal_Validate(flash_dev))
		return -1;

	/* Write the data */
	for (uint16_t i = 0; i < (len & ~1); i += 2) {
		/* Check if content has been changed prior to write.
		 * This should enhance performance and make this compatible with the F3 chip.
		 */
		uint8_t temp[2];
		if (PIOS_Flash_Internal_ReadData(chip_id, chip_offset + i, temp, sizeof(temp)) != 0)
			return -4;

		if (temp[0] == data[i] && temp[1] == data[i + 1])
			continue;

		FLASH_Status status;
		status = FLASH_ProgramHalfWord(FLASH_BASE + chip_offset + i, data[i + 1] << 8 | data[i]);
		PIOS_Assert(status == FLASH_COMPLETE);
	}
	/* Handle uneven writes by filling up with 0xff*/
	if ((len & 1) != 0) {
		FLASH_Status status;
		status = FLASH_ProgramHalfWord(FLASH_BASE + chip_offset + len - 1, 0xff << 8 | data[len - 1]);
		PIOS_Assert(status == FLASH_COMPLETE);
	}

	return 0;
}

/* Provide a flash driver to external drivers */
const struct pios_flash_driver pios_internal_flash_driver = {
	.start_transaction = PIOS_Flash_Internal_StartTransaction,
	.end_transaction   = PIOS_Flash_Internal_EndTransaction,
	.erase_sector      = PIOS_Flash_Internal_EraseSector,
	.write_data        = PIOS_Flash_Internal_WriteData,
	.read_data         = PIOS_Flash_Internal_ReadData,
};

#endif	/* defined(PIOS_INCLUDE_FLASH_INTERNAL) */

/**
 * @}
 * @}
 */
