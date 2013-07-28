/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PiosFlashInternal PIOS Flash internal flash driver
 * @{
 *
 * @file       pios_flash_internal.c  
 * @author     Tau Labs, http://github.com/TauLabs, Copyright (C) 2012-2013.
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

#include "stm32f4xx_flash.h"
#include "pios_flash_internal_priv.h"
#include "pios_wdg.h"
#include <stdbool.h>

static const uint16_t sector_to_st_sector_map[] = {
	[0] = FLASH_Sector_0,
	[1] = FLASH_Sector_1,
	[2] = FLASH_Sector_2,
	[3] = FLASH_Sector_3,
	[4] = FLASH_Sector_4,
	[5] = FLASH_Sector_5,
	[6] = FLASH_Sector_6,
	[7] = FLASH_Sector_7,
	[8] = FLASH_Sector_8,
	[9] = FLASH_Sector_9,
	[10] = FLASH_Sector_10,
	[11] = FLASH_Sector_11,
};

enum pios_internal_flash_dev_magic {
	PIOS_INTERNAL_FLASH_DEV_MAGIC = 0x33445902,
};

struct pios_internal_flash_dev {
	enum pios_internal_flash_dev_magic magic;

	const struct pios_flash_internal_cfg *cfg;

#if defined(PIOS_INCLUDE_FREERTOS)
	xSemaphoreHandle transaction_lock;
#endif	/* defined(PIOS_INCLUDE_FREERTOS) */
};

static bool PIOS_Flash_Internal_Validate(struct pios_internal_flash_dev *flash_dev) {
	return (flash_dev && (flash_dev->magic == PIOS_INTERNAL_FLASH_DEV_MAGIC));
}

#if defined(PIOS_INCLUDE_FREERTOS)
static struct pios_internal_flash_dev *PIOS_Flash_Internal_alloc(void)
{
	struct pios_internal_flash_dev *flash_dev;

	flash_dev = (struct pios_internal_flash_dev *)pvPortMalloc(sizeof(*flash_dev));
	if (!flash_dev) return (NULL);

	flash_dev->magic = PIOS_INTERNAL_FLASH_DEV_MAGIC;

	return(flash_dev);
}
#else
#ifndef PIOS_INTERNAL_FLASH_MAX_DEVS
#define PIOS_INTERNAL_FLASH_MAX_DEVS 1
#endif	/* PIOS_INTERNAL_FLASH_MAX_DEVS */
static struct pios_internal_flash_dev pios_internal_flash_devs[PIOS_INTERNAL_FLASH_MAX_DEVS];
static uint8_t pios_internal_flash_num_devs;
static struct pios_internal_flash_dev *PIOS_Flash_Internal_alloc(void)
{
	struct pios_internal_flash_dev *flash_dev;

	if (pios_internal_flash_num_devs >= PIOS_INTERNAL_FLASH_MAX_DEVS) {
		return (NULL);
	}

	flash_dev = &pios_internal_flash_devs[pios_internal_flash_num_devs++];
	flash_dev->magic = PIOS_INTERNAL_FLASH_DEV_MAGIC;

	return (flash_dev);
}

#endif /* defined(PIOS_INCLUDE_FREERTOS) */

int32_t PIOS_Flash_Internal_Init(uintptr_t *chip_id, const struct pios_flash_internal_cfg *cfg)
{
	struct pios_internal_flash_dev *flash_dev;

	flash_dev = PIOS_Flash_Internal_alloc();
	if (flash_dev == NULL)
		return -1;

#if defined(PIOS_INCLUDE_FREERTOS)
	flash_dev->transaction_lock = xSemaphoreCreateMutex();
#endif	/* defined(PIOS_INCLUDE_FREERTOS) */

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

#if defined(PIOS_INCLUDE_FREERTOS)
	if (xSemaphoreTake(flash_dev->transaction_lock, portMAX_DELAY) != pdTRUE)
		return -2;
#endif	/* defined(PIOS_INCLUDE_FREERTOS) */

	/* Unlock the internal flash so we can write to it */
	FLASH_Unlock();
	return 0;
}

static int32_t PIOS_Flash_Internal_EndTransaction(uintptr_t chip_id)
{
	struct pios_internal_flash_dev *flash_dev = (struct pios_internal_flash_dev *)chip_id;

	if (!PIOS_Flash_Internal_Validate(flash_dev))
		return -1;

#if defined(PIOS_INCLUDE_FREERTOS)
	if (xSemaphoreGive(flash_dev->transaction_lock) != pdTRUE)
		return -2;
#endif	/* defined(PIOS_INCLUDE_FREERTOS) */

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

	/* Is the requested sector within this chip? */
	if (chip_sector >= NELEMENTS(sector_to_st_sector_map))
		return -2;

	uint32_t st_sector = sector_to_st_sector_map[chip_sector];

	if (FLASH_EraseSector(st_sector, VoltageRange_3) != FLASH_COMPLETE)
		return -3;

	return 0;
}

static int32_t PIOS_Flash_Internal_WriteData(uintptr_t chip_id, uint32_t chip_offset, const uint8_t *data, uint16_t len)
{
	PIOS_Assert(data);

	struct pios_internal_flash_dev *flash_dev = (struct pios_internal_flash_dev *)chip_id;

	if (!PIOS_Flash_Internal_Validate(flash_dev))
		return -1;

	/* Write the data */
	for (uint16_t i = 0; i < len; i++) {
		FLASH_Status status;
		/*
		 * This is inefficient.  Should try to do word writes.
		 * Not sure if word writes need to be aligned though.
		 */
		status = FLASH_ProgramByte(FLASH_BASE + chip_offset + i, data[i]);
		PIOS_Assert(status == FLASH_COMPLETE);
	}

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
