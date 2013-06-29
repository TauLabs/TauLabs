#include <stdlib.h>		/* abort */
#include <stdio.h>		/* fopen/fread/fwrite/fseek */
#include <assert.h>		/* assert */
#include <string.h>		/* memset */

#include <stdbool.h>
#include "FreeRTOS.h"
#include "pios_flash_posix_priv.h"

enum flash_posix_magic {
	FLASH_POSIX_MAGIC = 0x321dabc1,
};

struct flash_posix_dev {
	enum flash_posix_magic magic;
	const struct pios_flash_posix_cfg * cfg;
	bool transaction_in_progress;
	FILE * flash_file;
};

static struct flash_posix_dev * PIOS_Flash_Posix_Alloc(void)
{
	struct flash_posix_dev * flash_dev = pvPortMalloc(sizeof(struct flash_posix_dev));

	flash_dev->magic = FLASH_POSIX_MAGIC;

	return flash_dev;
}

int32_t PIOS_Flash_Posix_Init(uintptr_t * chip_id, const struct pios_flash_posix_cfg * cfg)
{
	/* Check inputs */
	assert(chip_id);
	assert(cfg);
	assert(cfg->size_of_flash);
	assert(cfg->size_of_sector);
	assert((cfg->size_of_flash % cfg->size_of_sector) == 0);

	struct flash_posix_dev * flash_dev = PIOS_Flash_Posix_Alloc();
	assert(flash_dev);

	flash_dev->cfg = cfg;
	flash_dev->transaction_in_progress = false;

	flash_dev->flash_file = fopen ("theflash.bin", "r+");
	if (flash_dev->flash_file == NULL) {
		return -1;
	}

	if (fseek (flash_dev->flash_file, flash_dev->cfg->size_of_flash, SEEK_SET) != 0) {
		return -2;
	}

	*chip_id = (uintptr_t)flash_dev;

	return 0;
}

void PIOS_Flash_Posix_Destroy(uintptr_t chip_id)
{
	struct flash_posix_dev * flash_dev = (struct flash_posix_dev *)chip_id;

	fclose(flash_dev->flash_file);

	free(flash_dev);
}

/**********************************
 *
 * Provide a PIOS flash driver API
 *
 *********************************/
#include "pios_flash_priv.h"

static int32_t PIOS_Flash_Posix_StartTransaction(uintptr_t chip_id)
{
	struct flash_posix_dev * flash_dev = (struct flash_posix_dev *)chip_id;

	assert(!flash_dev->transaction_in_progress);

	flash_dev->transaction_in_progress = true;

	return 0;
}

static int32_t PIOS_Flash_Posix_EndTransaction(uintptr_t chip_id)
{
	struct flash_posix_dev * flash_dev = (struct flash_posix_dev *)chip_id;

	assert(flash_dev->transaction_in_progress);

	flash_dev->transaction_in_progress = false;

	return 0;
}

static int32_t PIOS_Flash_Posix_EraseSector(uintptr_t chip_id, uint32_t chip_sector, uint32_t chip_offset)
{
	struct flash_posix_dev * flash_dev = (struct flash_posix_dev *)chip_id;

	assert(flash_dev->transaction_in_progress);

	if (fseek (flash_dev->flash_file, chip_offset, SEEK_SET) != 0) {
		assert(0);
	}

	unsigned char * buf = pvPortMalloc(flash_dev->cfg->size_of_sector);
	assert (buf);
	memset((void *)buf, 0xFF, flash_dev->cfg->size_of_sector);

	size_t s;
	s = fwrite (buf, 1, flash_dev->cfg->size_of_sector, flash_dev->flash_file);

	free(buf);

	assert (s == flash_dev->cfg->size_of_sector);

	return 0;
}

static int32_t PIOS_Flash_Posix_WriteData(uintptr_t chip_id, uint32_t chip_offset, const uint8_t * data, uint16_t len)
{
	/* Check inputs */
	assert(data);

	struct flash_posix_dev * flash_dev = (struct flash_posix_dev *)chip_id;

	assert(flash_dev->transaction_in_progress);

	if (fseek (flash_dev->flash_file, chip_offset, SEEK_SET) != 0) {
		assert(0);
	}

	size_t s;
	s = fwrite (data, 1, len, flash_dev->flash_file);

	assert (s == len);

	return 0;
}

static int32_t PIOS_Flash_Posix_ReadData(uintptr_t chip_id, uint32_t chip_offset, uint8_t * data, uint16_t len)
{
	/* Check inputs */
	assert(data);

	struct flash_posix_dev * flash_dev = (struct flash_posix_dev *)chip_id;

	assert(flash_dev->transaction_in_progress);

	if (fseek (flash_dev->flash_file, chip_offset, SEEK_SET) != 0) {
		assert(0);
	}

	size_t s;
	s = fread (data, 1, len, flash_dev->flash_file);

	assert (s == len);

	return 0;
}

/* Provide a flash driver to external drivers */
const struct pios_flash_driver pios_posix_flash_driver = {
	.start_transaction = PIOS_Flash_Posix_StartTransaction,
	.end_transaction   = PIOS_Flash_Posix_EndTransaction,
	.erase_sector      = PIOS_Flash_Posix_EraseSector,
	.write_data        = PIOS_Flash_Posix_WriteData,
	.read_data         = PIOS_Flash_Posix_ReadData,
};

