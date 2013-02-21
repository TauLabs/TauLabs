#include <stdint.h>

struct pios_flash_ut_cfg {
	uint32_t size_of_flash;
	uint32_t size_of_sector;
};

int32_t PIOS_Flash_UT_Init(uintptr_t * flash_id, const struct pios_flash_ut_cfg * cfg);
void PIOS_Flash_UT_Destroy(uintptr_t flash_id);

extern const struct pios_flash_driver pios_ut_flash_driver;
