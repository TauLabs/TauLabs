/**
 ******************************************************************************
 *
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_FLASH Flash device handler
 * @{
 *
 * @file       pios_at45_flash_priv.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Driver for talking to AT45 flash chips
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

#ifndef PIOS_FLASH_AT45_PRIV_H_
#define PIOS_FLASH_AT45_PRIV_H_

#include "pios_flash_at45.h"		/* API definition for flash drivers */

extern const struct pios_flash_driver pios_at45_flash_driver;

#define MANUFACTURER_ATMEL    0X1F  // 0x1f - Manufacturer ID Atmel

struct pios_flash_at45_cfg {
	uint8_t expect_manufacturer;
	uint8_t expect_memorytype;
	uint8_t expect_capacity;
};

int32_t PIOS_Flash_AT45_Init(uintptr_t * flash_id, uint32_t spi_id, uint32_t slave_num, const struct pios_flash_at45_cfg * cfg);

#endif	/* PIOS_FLASH_AT45_PRIV_H_ */
