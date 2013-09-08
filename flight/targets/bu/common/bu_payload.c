/**
 ******************************************************************************
 * @file       bu_payload.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup Bootloader
 * @{
 * @addtogroup Bootloader
 * @{
 * @brief Includes the bootloader image as payload in the bootloader uploader image
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

#include <pios_stringify.h>	/* __stringify */

asm(
	".section .rodata\n"

	".global _bu_payload_start\n"
	"_bu_payload_start:\n"
	".incbin \"" __stringify(BU_PAYLOAD_FILE) "\"\n"
	".global _bu_payload_end\n"
	"_bu_payload_end:\n"

	".global _bu_payload_size\n"
	"_bu_payload_size:\n"
	".word _bu_payload_end - _bu_payload_start\n"
	".previous\n"
);

