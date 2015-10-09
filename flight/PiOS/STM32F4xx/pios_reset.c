/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_RESET Reset functions
 * @brief Hardware functions to deal with the reset register
 * @{
 *
 * @file       pios_reset.c
 * @author     Tau Labs, Copyright (C) 2015.
 * @brief      Reset information
 *
 ******************************************************************************/
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


/* Project Includes */
#include "pios.h"

#include "pios_reset.h"

void PIOS_RESET_Clear()
{
    RCC_ClearFlag();
}

int16_t PIOS_RESET_GetResetReason (void)
{
	uint8_t reboot_reason = PIOS_RESET_FLAG_UNDEFINED;
	
	// ****************************************************** //
	// The order of these checks is important. DO NOT CHANGE. //
	// ****************************************************** //
	if(RCC_GetFlagStatus(RCC_FLAG_PORRST) == SET) {         // Check #1
		reboot_reason = PIOS_RESET_FLAG_POWERON;
	} else if (RCC_GetFlagStatus(RCC_FLAG_BORRST) == SET) { // Check #2
		reboot_reason = PIOS_RESET_FLAG_BROWNOUT;
	} else if(RCC_GetFlagStatus(RCC_FLAG_SFTRST) == SET) {  // Check #3
		reboot_reason = PIOS_RESET_FLAG_SOFTWARE;
	} else if(RCC_GetFlagStatus(RCC_FLAG_IWDGRST) == SET) { // Check #4
		reboot_reason = PIOS_RESET_FLAG_INDEPENDENT_WATCHDOG;
	} else if(RCC_GetFlagStatus(RCC_FLAG_WWDGRST) == SET) { // Check #5
		reboot_reason = PIOS_RESET_FLAG_WINDOW_WATCHDOG;
	} else if(RCC_GetFlagStatus(RCC_FLAG_LPWRRST) == SET) { // Check #6
		reboot_reason = PIOS_RESET_FLAG_LOW_POWER;
	} else if(RCC_GetFlagStatus(RCC_FLAG_PINRST) == SET) {  // Check #7. This MUST be last. This because the reset circuit works internally by triggering the NRST pin. See STM32 RCC docs.
		reboot_reason = PIOS_RESET_FLAG_PIN;
	}

	return reboot_reason;
}
