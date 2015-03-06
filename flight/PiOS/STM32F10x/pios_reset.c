/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_RESET Reset functions
 * @brief Hardware functions to deal with the reset register
 * @{
 *
 * @file       pios_reset.c
 * @author     Kenn Sebesta, Copyright (C) 2015.
 * @brief      Reset information
 *
 ******************************************************************************/


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
	} else if(RCC_GetFlagStatus(RCC_FLAG_SFTRST) == SET) {  // Check #2
		reboot_reason = PIOS_RESET_FLAG_SOFTWARE;
	} else if(RCC_GetFlagStatus(RCC_FLAG_IWDGRST) == SET) { // Check #3
		reboot_reason = PIOS_RESET_FLAG_INDEPENDENT_WATCHDOG;
	} else if(RCC_GetFlagStatus(RCC_FLAG_WWDGRST) == SET) { // Check #4
		reboot_reason = PIOS_RESET_FLAG_WINDOW_WATCHDOG;
	} else if(RCC_GetFlagStatus(RCC_FLAG_LPWRRST) == SET) { // Check #5
		reboot_reason = PIOS_RESET_FLAG_LOW_POWER;
	} else if(RCC_GetFlagStatus(RCC_FLAG_PINRST) == SET) {  // Check #6. This MUST be last. This because the reset circuit works internally by triggering the NRST pin. See STM32 RCC docs.
		reboot_reason = PIOS_RESET_FLAG_PIN;
	}

	return reboot_reason;
}
