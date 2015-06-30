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

/**
 * @brief PIOS_RESET_Clear Does nothing on POSIX systems
 */
void PIOS_RESET_Clear()
{
	// This space intentionally left blank
}

/**
 * @brief PIOS_RESET_GetResetReason Does nothing on POSIX systems
 * @return Always return Undefined
 */
int16_t PIOS_RESET_GetResetReason (void)
{
	return PIOS_RESET_FLAG_SOFTWARE;
}
