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
