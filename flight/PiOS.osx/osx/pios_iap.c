/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_IAP IAP Functions
 * @brief OSX SITL PIOS IAP Functions
 * @{
 *
 * @file       pios_iap.c
 * @author     joe 2010
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      In application programming functions
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

/****************************************************************************************
 *  Header files
 ****************************************************************************************/
#include <pios.h>

/*!
 * \brief	PIOS_IAP_Init - performs required initializations for iap module.
 * \param   none.
 * \return	none.
 * \retval	none.
 *
 *	Created: Sep 8, 2010 10:10:48 PM by joe
 */
void PIOS_IAP_Init( void )
{

}

/*!
 * \brief     Determines if an In-Application-Programming request has been made.
 * \param   *comm - Which communication stream to use for the IAP (USB, Telemetry, I2C, SPI, etc)
 * \return    TRUE - if correct sequence found, along with 'comm' updated.
 * 			FALSE - Note that 'comm' will have an invalid comm identifier.
 * \retval
 *
 */
uint32_t	PIOS_IAP_CheckRequest( void )
{

	return false;
}

uint32_t	PIOS_Boot_CheckRequest( void )
{
	return false;
}

/*!
 * \brief   Sets the 1st word of the request sequence.
 * \param   n/a
 * \return  n/a
 * \retval
 */
void	PIOS_IAP_SetRequest1(void)
{
}

void	PIOS_IAP_SetRequest2(void)
{
}

void	PIOS_IAP_SetRequest3(void)
{
}

void	PIOS_IAP_ClearRequest(void)
{
}

uint16_t PIOS_IAP_ReadBootCount(void)
{
	return 0;
}

void PIOS_IAP_WriteBootCount (uint16_t boot_count)
{
}
