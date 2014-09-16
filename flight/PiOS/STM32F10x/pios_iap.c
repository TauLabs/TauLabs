/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_IAP IAP Functions
 * @brief STM32F1xx PIOS IAP Functions
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

/****************************************************************************************
 *  Private Definitions/Macros
 ****************************************************************************************/

/* these definitions reside here for protection and privacy. */
#define IAP_MAGIC_WORD_1	0x1122
#define IAP_MAGIC_WORD_2	0xAA55
#define IAP_MAGIC_WORD_3	0xBB11

#define UPPERWORD16(lw)	(uint16_t)((uint32_t)(lw)>>16)
#define LOWERWORD16(lw)	(uint16_t)((uint32_t)(lw)&0x0000ffff)
#define UPPERBYTE(w)	(uint8_t)((w)>>8)
#define LOWERBYTE(w)	(uint8_t)((w)&0x00ff)

/****************************************************************************************
 *  Private Functions
 ****************************************************************************************/

/****************************************************************************************
 *  Private (static) Data
 ****************************************************************************************/

/****************************************************************************************
 *  Public/Global Data
 ****************************************************************************************/

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
	/* Enable CRC clock */
	RCC_AHBPeriphClockCmd(RCC_AHBPeriph_CRC, ENABLE);

	/* Enable PWR and BKP clock */
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_PWR | RCC_APB1Periph_BKP, ENABLE);

	/* Enable write access to Backup domain */
	PWR_BackupAccessCmd(ENABLE);

	/* Clear Tamper pin Event(TE) pending flag */
	BKP_ClearFlag();

}

/*!
 * \brief     Determines if an In-Application-Programming request has been made.
 * \return    true - if correct sequence found
 */
uint32_t	PIOS_IAP_CheckRequest( void )
{
	uint32_t	retval = false;
	uint16_t	reg1;
	uint16_t	reg2;

	reg1 = BKP_ReadBackupRegister( MAGIC_REG_1 );
	reg2 = BKP_ReadBackupRegister( MAGIC_REG_2 );

	if( reg1 == IAP_MAGIC_WORD_1 && reg2 == IAP_MAGIC_WORD_2 ) {
		// We have a match.
		retval = true;
	} else {
		retval = false;
	}
	return retval;
}

/*!
 * \brief     Determines if a boot request has been made.
 * \return    true - if correct sequence found
 */
uint32_t	PIOS_Boot_CheckRequest( void )
{
	uint32_t	retval = false;
	uint16_t	reg1;
	uint16_t	reg2;

	reg1 = BKP_ReadBackupRegister( MAGIC_REG_1 );
	reg2 = BKP_ReadBackupRegister( MAGIC_REG_2 );

	if( reg1 == IAP_MAGIC_WORD_1 && reg2 == IAP_MAGIC_WORD_3 ) {
		// We have a match.
		retval = true;
	} else {
		retval = false;
	}
	return retval;
}

/*!
 * \brief   Sets the 1st word of the request sequence.
 * \param   n/a
 * \return  n/a
 * \retval
 */
void	PIOS_IAP_SetRequest1(void)
{
	BKP_WriteBackupRegister( MAGIC_REG_1, IAP_MAGIC_WORD_1);
}

void	PIOS_IAP_SetRequest2(void)
{
	BKP_WriteBackupRegister( MAGIC_REG_2, IAP_MAGIC_WORD_2);
}

void	PIOS_IAP_SetRequest3(void)
{
	BKP_WriteBackupRegister( MAGIC_REG_2, IAP_MAGIC_WORD_3);
}

void	PIOS_IAP_ClearRequest(void)
{
	BKP_WriteBackupRegister( MAGIC_REG_1, 0);
	BKP_WriteBackupRegister( MAGIC_REG_2, 0);
}

uint16_t PIOS_IAP_ReadBootCount(void)
{
	return BKP_ReadBackupRegister ( IAP_BOOTCOUNT );
}

void PIOS_IAP_WriteBootCount (uint16_t boot_count)
{
	BKP_WriteBackupRegister ( IAP_BOOTCOUNT, boot_count );
}
