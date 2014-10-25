/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_IAP IAP Functions
 * @brief Common PIOS IAP Function header
 * @{
 *
 * @file       pios_iap.h
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

#ifndef PIOS_IAP_H_
#define PIOS_IAP_H_


/****************************************************************************************
 *  Header files
 ****************************************************************************************/

/*****************************************************************************************
 *	Public Definitions/Macros
 ****************************************************************************************/
#if defined(STM32F4XX) || defined(STM32F30X)
#define MAGIC_REG_1     RTC_BKP_DR1
#define MAGIC_REG_2     RTC_BKP_DR2
#define IAP_BOOTCOUNT   RTC_BKP_DR3
#else
#define MAGIC_REG_1     BKP_DR1
#define MAGIC_REG_2     BKP_DR2
#define IAP_BOOTCOUNT   BKP_DR3
#endif

/****************************************************************************************
 *  Public Functions
 ****************************************************************************************/
void		PIOS_IAP_Init(void);
uint32_t	PIOS_IAP_CheckRequest( void );
uint32_t	PIOS_Boot_CheckRequest( void );
void		PIOS_IAP_SetRequest1(void);
void		PIOS_IAP_SetRequest2(void);
void		PIOS_IAP_SetRequest3(void);
void		PIOS_IAP_ClearRequest(void);
uint16_t	PIOS_IAP_ReadBootCount(void);
void		PIOS_IAP_WriteBootCount(uint16_t);

/****************************************************************************************
 *  Public Data
 ****************************************************************************************/

#endif /* PIOS_IAP_H_ */
