/**
 ******************************************************************************
 * @addtogroup TauLabsBootloader Tau Labs Bootloaders
 * @{
 * @addtogroup BootloaderUpdate Update the bootloader stored in a target
 * @{
 *
 * @file       pios_config.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Non pinout-specific configuration of PiOS for bootloader updater
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


#ifndef PIOS_CONFIG_H
#define PIOS_CONFIG_H
#define PIOS_INCLUDE_BL_HELPER
#define PIOS_INCLUDE_DELAY
#define PIOS_INCLUDE_LED
#define PIOS_INCLUDE_SYS
#define PIOS_INCLUDE_GPIO
#define PIOS_NO_GPS
//#define DEBUG_SSP
#define PIOS_INCLUDE_IRQ

#endif /* PIOS_CONFIG_H */
/**
 * @}
 * @}
 */
