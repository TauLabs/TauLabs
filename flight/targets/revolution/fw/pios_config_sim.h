/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup Revolution OpenPilot Revolution support files
 * @{
 *
 * @file       pios_config_sim.h 
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Simulation specific options that modify PiOS capabilities
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


#ifndef PIOS_CONFIG_POSIX_H
#define PIOS_CONFIG_POSIX_H


/* Enable/Disable PiOS Modules */
#define PIOS_INCLUDE_SYS
#define PIOS_INCLUDE_DELAY
#define PIOS_INCLUDE_LED
#define PIOS_INCLUDE_FREERTOS
#define PIOS_INCLUDE_COM
//#define PIOS_INCLUDE_GPS
#define PIOS_INCLUDE_IRQ
#define PIOS_INCLUDE_TELEMETRY_RF
#define PIOS_INCLUDE_TCP
#define PIOS_INCLUDE_UDP
#define PIOS_INCLUDE_SERVO
#define PIOS_INCLUDE_RCVR
#define PIOS_INCLUDE_GCSRCVR
#define PIOS_INCLUDE_IAP
#define PIOS_INCLUDE_BL_HELPER
#define PIOS_INCLUDE_FLASH
#define PIOS_INCLUDE_LOGFS_SETTINGS

#define PIOS_RCVR_MAX_CHANNELS			12
#define PIOS_RCVR_MAX_DEVS              3

/* Defaults for Logging */
#define LOG_FILENAME 			"PIOS.LOG"
#define STARTUP_LOG_ENABLED		1

/* COM Module */
#define GPS_BAUDRATE			19200
#define TELEM_BAUDRATE			19200
#define AUXUART_ENABLED			0
#define AUXUART_BAUDRATE		19200

#define TELEM_QUEUE_SIZE                20
#define PIOS_TELEM_STACK_SIZE           2048

/* Stabilization options */

#define HEAP_LIMIT_WARNING		4000
#define HEAP_LIMIT_CRITICAL		1000
#define IRQSTACK_LIMIT_WARNING		150
#define IRQSTACK_LIMIT_CRITICAL		80
#define CPULOAD_LIMIT_WARNING		80
#define CPULOAD_LIMIT_CRITICAL		95

#define IDLE_COUNTS_PER_SEC_AT_NO_LOAD 9959

#define REVOLUTION

// Enable POI tracking mode for camera stabilization
#define CAMERASTAB_POI_MODE

#endif /* PIOS_CONFIG_POSIX_H */
