/**
 ******************************************************************************
 *
 * @file       plop_config.h  
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief      plop configuration header. 
 *             Central compile time config for the project.
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


#ifndef plop_CONFIG_POSIX_H
#define plop_CONFIG_POSIX_H


/* Enable/Disable plop Modules */
#define plop_INCLUDE_SYS
#define plop_INCLUDE_DELAY
#define plop_INCLUDE_LED
#define plop_INCLUDE_SDCARD
#define plop_INCLUDE_FREERTOS
#define plop_INCLUDE_COM
//#define plop_INCLUDE_GPS
#define plop_INCLUDE_IRQ
#define plop_INCLUDE_TELEMETRY_RF
#define plop_INCLUDE_TCP
#define plop_INCLUDE_UDP
#define plop_INCLUDE_SERVO
#define plop_INCLUDE_RCVR
#define plop_INCLUDE_GCSRCVR
#define plop_INCLUDE_IAP
#define plop_INCLUDE_BL_HELPER

#define plop_RCVR_MAX_CHANNELS			12
#define plop_RCVR_MAX_DEVS              3

/* Defaults for Logging */
#define LOG_FILENAME 			"plop.LOG"
#define STARTUP_LOG_ENABLED		1

/* COM Module */
#define GPS_BAUDRATE			19200
#define TELEM_BAUDRATE			19200
#define AUXUART_ENABLED			0
#define AUXUART_BAUDRATE		19200

#define TELEM_QUEUE_SIZE                20
#define plop_TELEM_STACK_SIZE           2048

/* Stabilization options */
#define plop_QUATERNION_STABILIZATION

/* GPS options */
#define plop_GPS_SETS_HOMELOCATION

#define HEAP_LIMIT_WARNING		4000
#define HEAP_LIMIT_CRITICAL		1000
#define IRQSTACK_LIMIT_WARNING		150
#define IRQSTACK_LIMIT_CRITICAL		80
#define CPULOAD_LIMIT_WARNING		80
#define CPULOAD_LIMIT_CRITICAL		95

#define REVOLUTION

#endif /* plop_CONFIG_POSIX_H */
