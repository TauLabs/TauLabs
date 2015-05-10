/**
 ******************************************************************************
 *
 * @file       pios.h  
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      Main PiOS header. 
 *                 - Central header for the project.
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


#ifndef PIOS_H
#define PIOS_H

/* PIOS Feature Selection */
#include "pios_config_sim.h"

#if defined(PIOS_INCLUDE_CHIBIOS)
/* @note    This is required because of difference in chip define between ChibiOS and ST libs.
 *          It is also used to force inclusion of chibios_transition defines. */
#include "hal.h"
#endif /* defined(PIOS_INCLUDE_CHIBIOS) */

#include <pios_posix.h>

#if defined(PIOS_INCLUDE_FREERTOS)
/* FreeRTOS Includes */
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"

#define vPortInitialiseBlocks(); 
#endif

/* C Lib Includes */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* Generic initcall infrastructure */
#include "pios_initcall.h"

/* PIOS Board Specific Device Configuration */
#include "pios_board_sim.h"

/* PIOS Hardware Includes (posix) */
#include <pios_heap.h>
#include <pios_sys.h>
#include <pios_delay.h>
#include <pios_led.h>
#include <pios_udp.h>
#include <pios_tcp.h>
#include <pios_com.h>
#include <pios_servo.h>
#include <pios_wdg.h>
#include <pios_debug.h>
#include <pios_crc.h>
#include <pios_rcvr.h>
#include <pios_irq.h>
#include <pios_sensors.h>
#include <pios_sim.h>
#include <pios_flashfs.h>

#if defined(PIOS_INCLUDE_IAP)
#include <pios_iap.h>
#endif
#if defined(PIOS_INCLUDE_BL_HELPER)
#include <pios_bl_helper.h>
#endif

#define NELEMENTS(x) (sizeof(x) / sizeof(*(x)))

#if defined(PIOS_INCLUDE_FREERTOS)
// portTICK_RATE_MS is in [ms/tick].
// See http://sourceforge.net/tracker/?func=detail&aid=3498382&group_id=111543&atid=659636
#define TICKS2MS(t)	((t) * (portTICK_RATE_MS))
#define MS2TICKS(m)	((m) / (portTICK_RATE_MS))
#endif /* defined(PIOS_INCLUDE_FREERTOS) */

#endif /* PIOS_H */
