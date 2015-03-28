/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup PipXtreme OpenPilot PipXtreme support files
 * @{
 *
 * @file       pios_config.h 
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      Board specific options that modify PiOS capabilities
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

/* Enable/Disable PiOS Modules */
#define PIOS_INCLUDE_DELAY
#define PIOS_INCLUDE_IRQ
#define PIOS_INCLUDE_LED
#define PIOS_INCLUDE_IAP
#define PIOS_INCLUDE_RFM22B
#define PIOS_INCLUDE_RFM22B_COM
#define PIOS_INCLUDE_RCVR
#define PIOS_INCLUDE_TIM

/* Supported receiver interfaces */
#define PIOS_INCLUDE_PPM

/* Supported USART-based PIOS modules */
#define PIOS_INCLUDE_SPI
#define PIOS_INCLUDE_SYS
#define PIOS_INCLUDE_USART
#define PIOS_INCLUDE_USB
#define PIOS_INCLUDE_USB_HID
#define PIOS_INCLUDE_USB_CDC
#define PIOS_INCLUDE_COM
#define PIOS_INCLUDE_FREERTOS
#define PIOS_INCLUDE_GPIO
#define PIOS_INCLUDE_EXTI
#define PIOS_INCLUDE_RTC
#define PIOS_INCLUDE_WDG
#define PIOS_INCLUDE_BL_HELPER
#define PIOS_INCLUDE_RFM22B
#define PIOS_INCLUDE_PACKET_HANDLER

#define PIOS_INCLUDE_TARANIS_SPORT
 
#define PIOS_INCLUDE_FLASH
#define PIOS_INCLUDE_FLASH_INTERNAL
#define PIOS_INCLUDE_LOGFS_SETTINGS

/* Defaults for Logging */
#define LOG_FILENAME 			"PIOS.LOG"
#define STARTUP_LOG_ENABLED		1

/* COM Module */
#define GPS_BAUDRATE			19200
#define TELEM_BAUDRATE			19200
#define AUXUART_ENABLED			0
#define AUXUART_BAUDRATE		19200

/* Alarm Thresholds */
#define HEAP_LIMIT_WARNING             220
#define HEAP_LIMIT_CRITICAL             40
#define IRQSTACK_LIMIT_WARNING		100
#define IRQSTACK_LIMIT_CRITICAL		60
#define CPULOAD_LIMIT_WARNING		85
#define CPULOAD_LIMIT_CRITICAL		95

/* Task stack sizes */
#define PIOS_ACTUATOR_STACK_SIZE       1020
#define PIOS_MANUAL_STACK_SIZE          724
#define PIOS_SYSTEM_STACK_SIZE          460
#define PIOS_STABILIZATION_STACK_SIZE   524
#define PIOS_TELEM_STACK_SIZE           500
#define PIOS_EVENTDISPATCHER_STACK_SIZE 520
#define IDLE_COUNTS_PER_SEC_AT_NO_LOAD 1995998

// This can't be too high to stop eventdispatcher thread overflowing
#define PIOS_EVENTDISAPTCHER_QUEUE      10

/* PIOS Initcall infrastructure */
#define PIOS_INCLUDE_INITCALL

#define PIOS_INCLUDE_DEBUG_CONSOLE

/* Turn on debugging signals on the telemetry port */
//#define PIOS_RFM22B_DEBUG_ON_TELEM

#define TAULINK_VERSION_STICK 0x01
#define TAULINK_VERSION_MODULE 0x02

#endif /* PIOS_CONFIG_H */
/**
 * @}
 * @}
 */
