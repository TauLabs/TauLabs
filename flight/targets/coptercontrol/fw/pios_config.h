/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup CopterControl OpenPilot coptercontrol support files
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
#define PIOS_INCLUDE_ADC
#define PIOS_INCLUDE_DELAY
#if defined(USE_I2C)
#define PIOS_INCLUDE_I2C
//#define PIOS_INCLUDE_I2C_ESC
#endif
#if defined(USE_PCF8591)
#define PIOS_INCLUDE_PCF8591
#endif
#define PIOS_INCLUDE_IRQ
#define PIOS_INCLUDE_LED
#define PIOS_INCLUDE_IAP
#define PIOS_INCLUDE_TIM
#define PIOS_INCLUDE_HPWM

#define PIOS_INCLUDE_RCVR

/* Supported receiver interfaces */
#define PIOS_INCLUDE_DSM
#define PIOS_INCLUDE_HSUM
#define PIOS_INCLUDE_SBUS
#define PIOS_INCLUDE_PPM
#define PIOS_INCLUDE_PWM
#define PIOS_INCLUDE_GCSRCVR

/* Supported USART-based PIOS modules */
#define PIOS_INCLUDE_TELEMETRY_RF
#define PIOS_INCLUDE_GPS
#define PIOS_GPS_MINIMAL
#define PIOS_INCLUDE_GPS_NMEA_PARSER /* Include the NMEA protocol parser */
#define PIOS_INCLUDE_GPS_UBX_PARSER  /* Include the UBX protocol parser */
#define PIOS_INCLUDE_MAVLINK
#define PIOS_INCLUDE_FRSKY_SENSOR_HUB
#define PIOS_INCLUDE_LIGHTTELEMETRY

#define PIOS_INCLUDE_SERVO
#define PIOS_INCLUDE_SPI
#define PIOS_INCLUDE_SYS
#define PIOS_INCLUDE_USART
#define PIOS_INCLUDE_USB
#define PIOS_INCLUDE_USB_HID
#define PIOS_INCLUDE_USB_RCTX
#define PIOS_INCLUDE_USB_CDC
#define PIOS_INCLUDE_COM
#define PIOS_INCLUDE_FREERTOS
#define PIOS_INCLUDE_GPIO
#define PIOS_INCLUDE_EXTI
#define PIOS_INCLUDE_RTC
#define PIOS_INCLUDE_WDG
#define PIOS_INCLUDE_BL_HELPER

#define PIOS_INCLUDE_ADXL345
#define PIOS_INCLUDE_MPU6000
#define PIOS_MPU6000_ACCEL

#define PIOS_INCLUDE_LOGFS_SETTINGS

#define PIOS_INCLUDE_FLASH
#define PIOS_INCLUDE_FLASH_JEDEC
#define PIOS_INCLUDE_FLASH_INTERNAL

#define PIOS_INCLUDE_SESSION_MANAGEMENT

/* Alarm Thresholds */
#define HEAP_LIMIT_WARNING             220
#define HEAP_LIMIT_CRITICAL             40
#define IRQSTACK_LIMIT_WARNING		100
#define IRQSTACK_LIMIT_CRITICAL		60
#define CPULOAD_LIMIT_WARNING		85
#define CPULOAD_LIMIT_CRITICAL		95

/* Task stack sizes */
#define PIOS_ACTUATOR_STACK_SIZE        800
#define PIOS_MANUAL_STACK_SIZE          600
#define PIOS_SYSTEM_STACK_SIZE          660
#define PIOS_STABILIZATION_STACK_SIZE   624
#define PIOS_TELEM_STACK_SIZE           500
#define PIOS_EVENTDISPATCHER_STACK_SIZE 720
#define PIOS_MAVLINK_STACK_SIZE         600
#define PIOS_COMUSBBRIDGE_STACK_SIZE    280
#define IDLE_COUNTS_PER_SEC_AT_NO_LOAD 1995998

// This can't be too high to stop eventdispatcher thread overflowing
#define PIOS_EVENTDISAPTCHER_QUEUE      10

/* PIOS Initcall infrastructure */
#define PIOS_INCLUDE_INITCALL

#define COPTERCONTROL
// Conditional related to making CC run navigation
#if defined(NAVGIATION)
#define PIOS_GPS_PROVIDES_AIRSPEED
#endif

#endif /* PIOS_CONFIG_H */
/**
 * @}
 * @}
 */
