/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup DiscoveryF4 DiscoveryF4 support files
 * @{
 *
 * @file       pios_config.h 
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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

/* Major features */
#define PIOS_INCLUDE_FREERTOS
#define PIOS_INCLUDE_BL_HELPER

/* Enable/Disable PiOS Modules */
//#define PIOS_INCLUDE_ADC
#define PIOS_INCLUDE_DELAY
#define PIOS_INCLUDE_I2C
#define PIOS_INCLUDE_IRQ
#define PIOS_INCLUDE_LED
#define PIOS_INCLUDE_IAP
//#define PIOS_INCLUDE_TIM

#define PIOS_INCLUDE_RCVR

/* Supported receiver interfaces */
//#define PIOS_INCLUDE_DSM
//#define PIOS_INCLUDE_SBUS
//#define PIOS_INCLUDE_PPM
//#define PIOS_INCLUDE_PWM
#define PIOS_INCLUDE_GCSRCVR

/* Supported USART-based PIOS modules */
#define PIOS_INCLUDE_TELEMETRY_RF
//#define PIOS_INCLUDE_GPS
//#define PIOS_GPS_MINIMAL

//#define PIOS_INCLUDE_SERVO
#define PIOS_INCLUDE_SPI
#define PIOS_INCLUDE_SYS
#define PIOS_INCLUDE_USART
#define PIOS_INCLUDE_USB
#define PIOS_INCLUDE_USB_HID
#define PIOS_INCLUDE_USB_CDC
#define PIOS_INCLUDE_COM
//#define PIOS_INCLUDE_GPIO
#define PIOS_INCLUDE_EXTI
#define PIOS_INCLUDE_RTC
#define PIOS_INCLUDE_WDG
#define PIOS_INCLUDE_FASTHEAP

//#define PIOS_INCLUDE_MPU6050
//#define PIOS_MPU6050_ACCEL

#define PIOS_INCLUDE_FLASH
#define PIOS_INCLUDE_FLASH_INTERNAL
#define PIOS_INCLUDE_LOGFS_SETTINGS

/* Other Interfaces */
//#define PIOS_INCLUDE_I2C_ESC

/* Flags that alter behaviors - mostly to lower resources for CC */
#define PIOS_INCLUDE_INITCALL           /* Include init call structures */
#define PIOS_TELEM_PRIORITY_QUEUE       /* Enable a priority queue in telemetry */
//#define PIOS_QUATERNION_STABILIZATION   /* Stabilization options HEAVILY BROKEN!! */
//#define PIOS_GPS_SETS_HOMELOCATION      /* GPS options */

/* Alarm Thresholds */
#define HEAP_LIMIT_WARNING		4000
#define HEAP_LIMIT_CRITICAL		1000
#define IRQSTACK_LIMIT_WARNING		150
#define IRQSTACK_LIMIT_CRITICAL		80
#define CPULOAD_LIMIT_WARNING		80
#define CPULOAD_LIMIT_CRITICAL		95

/*
 * This has been calibrated 2013/07/20 using b0a3b349ddeaf3fc124a418da703577b2d6233e0
 * FreeRTOS 7.5.0, gcc-arm-none-eabi-4_7-2013q1, -Os
 * Calibration has been done by disabling the init task, breaking into debugger after
 * approximately 60 seconds, then doing the following math:
 *
 * IDLE_COUNTS_PER_SEC_AT_NO_LOAD = (uint32_t)((double)idleCounter / xTickCount * 1000 + 0.5)
 *
 * This has to be redone every time the toolchain, toolchain flags or FreeRTOS
 * configuration like number of task priorities or similar changes.
 * A change in the cpu load calculation or the idle task handler will invalidate this as well.
 */
#define IDLE_COUNTS_PER_SEC_AT_NO_LOAD (6995810)

//This enables altitude hold in manualcontrol module
//#define REVOLUTION

#endif /* PIOS_CONFIG_H */
/**
 * @}
 * @}
 */
