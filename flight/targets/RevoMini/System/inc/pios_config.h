/**
 ******************************************************************************
 * @addtogroup OpenPilotSystem OpenPilot System
 * @{
 * @addtogroup OpenPilotCore OpenPilot Core
 * @{
 *
 * @file       plop_config.h  
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief      plop configuration header. 
 *             Central compile time config for the project.
 *             In particular, plop_config.h is where you define which plop libraries
 *             and features are included in the firmware.
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


#ifndef plop_CONFIG_H
#define plop_CONFIG_H

/* Major features */
#define plop_INCLUDE_FREERTOS
#define plop_INCLUDE_BL_HELPER

/* Enable/Disable plop Modules */
#define plop_INCLUDE_ADC
#define plop_INCLUDE_DELAY
#define plop_INCLUDE_I2C
#define plop_I2C_DIAGNOSTICS
#define I2C_WDG_STATS_DIAGNOSTICS
#define plop_INCLUDE_IRQ
#define plop_INCLUDE_LED
#define plop_INCLUDE_IAP
#define plop_INCLUDE_SERVO
#define plop_INCLUDE_SPI
#define plop_INCLUDE_SYS
#define plop_INCLUDE_USART
#define plop_INCLUDE_USB
#define plop_INCLUDE_USB_HID
#define plop_INCLUDE_USB_CDC
//#define plop_INCLUDE_GPIO
#define plop_INCLUDE_EXTI
#define plop_INCLUDE_RTC
#define plop_INCLUDE_WDG

/* Variables related to the RFM22B functionality */
#define plop_INCLUDE_RFM22B
#define plop_INCLUDE_RFM22B_COM
 
/* Select the sensors to include */
#define plop_INCLUDE_HMC5883
#define plop_HMC5883_HAS_Gplop
#define plop_INCLUDE_MPU6000
#define plop_MPU6000_ACCEL
#define plop_INCLUDE_MS5611
//#define plop_INCLUDE_ETASV3
//#define plop_INCLUDE_HCSR04
#define FLASH_FREERTOS
/* Com systems to include */
#define plop_INCLUDE_COM
#define plop_INCLUDE_COM_TELEM
#define plop_INCLUDE_COM_FLEXI

#define plop_INCLUDE_GPS
#define plop_INCLUDE_GPS_NMEA_PARSER
#define plop_INCLUDE_GPS_UBX_PARSER
#define plop_GPS_SETS_HOMELOCATION

/* Supported receiver interfaces */
#define plop_INCLUDE_RCVR
#define plop_INCLUDE_DSM
//#define plop_INCLUDE_SBUS
#define plop_INCLUDE_PPM
#define plop_INCLUDE_PWM
#define plop_INCLUDE_GCSRCVR

#define plop_INCLUDE_SETTINGS
#define plop_INCLUDE_FLASH
/* A really shitty setting saving implementation */
#define plop_INCLUDE_FLASH_SECTOR_SETTINGS

//#define plop_INCLUDE_DEBUG_CONSOLE

/* Other Interfaces */
//#define plop_INCLUDE_I2C_ESC

/* Flags that alter behaviors - mostly to lower resources for CC */
#define plop_INCLUDE_INITCALL           /* Include init call structures */
#define plop_TELEM_PRIORITY_QUEUE       /* Enable a priority queue in telemetry */
//#define plop_QUATERNION_STABILIZATION   /* Stabilization options */
#define plop_GPS_SETS_HOMELOCATION      /* GPS options */

/* Alarm Thresholds */
#define HEAP_LIMIT_WARNING		1000
#define HEAP_LIMIT_CRITICAL		500
#define IRQSTACK_LIMIT_WARNING		150
#define IRQSTACK_LIMIT_CRITICAL		80
#define CPULOAD_LIMIT_WARNING		80
#define CPULOAD_LIMIT_CRITICAL		95

/*
 * This has been calibrated 2013/03/11 using next @ 6d21c7a590619ebbc074e60cab5e134e65c9d32b.
 * Calibration has been done by disabling the init task, breaking into debugger after
 * approximately after 60 seconds, then doing the following math:
 *
 * IDLE_COUNTS_PER_SEC_AT_NO_LOAD = (uint32_t)((double)idleCounter / xTickCount * 1000 + 0.5)
 *
 * This has to be redone every time the toolchain, toolchain flags or FreeRTOS
 * configuration like number of task priorities or similar changes.
 * A change in the cpu load calculation or the idle task handler will invalidate this as well.
 */
#define IDLE_COUNTS_PER_SEC_AT_NO_LOAD (6984538)

#define REVOLUTION

#endif /* plop_CONFIG_H */
/**
 * @}
 * @}
 */
