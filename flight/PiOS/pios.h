/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 *
 * @file       pios.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Main PiOS header to include all the compiled in PiOS options
 *
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
#include "pios_config.h"

#if defined(PIOS_INCLUDE_FREERTOS)
/* FreeRTOS Includes */
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#endif

/* C Lib Includes */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

/* STM32 Std Perf Lib */
#if defined(STM32F4XX)
# include <stm32f4xx.h>
# include <stm32f4xx_rcc.h>
#elif defined(STM32F30X)
#include <stm32f30x.h>
#include <stm32f30x_rcc.h>
#elif defined(STM32F2XX)
#include <stm32f2xx.h>
#include <stm32f2xx_syscfg.h>
#else
#include <stm32f10x.h>
#endif

#if defined(PIOS_INCLUDE_SDCARD)
/* Dosfs Includes */
#include <dosfs.h>

/* Mass Storage Device Includes */
//#include <msd.h>
#endif

/* Generic initcall infrastructure */
#if defined(PIOS_INCLUDE_INITCALL)
#include "pios_initcall.h"
#endif

/* PIOS Board Specific Device Configuration */
#include "pios_board.h"

/* PIOS Hardware Includes (STM32F10x) */
#include <pios_sys.h>
#include <pios_delay.h>
#include <pios_led.h>
#include <pios_sdcard.h>
#include <pios_usart.h>
#include <pios_irq.h>
#include <pios_adc.h>
#include <pios_internal_adc.h>
#include <pios_servo.h>
#include <pios_brushless.h>
#include <pios_rtc.h>
#include <pios_i2c.h>
#include <pios_can.h>
#include <pios_spi.h>
#include <pios_overo.h>
#include <pios_ppm.h>
#include <pios_pwm.h>
#include <pios_rcvr.h>
#if defined(PIOS_INCLUDE_DMA_CB_SUBSCRIBING_FUNCTION)
#include <pios_dma.h>
#endif
#if defined(PIOS_INCLUDE_FREERTOS)
#include <pios_sensors.h>
#endif
#include <pios_dsm.h>
#include <pios_sbus.h>
#include <pios_usb_hid.h>
#include <pios_debug.h>
#include <pios_gpio.h>
#include <pios_exti.h>
#include <pios_wdg.h>

/* PIOS Hardware Includes (Common) */
#include <pios_heap.h>
#include <pios_sdcard.h>
#include <pios_com.h>
#if defined(PIOS_INCLUDE_MPXV7002)
#include <pios_mpxv7002.h>
#endif
#if defined(PIOS_INCLUDE_MPXV5004)
#include <pios_mpxv5004.h>
#endif
#if defined(PIOS_INCLUDE_ETASV3)
#include <pios_etasv3.h>
#endif
#if defined(PIOS_INCLUDE_BMP085)
#include <pios_bmp085.h>
#endif
#if defined(PIOS_INCLUDE_HCSR04)
#include <pios_hcsr04.h>
#endif
#if defined(PIOS_INCLUDE_HMC5843)
#include <pios_hmc5843.h>
#endif
#if defined(PIOS_INCLUDE_HMC5983)
#include <pios_hmc5983.h>
#endif
#if defined(PIOS_INCLUDE_I2C_ESC)
#include <pios_i2c_esc.h>
#endif
#if defined(PIOS_INCLUDE_IMU3000)
#include <pios_imu3000.h>
#endif
#if defined(PIOS_INCLUDE_MPU6050)
#include <pios_mpu6050.h>
#endif
#if defined(PIOS_INCLUDE_MPU9150)
#include <pios_mpu9150.h>
#endif
#if defined(PIOS_INCLUDE_MPU6000)
#include <pios_mpu6000.h>
#endif
#if defined(PIOS_INCLUDE_L3GD20)
#include <pios_l3gd20.h>
#endif
#if defined(PIOS_INCLUDE_LSM303)
#include <pios_lsm303.h>
#endif
#if defined(PIOS_INCLUDE_MS5611)
#include <pios_ms5611.h>
#endif
#if defined(PIOS_INCLUDE_MS5611_SPI)
#include <pios_ms5611_spi.h>
#endif
#if defined(PIOS_INCLUDE_IAP)
#include <pios_iap.h>
#endif
#if defined(PIOS_INCLUDE_ADXL345)
#include <pios_adxl345.h>
#endif
#if defined(PIOS_INCLUDE_BMA180)
#include <pios_bma180.h>
#endif
#if defined(PIOS_INCLUDE_VIDEO)
#include <pios_video.h>
#endif
#if defined(PIOS_INCLUDE_WAVE)
#include <pios_wavplay.h>
#endif

#if defined(PIOS_INCLUDE_FLASH)
#include <pios_flash.h>
#include <pios_flashfs.h>
#endif

#if defined(PIOS_INCLUDE_BL_HELPER)
#include <pios_bl_helper.h>
#endif

#if defined(PIOS_INCLUDE_USB)
#include <pios_usb.h>
#endif

#if defined(PIOS_INCLUDE_RFM22B)
#include <pios_rfm22b.h>
#ifdef PIOS_INCLUDE_RFM22B_COM
#include <pios_rfm22b_com.h>
#endif
#endif

#include <pios_crc.h>

#define NELEMENTS(x) (sizeof(x) / sizeof(*(x)))

// portTICK_RATE_MS is in [ms/tick].
// See http://sourceforge.net/tracker/?func=detail&aid=3498382&group_id=111543&atid=659636
#define TICKS2MS(t)	((t) * (portTICK_RATE_MS))
#define MS2TICKS(m)	((m) / (portTICK_RATE_MS))

#endif /* PIOS_H */

/**
 * @}
 */
