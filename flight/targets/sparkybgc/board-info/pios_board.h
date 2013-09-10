/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup SparkyBGC Tau Labs Sparky BGC support files
 * @{
 *
 * @file       pios_board.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Board header file for Sparky BGC board.
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


#ifndef STM32F30X_SPARKY_H_
#define STM32F30X_SPARKY_H_

#include <stdbool.h>

#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
#define DEBUG_LEVEL 0
#define DEBUG_PRINTF(level, ...) {if(level <= DEBUG_LEVEL && pios_com_aux_id > 0) { PIOS_COM_SendFormattedStringNonBlocking(pios_com_aux_id, __VA_ARGS__); }}
#else
#define DEBUG_PRINTF(level, ...)
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */

//------------------------
// Timers and Channels Used
//------------------------
/*
Timer | Channel 1 | Channel 2 | Channel 3 | Channel 4
------+-----------+-----------+-----------+----------
TIM1  |           |           |           |
TIM2  | --------------- PIOS_DELAY -----------------
TIM3  |           |           |           |
TIM4  |           |           |           |
TIM5  |           |           |           |
TIM6  |           |           |           |
TIM7  |           |           |           |
TIM8  |           |           |           |
------+-----------+-----------+-----------+----------
*/


//------------------------
// BOOTLOADER_SETTINGS
//------------------------
#define BOARD_READABLE					true
#define BOARD_WRITABLE					true
#define MAX_DEL_RETRYS					3


//------------------------
// PIOS_LED
//------------------------
#define PIOS_LED_HEARTBEAT				0
#define PIOS_LED_ALARM					1


//------------------------
// PIOS_WDG
//------------------------
#define PIOS_WATCHDOG_TIMEOUT			250
#define PIOS_WDG_REGISTER				RTC_BKP_DR4

//------------------------
// PIOS_I2C
// See also pios_board.c
//------------------------
#define PIOS_I2C_MAX_DEVS				2
extern uint32_t pios_i2c_flexi_id;
#define PIOS_I2C_MAIN_ADAPTER			(pios_i2c_flexi_id)	//this is dirty and should be removed in favor a cleaner sensor api


//-------------------------
// PIOS_COM
//
// See also pios_board.c
//-------------------------
extern uintptr_t pios_com_telem_rf_id;
extern uintptr_t pios_com_gps_id;
extern uintptr_t pios_com_telem_usb_id;
extern uintptr_t pios_com_bridge_id;
extern uintptr_t pios_com_vcp_id;
extern uintptr_t pios_com_mavlink_id;
#define PIOS_COM_GPS                    (pios_com_gps_id)
#define PIOS_COM_TELEM_USB              (pios_com_telem_usb_id)
#define PIOS_COM_TELEM_RF               (pios_com_telem_rf_id)
#define PIOS_COM_BRIDGE                 (pios_com_bridge_id)
#define PIOS_COM_VCP                    (pios_com_vcp_id)
#define PIOS_COM_MAVLINK                (pios_com_mavlink_id)

#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
extern uintptr_t pios_com_debug_id;
#define PIOS_COM_DEBUG                  (pios_com_debug_id)
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */



//------------------------
// TELEMETRY
//------------------------
#define TELEM_QUEUE_SIZE				80
#define PIOS_TELEM_STACK_SIZE			624


//-------------------------
// System Settings
// 
// See also system_stm32f30x.c
//-------------------------
//These macros are deprecated

#define PIOS_SYSCLK									72000000
#define PIOS_PERIPHERAL_APB1_CLOCK					(PIOS_SYSCLK / 2)
#define PIOS_PERIPHERAL_APB2_CLOCK					(PIOS_SYSCLK / 1)


//-------------------------
// Interrupt Priorities
//-------------------------
#define PIOS_IRQ_PRIO_LOW				12              // lower than RTOS
#define PIOS_IRQ_PRIO_MID				8               // higher than RTOS
#define PIOS_IRQ_PRIO_HIGH				5               // for SPI, ADC, I2C etc...
#define PIOS_IRQ_PRIO_HIGHEST			4               // for USART etc...


//------------------------
// PIOS_RCVR
// See also pios_board.c
//------------------------
#define PIOS_RCVR_MAX_CHANNELS			12
#define PIOS_GCSRCVR_TIMEOUT_MS			100

//-------------------------
// Receiver PPM input
//-------------------------
#define PIOS_PPM_NUM_INPUTS				12

//-------------------------
// Receiver PWM input
//-------------------------
#define PIOS_PWM_NUM_INPUTS				10

//-------------------------
// Receiver DSM input
//-------------------------
#define PIOS_DSM_NUM_INPUTS				12

//-------------------------
// Receiver S.Bus input
//-------------------------
#define PIOS_SBUS_NUM_INPUTS			(16+2)

//-------------------------
// Servo outputs
//-------------------------
#define PIOS_SERVO_UPDATE_HZ			50
#define PIOS_SERVOS_INITIAL_POSITION	0 /* dont want to start motors, have no pulse till settings loaded */

//--------------------------
// Timer controller settings
//--------------------------
#define PIOS_TIM_MAX_DEVS				8

//-------------------------
// ADC
// ADC0 : PA1 ADC1_IN2
// ADC1 : PA4 ADC2_IN1
// ADC2 : PA7 ADC2_IN4 (disabled by default and should have external resistor)
//-------------------------
#define PIOS_INTERNAL_ADC_MAX_INSTANCES          4
#define PIOS_INTERNAL_ADC_COUNT                  4
#define PIOS_INTERNAL_ADC_MAPPING                { ADC1, ADC2, ADC3, ADC4 }

#if defined(PIOS_INCLUDE_ADC)
extern uintptr_t pios_internal_adc_id;
#define PIOS_INTERNAL_ADC                               (pios_internal_adc_id)
#endif
#define PIOS_ADC_SUB_DRIVER_MAX_INSTANCES       3

//-------------------------
// DMA
//-------------------------
#define PIOS_DMA_MAX_CHANNELS                   12
#define PIOS_DMA_MAX_HANDLERS_PER_CHANNEL       3
#define PIOS_DMA_CHANNELS {DMA1_Channel1, DMA1_Channel2, DMA1_Channel3, DMA1_Channel4, DMA1_Channel5, DMA1_Channel6, DMA1_Channel7, DMA2_Channel1, DMA2_Channel2, DMA2_Channel3, DMA2_Channel4, DMA2_Channel5}

//-------------------------
// USB
//-------------------------
#define PIOS_USB_MAX_DEVS				1
#define PIOS_USB_ENABLED				1 /* Should remove all references to this */
#define PIOS_USB_HID_MAX_DEVS			1


#endif /* STM32F30X_SPARKY_H_ */


/**
 * @}
 * @}
 */
