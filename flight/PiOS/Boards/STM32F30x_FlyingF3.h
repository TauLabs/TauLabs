/******************************************************************************
 * @file       STM32F4xx_FlyingF3.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @addtogroup PhoenixPilotSystem PhoenixPilot System
 * @{
 * @addtogroup OpenPilotCore OpenPilot Core
 * @{
 * @brief PiOS configuration header for flying f3 board.
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


#ifndef STM3210E_INS_H_
#define STM3210E_INS_H_

#include <stdbool.h>

#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
#define DEBUG_LEVEL 0
#define DEBUG_PRINTF(level, ...) {if(level <= DEBUG_LEVEL && pios_com_debug_id > 0) { PIOS_COM_SendFormattedStringNonBlocking(pios_com_debug_id, __VA_ARGS__); }}
#else
#define DEBUG_PRINTF(level, ...)
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */

#define PIOS_ADC_MAX_OVERSAMPLING               1
#define PIOS_ADC_USE_ADC2                       0

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
// DMA Channels Used
//------------------------
/* Channel 1  -                                 */
/* Channel 2  - SPI1 RX                         */
/* Channel 3  - SPI1 TX                         */
/* Channel 4  - SPI2 RX                         */
/* Channel 5  - SPI2 TX                         */
/* Channel 6  -                                 */
/* Channel 7  -                                 */
/* Channel 8  -                                 */
/* Channel 9  -                                 */
/* Channel 10 -                                 */
/* Channel 11 -                                 */
/* Channel 12 -                                 */

//------------------------
// BOOTLOADER_SETTINGS
//------------------------
#define BOARD_READABLE					true
#define BOARD_WRITABLE					true
#define MAX_DEL_RETRYS					3


//------------------------
// PIOS_LED
//------------------------
#define PIOS_LED_BLUE_SE				0
#define PIOS_LED_RED_S					1
#define PIOS_LED_ORANGE_SW				2
#define PIOS_LED_GREEN_W				3
#define PIOS_LED_BLUE_NW				4
#define PIOS_LED_RED_N					5
#define PIOS_LED_ORANGE_NE				6
#define PIOS_LED_GREEN_E				7

#define PIOS_LED_HEARTBEAT				PIOS_LED_BLUE_SE
#define PIOS_LED_ALARM					PIOS_LED_RED_S
#define PIOS_LED_USB					PIOS_LED_GREEN_W

#define USB_LED_ON						PIOS_LED_On(PIOS_LED_USB)
#define USB_LED_OFF						PIOS_LED_Off(PIOS_LED_USB)
#define USB_LED_TOGGLE					PIOS_LED_Toggle(PIOS_LED_USB)

//-------------------------
// PIOS_SPI
// See also pios_board.c
//-------------------------
#define PIOS_SPI_MAX_DEVS				1

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
extern uint32_t pios_i2c_external_id;
#define PIOS_I2C_MAIN_ADAPTER			(pios_i2c_external_id)	//this is dirty and should be removed in favor a cleaner sensor api

//-------------------------
// PIOS_USART
//
// See also pios_board.c
//-------------------------
#define PIOS_USART_MAX_DEVS				3

//-------------------------
// PIOS_COM
//
// See also pios_board.c
//-------------------------
#define PIOS_COM_MAX_DEVS               4
extern uintptr_t pios_com_telem_rf_id;
extern uintptr_t pios_com_gps_id;
extern uintptr_t pios_com_telem_usb_id;
extern uintptr_t pios_com_bridge_id;
extern uintptr_t pios_com_vcp_id;
#define PIOS_COM_GPS                    (pios_com_gps_id)
#define PIOS_COM_TELEM_USB              (pios_com_telem_usb_id)
#define PIOS_COM_TELEM_RF               (pios_com_telem_rf_id)
#define PIOS_COM_BRIDGE                 (pios_com_bridge_id)
#define PIOS_COM_VCP                    (pios_com_vcp_id)

#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
extern uintptr_t pios_com_debug_id;
#define PIOS_COM_DEBUG                  (pios_com_debug_id)
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */



//------------------------
// TELEMETRY
//------------------------
#define TELEM_QUEUE_SIZE				40
#define PIOS_TELEM_STACK_SIZE			768


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
#define PIOS_RCVR_MAX_DEVS				3
#define PIOS_RCVR_MAX_CHANNELS			12
#define PIOS_GCSRCVR_TIMEOUT_MS			100

//-------------------------
// Receiver PPM input
//-------------------------
#define PIOS_PPM_MAX_DEVS				1
#define PIOS_PPM_NUM_INPUTS				12

//-------------------------
// Receiver PWM input
//-------------------------
#define PIOS_PWM_MAX_DEVS				1
#define PIOS_PWM_NUM_INPUTS				10

//-------------------------
// Receiver DSM input
//-------------------------
#define PIOS_DSM_MAX_DEVS				2
#define PIOS_DSM_NUM_INPUTS				12

//-------------------------
// Receiver S.Bus input
//-------------------------
#define PIOS_SBUS_MAX_DEVS				1
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
// None.
//-------------------------
#define PIOS_INTERNAL_ADC_COUNT                         4
#define PIOS_INTERNAL_ADC_MAPPING                { ADC1, ADC2, ADC3, ADC4 }
#define PIOS_INTERNAL_ADC_MAX_INSTANCES                 4

//-------------------------
// USB
//-------------------------
#define PIOS_USB_MAX_DEVS				1
#define PIOS_USB_ENABLED				1 /* Should remove all references to this */
#define PIOS_USB_HID_MAX_DEVS			1

//-------------------------
// DMA
//-------------------------
#define PIOS_DMA_MAX_CHANNELS                   12
#define PIOS_DMA_MAX_HANDLERS_PER_CHANNEL       3
#define PIOS_DMA_CHANNELS {DMA1_Channel1, DMA1_Channel2, DMA1_Channel3, DMA1_Channel4, DMA1_Channel5, DMA1_Channel6, DMA1_Channel7, DMA2_Channel1, DMA2_Channel2, DMA2_Channel3, DMA2_Channel4, DMA2_Channel5}
#endif /* STM3210E_INS_H_ */
