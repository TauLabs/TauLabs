/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup Naze32 Tau Labs naze32 support files
 * @{
 *
 * @file       pios_board.h 
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Board header file for Naze32
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


#ifndef STM32103CB_NAZE32_H_
#define STM32103CB_NAZE32_H_

#include <stdbool.h>

//------------------------
// Timers and Channels Used
//------------------------
/*
Timer | Channel 1 | Channel 2 | Channel 3 | Channel 4
------+-----------+-----------+-----------+----------
TIM1  |  Servo 4  |           |           |
TIM2  |  RC In 5  |  RC In 6  |  Servo 6  |
TIM3  |  Servo 5  |  RC In 2  |  RC In 3  |  RC In 4
TIM4  |  RC In 1  |  Servo 3  |  Servo 2  |  Servo 1
------+-----------+-----------+-----------+----------
*/

//------------------------
// DMA Channels Used
//------------------------
/* Channel 1  -                                 */
/* Channel 2  -                                 */
/* Channel 3  -                                 */
/* Channel 4  -                                 */
/* Channel 5  -                                 */
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
#define PIOS_LED_HEARTBEAT				0
#define PIOS_LED_ALARM					1

//------------------------
// PIOS_WDG
//------------------------
#define PIOS_WATCHDOG_TIMEOUT			250
#define PIOS_WDG_REGISTER				BKP_DR4

//------------------------
// PIOS_I2C
// See also pios_board.c
//------------------------
#define PIOS_I2C_MAX_DEVS				1

//-------------------------
// PIOS_COM
//
// See also pios_board.c
//-------------------------
extern uintptr_t pios_com_telem_rf_id;
extern uintptr_t pios_com_gps_id;
extern uintptr_t pios_com_bridge_id;
extern uintptr_t pios_com_mavlink_id;

#define PIOS_COM_GPS                    (pios_com_gps_id)
#define PIOS_COM_TELEM_RF               (pios_com_telem_rf_id)
#define PIOS_COM_BRIDGE                 (pios_com_bridge_id)
#define PIOS_COM_MAVLINK                (pios_com_mavlink_id)

//------------------------
// TELEMETRY
//------------------------
#define TELEM_QUEUE_SIZE				20

//-------------------------
// System Settings
// 
// See also system_stm32f10x.c
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
#define PIOS_IRQ_PRIO_EXTREME			0 		// for I2C

//------------------------
// PIOS_RCVR
// See also pios_board.c
//------------------------
#define PIOS_RCVR_MAX_CHANNELS			12
#define PIOS_GCSRCVR_TIMEOUT_MS			100

//-------------------------
// Receiver PPM input
//-------------------------
#define PIOS_PPM_NUM_INPUTS				8

//-------------------------
// Receiver PWM input
//-------------------------
#define PIOS_PWM_NUM_INPUTS				8

//-------------------------
// Receiver DSM input
//-------------------------
#define PIOS_DSM_NUM_INPUTS				12

//-------------------------
// Servo outputs
//-------------------------
#define PIOS_SERVO_UPDATE_HZ			50
#define PIOS_SERVOS_INITIAL_POSITION	0 /* dont want to start motors, have no pulse till settings loaded */

//--------------------------
// Timer controller settings
//--------------------------
#define PIOS_TIM_MAX_DEVS				3

//-------------------------
// ADC
//-------------------------

//-------------------------
// GPIO
//-------------------------
#define PIOS_GPIO_PORTS				{  }
#define PIOS_GPIO_PINS				{  }
#define PIOS_GPIO_CLKS				{  }
#define PIOS_GPIO_NUM				0

#endif /* STM32103CB_CC_H_ */

/**
 * @}
 * @}
 */
