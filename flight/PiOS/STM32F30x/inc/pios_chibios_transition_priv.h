/**
 ******************************************************************************
 * @file       pios_chibios_transition_priv.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief ChibiOS transition header, mapping ISR names.
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

#ifndef PIOS_CHIBIOS_TRANSITION_PRIV_H_
#define PIOS_CHIBIOS_TRANSITION_PRIV_H_

#define WWDG_IRQHandler                 Vector40 // Window WatchDog
#define PVD_IRQHandler                  Vector44 // PVD through EXTI Line detection
#define TAMP_STAMP_IRQHandler           Vector48 // Tamper and TimeStamps through the EXTI line
#define RTC_WKUP_IRQHandler             Vector4C // RTC Wakeup through the EXTI line
#define FLASH_IRQHandler                Vector50 // FLASH
#define RCC_IRQHandler                  Vector54 // RCC
#define EXTI0_IRQHandler                Vector58 // EXTI Line0
#define EXTI1_IRQHandler                Vector5C // EXTI Line1
#define EXTI2_TS_IRQHandler             Vector60 // EXTI Line2 and Touch Sense Interrupt
#define EXTI3_IRQHandler                Vector64 // EXTI Line3
#define EXTI4_IRQHandler                Vector68 // EXTI Line4
#define DMA1_Channel1_IRQHandler        Vector6C // DMA1 Channel 1
#define DMA1_Channel2_IRQHandler        Vector70 // DMA1 Channel 2
#define DMA1_Channel3_IRQHandler        Vector74 // DMA1 Channel 3
#define DMA1_Channel4_IRQHandler        Vector78 // DMA1 Channel 4
#define DMA1_Channel5_IRQHandler        Vector7C // DMA1 Channel 5
#define DMA1_Channel6_IRQHandler        Vector80 // DMA1 Channel 6
#define DMA1_Channel7_IRQHandler        Vector84 // DMA1 Channel 7
#define ADC1_2_IRQHandler               Vector88 // ADC1 and ADC2
#define USB_HP_CAN1_TX_IRQHandler       Vector8C // USB Device High Priority or CAN1 TX
#define USB_LP_CAN1_RX0_IRQHandler      Vector90 // USB Device Low Priority or CAN1 RX0
#define CAN1_RX1_IRQHandler             Vector94 // CAN1 RX1
#define CAN1_SCE_IRQHandler             Vector98 // CAN1 SCE
#define EXTI9_5_IRQHandler              Vector9C // External Line[9:5]s
#define TIM1_BRK_TIM15_IRQHandler       VectorA0 // TIM1 Break and TIM15
#define TIM1_UP_TIM16_IRQHandler        VectorA4 // TIM1 Update and TIM16
#define TIM1_TRG_COM_TIM17_IRQHandler   VectorA8 // TIM1 Trigger and Commutation and TIM17
#define TIM1_CC_IRQHandler              VectorAC // TIM1 Capture Compare
#define TIM2_IRQHandler                 VectorB0 // TIM2
#define TIM3_IRQHandler                 VectorB4 // TIM3
#define TIM4_IRQHandler                 VectorB8 // TIM4
#define I2C1_EV_EXTI23_IRQHandler       VectorBC // I2C1 Event and EXTI23
#define I2C1_ER_IRQHandler              VectorC0 // I2C1 Error
#define I2C2_EV_EXTI24_IRQHandler       VectorC4 // I2C2 Event and EXTI24
#define I2C2_ER_IRQHandler              VectorC8 // I2C2 Error
#define SPI1_IRQHandler                 VectorCC // SPI1
#define SPI2_IRQHandler                 VectorD0 // SPI2
#define USART1_EXTI25_IRQHandler        VectorD4 // USART1 and EXTI25
#define USART2_EXTI26_IRQHandler        VectorD8 // USART2 and EXTI26
#define USART3_EXTI28_IRQHandler        VectorDC // USART3 and EXTI28
#define EXTI15_10_IRQHandler            VectorE0 // External Line[15:10]s
#define RTC_Alarm_IRQHandler            VectorE4 // RTC Alarm (A and B) through EXTI Line
#define USB_WKUP_IRQHandler             VectorE8 // USB FS Wakeup through EXTI line
#define TIM8_BRK_IRQHandler             VectorEC // TIM8 Break
#define TIM8_UP_IRQHandler              VectorF0 // TIM8 Update
#define TIM8_TRG_COM_IRQHandler         VectorF4 // TIM8 Trigger and Commutation
#define TIM8_CC_IRQHandler              VectorF8 // TIM8 Capture Compare
#define ADC3_IRQHandler                 VectorFC // ADC3
#define SPI3_IRQHandler                 Vector10C // SPI3
#define UART4_EXTI34_IRQHandler         Vector110 // UART4 and EXTI34
#define UART5_EXTI35_IRQHandler         Vector114 // UART5 and EXTI35
#define TIM6_DAC_IRQHandler             Vector118 // TIM6 and DAC1&2 underrun errors
#define TIM7_IRQHandler                 Vector11C // TIM7
#define DMA2_Channel1_IRQHandler        Vector120 // DMA2 Channel 1
#define DMA2_Channel2_IRQHandler        Vector124 // DMA2 Channel 2
#define DMA2_Channel3_IRQHandler        Vector128 // DMA2 Channel 3
#define DMA2_Channel4_IRQHandler        Vector12C // DMA2 Channel 4
#define DMA2_Channel5_IRQHandler        Vector130 // DMA2 Channel 5
#define ADC4_IRQHandler                 Vector134 // ADC4
#define COMP1_2_3_IRQHandler            Vector140 // COMP1, COMP2 and COMP3
#define COMP4_5_6_IRQHandler            Vector144 // COMP4, COMP5 and COMP6
#define COMP7_IRQHandler                Vector148 // COMP7
#define USB_HP_IRQHandler               Vector168 // USB High Priority remap
#define USB_LP_IRQHandler               Vector16C // USB Low Priority remap
#define USB_WKUP_RMP_IRQHandler         Vector170 // USB Wakup remap
#define FPU_IRQHandler                  Vector184 // FPU

#endif /* PIOS_CHIBIOS_TRANSITION_PRIV_H_ */
