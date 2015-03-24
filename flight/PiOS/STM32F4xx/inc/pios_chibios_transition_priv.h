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

#define WWDG_IRQHandler                   Vector40 // Window WatchDog
#define PVD_IRQHandler                    Vector44 // PVD through EXTI Line detection
#define TAMP_STAMP_IRQHandler             Vector48 // Tamper and TimeStamps through the EXTI line
#define RTC_WKUP_IRQHandler               Vector4C // RTC Wakeup through the EXTI line
#define FLASH_IRQHandler                  Vector50 // FLASH
#define RCC_IRQHandler                    Vector54 // RCC
#define EXTI0_IRQHandler                  Vector58 // EXTI Line0
#define EXTI1_IRQHandler                  Vector5C // EXTI Line1
#define EXTI2_IRQHandler                  Vector60 // EXTI Line2
#define EXTI3_IRQHandler                  Vector64 // EXTI Line3
#define EXTI4_IRQHandler                  Vector68 // EXTI Line4
#define DMA1_Stream0_IRQHandler           Vector6C // DMA1 Stream 0
#define DMA1_Stream1_IRQHandler           Vector70 // DMA1 Stream 1
#define DMA1_Stream2_IRQHandler           Vector74 // DMA1 Stream 2
#define DMA1_Stream3_IRQHandler           Vector78 // DMA1 Stream 3
#define DMA1_Stream4_IRQHandler           Vector7C // DMA1 Stream 4
#define DMA1_Stream5_IRQHandler           Vector80 // DMA1 Stream 5
#define DMA1_Stream6_IRQHandler           Vector84 // DMA1 Stream 6
#define ADC_IRQHandler                    Vector88 // ADC1, ADC2 and ADC3s
#define CAN1_TX_IRQHandler                Vector8C // CAN1 TX
#define CAN1_RX0_IRQHandler               Vector90 // CAN1 RX0
#define CAN1_RX1_IRQHandler               Vector94 // CAN1 RX1
#define CAN1_SCE_IRQHandler               Vector98 // CAN1 SCE
#define EXTI9_5_IRQHandler                Vector9C // External Line[9:5]s
#define TIM1_BRK_TIM9_IRQHandler          VectorA0 // TIM1 Break and TIM9
#define TIM1_UP_TIM10_IRQHandler          VectorA4 // TIM1 Update and TIM10
#define TIM1_TRG_COM_TIM11_IRQHandler     VectorA8 // TIM1 Trigger and Commutation and TIM11
#define TIM1_CC_IRQHandler                VectorAC // TIM1 Capture Compare
#define TIM2_IRQHandler                   VectorB0 // TIM2
#define TIM3_IRQHandler                   VectorB4 // TIM3
#define TIM4_IRQHandler                   VectorB8 // TIM4
#define I2C1_EV_IRQHandler                VectorBC // I2C1 Event
#define I2C1_ER_IRQHandler                VectorC0 // I2C1 Error
#define I2C2_EV_IRQHandler                VectorC4 // I2C2 Event
#define I2C2_ER_IRQHandler                VectorC8 // I2C2 Error
#define SPI1_IRQHandler                   VectorCC // SPI1
#define SPI2_IRQHandler                   VectorD0 // SPI2
#define USART1_IRQHandler                 VectorD4 // USART1
#define USART2_IRQHandler                 VectorD8 // USART2
#define USART3_IRQHandler                 VectorDC // USART3
#define EXTI15_10_IRQHandler              VectorE0 // External Line[15:10]s
#define RTC_Alarm_IRQHandler              VectorE4 // RTC Alarm (A and B) through EXTI Line
#define OTG_FS_WKUP_IRQHandler            VectorE8 // USB OTG FS Wakeup through EXTI line
#define TIM8_BRK_TIM12_IRQHandler         VectorEC // TIM8 Break and TIM12
#define TIM8_UP_TIM13_IRQHandler          VectorF0 // TIM8 Update and TIM13
#define TIM8_TRG_COM_TIM14_IRQHandler     VectorF4 // TIM8 Trigger and Commutation and TIM14
#define TIM8_CC_IRQHandler                VectorF8 // TIM8 Capture Compare
#define DMA1_Stream7_IRQHandler           VectorFC // DMA1 Stream7
#define FSMC_IRQHandler                   Vector100 // FSMC
#define SDIO_IRQHandler                   Vector104 // SDIO
#define TIM5_IRQHandler                   Vector108 // TIM5
#define SPI3_IRQHandler                   Vector10C // SPI3
#define USART4_IRQHandler                 Vector110 // UART4
#define USART5_IRQHandler                 Vector114 // UART5
#define TIM6_DAC_IRQHandler               Vector118 // TIM6 and DAC1&2 underrun errors
#define TIM7_IRQHandler                   Vector11C // TIM7
#define DMA2_Stream0_IRQHandler           Vector120 // DMA2 Stream 0
#define DMA2_Stream1_IRQHandler           Vector124 // DMA2 Stream 1
#define DMA2_Stream2_IRQHandler           Vector128 // DMA2 Stream 2
#define DMA2_Stream3_IRQHandler           Vector12C // DMA2 Stream 3
#define DMA2_Stream4_IRQHandler           Vector130 // DMA2 Stream 4
#define ETH_IRQHandler                    Vector134 // Ethernet
#define ETH_WKUP_IRQHandler               Vector138 // Ethernet Wakeup through EXTI line
#define CAN2_TX_IRQHandler                Vector13C // CAN2 TX
#define CAN2_RX0_IRQHandler               Vector140 // CAN2 RX0
#define CAN2_RX1_IRQHandler               Vector144 // CAN2 RX1
#define CAN2_SCE_IRQHandler               Vector148 // CAN2 SCE
#define OTG_FS_IRQHandler                 Vector14C // USB OTG FS
#define DMA2_Stream5_IRQHandler           Vector150 // DMA2 Stream 5
#define DMA2_Stream6_IRQHandler           Vector154 // DMA2 Stream 6
#define DMA2_Stream7_IRQHandler           Vector158 // DMA2 Stream 7
#define USART6_IRQHandler                 Vector15C // USART6
#define I2C3_EV_IRQHandler                Vector160 // I2C3 event
#define I2C3_ER_IRQHandler                Vector164 // I2C3 error
#define OTG_HS_EP1_OUT_IRQHandler         Vector168 // USB OTG HS End Point 1 Out
#define OTG_HS_EP1_IN_IRQHandler          Vector16C // USB OTG HS End Point 1 In
#define OTG_HS_WKUP_IRQHandler            Vector170 // USB OTG HS Wakeup through EXTI
#define OTG_HS_IRQHandler                 Vector174 // USB OTG HS
#define DCMI_IRQHandler                   Vector178 // DCMI
#define CRYP_IRQHandler                   Vector17C // CRYP crypto
#define HASH_RNG_IRQHandler               Vector180 // Hash and Rng
#define FPU_IRQHandler                    Vector184 // FPU

#endif /* PIOS_CHIBIOS_TRANSITION_PRIV_H_ */
