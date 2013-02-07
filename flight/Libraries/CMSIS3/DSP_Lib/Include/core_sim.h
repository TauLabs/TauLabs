/**************************************************************************//**
 * @file     core_sim.h
 * @brief    CMSIS Cortex-SIM Core Peripheral Access Layer Header File
 * @version  V3.01
 * @date     13. March 2012
 *
 * @note
 * Copyright (C) 2009-2012 ARM Limited. All rights reserved.
 *
 * @par
 * ARM Limited (ARM) is supplying this software for use with Cortex-M
 * processor based microcontrollers.  This file can be freely distributed
 * within development tools that are supporting such ARM based processors.
 *
 * @par
 * THIS SOFTWARE IS PROVIDED "AS IS".  NO WARRANTIES, WHETHER EXPRESS, IMPLIED
 * OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE.
 * ARM SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR
 * CONSEQUENTIAL DAMAGES, FOR ANY REASON WHATSOEVER.
 *
 ******************************************************************************/
#if defined ( __ICCARM__ )
 #pragma system_include  /* treat file as system include file for MISRA check */
#endif

#ifdef __cplusplus
 extern "C" {
#endif

#ifndef __CORE_SIM_H_GENERIC
#define __CORE_SIM_H_GENERIC


/*******************************************************************************
 *                 CMSIS definitions
 ******************************************************************************/
/** \ingroup Cortex_SIM
  @{
 */

/*  CMSIS SIM definitions */
#define __SIM_CMSIS_VERSION_MAIN  (0x03)                                   /*!< [31:16] CMSIS HAL main version   */
#define __SIM_CMSIS_VERSION_SUB   (0x01)                                   /*!< [15:0]  CMSIS HAL sub version    */
#define __SIM_CMSIS_VERSION       ((__SIM_CMSIS_VERSION_MAIN << 16) | \
                                    __SIM_CMSIS_VERSION_SUB          )     /*!< CMSIS HAL version number         */

#define __CORTEX_M                (0x00)                                   /*!< Cortex-M Core                    */


#if   defined ( __CC_ARM )
  #define __ASM            __asm                                      /*!< asm keyword for ARM Compiler          */
  #define __INLINE         __inline                                   /*!< inline keyword for ARM Compiler       */
  #define __STATIC_INLINE  static __inline

#elif defined ( __ICCARM__ )
  #define __ASM            __asm                                      /*!< asm keyword for IAR Compiler          */
  #define __INLINE         inline                                     /*!< inline keyword for IAR Compiler. Only available in High optimization mode! */
  #define __STATIC_INLINE  static inline

#elif defined ( __GNUC__ )
  #define __ASM            __asm                                      /*!< asm keyword for GNU Compiler          */
  #define __INLINE         inline                                     /*!< inline keyword for GNU Compiler       */
  #define __STATIC_INLINE  static inline

#elif defined ( __TASKING__ )
  #define __ASM            __asm                                      /*!< asm keyword for TASKING Compiler      */
  #define __INLINE         inline                                     /*!< inline keyword for TASKING Compiler   */
  #define __STATIC_INLINE  static inline

#endif

#include <stdint.h>                      /* standard types definitions                      */

#endif /* __CORE_SIM_H_GENERIC */

#ifndef __CMSIS_GENERIC

#ifndef __CORE_SIM_H_DEPENDANT
#define __CORE_SIM_H_DEPENDANT

/* check device defines and use defaults */
#if defined __CHECK_DEVICE_DEFINES
  #ifndef __SIM_REV
    #define __SIM_REV               0x0000
    #warning "__SIM_REV not defined in device header file; using default!"
  #endif
#endif


#ifdef __cplusplus
  #define   __I     volatile             /*!< Defines 'read only' permissions                 */
#else
  #define   __I     volatile const       /*!< Defines 'read only' permissions                 */
#endif
#define     __O     volatile             /*!< Defines 'write only' permissions                */
#define     __IO    volatile             /*!< Defines 'read / write' permissions              */

/*@} end of group Cortex_SIM */


#endif /* __CORE_SIM_H_DEPENDANT */

#endif /* __CMSIS_GENERIC */

#ifdef __cplusplus
}
#endif
