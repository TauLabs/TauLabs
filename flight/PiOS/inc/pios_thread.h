/**
 ******************************************************************************
 * @file       pios_thread.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_Thread Thread Abstraction
 * @{
 * @brief Abstracts the concept of a thread to hide different implementations
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

#ifndef PIOS_THREAD_H_
#define PIOS_THREAD_H_

#define PIOS_THREAD_TIMEOUT_MAX 0xffffffff

#include <stdint.h>
#include <stdbool.h>

#if defined(PIOS_INCLUDE_FREERTOS)

#include "FreeRTOSConfig.h"

enum pios_thread_prio_e
{
	PIOS_THREAD_PRIO_LOW = 1,
	PIOS_THREAD_PRIO_NORMAL = 2,
	PIOS_THREAD_PRIO_HIGH = 3,
	PIOS_THREAD_PRIO_HIGHEST = 4, /* @note: this has to match (configMAX_PRIORITIES - 1) */
};

#define PIOS_THREAD_STACK_SIZE_MIN (configMINIMAL_STACK_SIZE * 4)

struct pios_thread
{
	uintptr_t task_handle;
};

#elif defined(PIOS_INCLUDE_CHIBIOS)

#include "ch.h"

enum pios_thread_prio_e
{
	PIOS_THREAD_PRIO_LOW = LOWPRIO,
	PIOS_THREAD_PRIO_NORMAL = NORMALPRIO,
	PIOS_THREAD_PRIO_HIGH = NORMALPRIO + 32,
	PIOS_THREAD_PRIO_HIGHEST = HIGHPRIO,
};

#define PIOS_THREAD_STACK_SIZE_MIN THD_WA_SIZE(4096 + PORT_INT_REQUIRED_STACK)

struct pios_thread
{
	Thread *threadp;
};

#endif /* defined(PIOS_INCLUDE_CHIBIOS) */

/*
 * The following functions implement the concept of a thread usable
 * with PIOS_INCLUDE_FREERTOS.
 *
 * see FreeRTOS documentation for details: http://www.freertos.org/a00019.html
 */

struct pios_thread *PIOS_Thread_Create(void (*fp)(void *), const char *namep, size_t stack_bytes, void *argp, enum pios_thread_prio_e prio);
void PIOS_Thread_Delete(struct pios_thread *threadp);
uint32_t PIOS_Thread_Systime(void);
void PIOS_Thread_Sleep(uint32_t time_ms);
void PIOS_Thread_Sleep_Until(uint32_t *previous_ms, uint32_t increment_ms);
uint32_t PIOS_Thread_Get_Stack_Usage(struct pios_thread *threadp);
uint32_t PIOS_Thread_Get_Runtime(struct pios_thread *threadp);
void PIOS_Thread_Scheduler_Suspend(void);
void PIOS_Thread_Scheduler_Resume(void);

#endif /* PIOS_THREAD_H_ */

/**
  * @}
  * @}
  */
