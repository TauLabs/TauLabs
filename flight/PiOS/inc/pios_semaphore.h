/**
 ******************************************************************************
 * @file       pios_semaphore.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_Semaphore Semaphore Abstraction
 * @{
 * @brief Abstracts the concept of a binary semaphore to hide different implementations
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


#ifndef PIOS_SEMAPHORE_H_
#define PIOS_SEMAPHORE_H_

#define PIOS_SEMAPHORE_TIMEOUT_MAX 0xffffffff

#include <stdint.h>
#include <stdbool.h>

struct pios_semaphore
{
#if defined(PIOS_INCLUDE_FREERTOS)
	uintptr_t sema_handle;
#elif defined(PIOS_INCLUDE_CHIBIOS)
	BinarySemaphore sema;
#elif defined(PIOS_INCLUDE_IRQ)
	uint32_t sema_count;
#endif /* defined(PIOS_INCLUDE_IRQ) */
};

/*
 * The following functions implement the concept of a binary semaphore usable
 * with PIOS_INCLUDE_FREERTOS, PIOS_INCLUDE_CHIBIOS or PIOS_INCLUDE_IRQ.
 *
 * Note that this is not the same as:
 * - counting semaphore
 * - mutex
 * - recursive mutex
 *
 * see FreeRTOS documentation for details: http://www.freertos.org/a00113.html
 * see ChibiOS documentation for details: http://chibios.sourceforge.net/html/group__synchronization.html
 */

struct pios_semaphore *PIOS_Semaphore_Create(void);
bool PIOS_Semaphore_Take(struct pios_semaphore *sema, uint32_t timeout_ms);
bool PIOS_Semaphore_Give(struct pios_semaphore *sema);

bool PIOS_Semaphore_Take_FromISR(struct pios_semaphore *sema, bool *woken);
bool PIOS_Semaphore_Give_FromISR(struct pios_semaphore *sema, bool *woken);

#endif /* PIOS_SEMAPHORE_H_ */

/**
  * @}
  * @}
  */
