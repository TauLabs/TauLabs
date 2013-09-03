/**
 ******************************************************************************
 * @file       pios_semaphore.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

#include "pios.h"
#include "pios_semaphore.h"

#if !defined(PIOS_INCLUDE_FREERTOS) && !defined(PIOS_INCLUDE_IRQ)
#error pios_semaphore.c requires either PIOS_INCLUDE_FREERTOS or PIOS_INCLUDE_IRQ to be defined
#endif

struct pios_semaphore *PIOS_Semaphore_Create(void)
{
	struct pios_semaphore *sema = PIOS_malloc(sizeof(struct pios_semaphore));

	if (sema == NULL)
		return NULL;

	/*
	 * The initial state of a binary semaphore is "given".
	 * FreeRTOS executes a "give" upon creation.
	 */
#if defined(PIOS_INCLUDE_FREERTOS)
	vSemaphoreCreateBinary(sema->sema_handle);
#else
	sema->sema_count = 1;
#endif

	return sema;
}

bool PIOS_Semaphore_Take(struct pios_semaphore *sema, uint32_t timeout_ms)
{
	PIOS_Assert(sema != NULL);

#if defined(PIOS_INCLUDE_FREERTOS)
	portTickType timeout_ticks;
	if (timeout_ms == PIOS_SEMAPHORE_TIMEOUT_MAX)
		timeout_ticks = portMAX_DELAY;
	else
		timeout_ticks = MS2TICKS(timeout_ms);

	return xSemaphoreTake(sema->sema_handle, timeout_ticks) == pdTRUE;
#else
	uint32_t start = PIOS_DELAY_GetRaw();

	uint32_t temp_sema_count;
	do {
		PIOS_IRQ_Disable();
		if ((temp_sema_count = sema->sema_count) != 0)
			--sema->sema_count;
		PIOS_IRQ_Enable();
	} while (temp_sema_count == 0 &&
		PIOS_DELAY_DiffuS(start) < timeout_ms * 1000);

	return temp_sema_count != 0;
#endif
}

bool PIOS_Semaphore_Give(struct pios_semaphore *sema)
{
	PIOS_Assert(sema != NULL);
#if defined(PIOS_INCLUDE_FREERTOS)
	return xSemaphoreGive(sema->sema_handle) == pdTRUE;
#else
	bool result = true;

	PIOS_IRQ_Disable();

	if (sema->sema_count == 0)
		++sema->sema_count;
	else
		result = false;

	PIOS_IRQ_Enable();

	return result;
#endif
}

bool PIOS_Semaphore_Take_FromISR(struct pios_semaphore *sema, bool *woken)
{
	PIOS_Assert(sema != NULL);

#if defined(PIOS_INCLUDE_FREERTOS)
	PIOS_Assert(woken != NULL);

	signed portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;

	bool result = xSemaphoreTakeFromISR(sema->sema_handle, &xHigherPriorityTaskWoken) == pdTRUE;

	*woken = *woken || xHigherPriorityTaskWoken == pdTRUE;

	return result;
#else
	bool result = true;

	PIOS_IRQ_Disable();

	if (sema->sema_count != 0)
		--sema->sema_count;
	else
		result = false;

	PIOS_IRQ_Enable();

	return result;
#endif
}

bool PIOS_Semaphore_Give_FromISR(struct pios_semaphore *sema, bool *woken)
{
	PIOS_Assert(sema != NULL);

#if defined(PIOS_INCLUDE_FREERTOS)
	PIOS_Assert(woken != NULL);

	signed portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;

	bool result = xSemaphoreGiveFromISR(sema->sema_handle, &xHigherPriorityTaskWoken) == pdTRUE;

	*woken = *woken || xHigherPriorityTaskWoken == pdTRUE;

	return result;
#else
	bool result = true;

	PIOS_IRQ_Disable();

	if (sema->sema_count == 0)
		++sema->sema_count;
	else
		result = false;

	PIOS_IRQ_Enable();

	return result;
#endif
}

