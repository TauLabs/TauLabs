/**
 ******************************************************************************
 * @file       pios_semaphore.c
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

#include "pios.h"
#include "pios_semaphore.h"

#if !defined(PIOS_INCLUDE_FREERTOS) && !defined(PIOS_INCLUDE_CHIBIOS) && !defined(PIOS_INCLUDE_IRQ)
#error "pios_semaphore.c requires either PIOS_INCLUDE_FREERTOS, PIOS_INCLUDE_CHIBIOS or PIOS_INCLUDE_IRQ to be defined"
#endif

#if defined(PIOS_INCLUDE_FREERTOS)

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"

// portTICK_RATE_MS is in [ms/tick].
// See http://sourceforge.net/tracker/?func=detail&aid=3498382&group_id=111543&atid=659636
#define TICKS2MS(t) ((t) * (portTICK_RATE_MS))
#define MS2TICKS(m) ((m) / (portTICK_RATE_MS))

/**
 *
 * @brief   Creates a binary semaphore.
 *
 * @returns instance of @p struct pios_semaphore or NULL on failure
 *
 */
struct pios_semaphore *PIOS_Semaphore_Create(void)
{
	struct pios_semaphore *sema = PIOS_malloc(sizeof(struct pios_semaphore));

	if (sema == NULL)
		return NULL;

	/*
	 * The initial state of a binary semaphore is "given".
	 * FreeRTOS executes a "give" upon creation.
	 */
	xSemaphoreHandle temp;
	vSemaphoreCreateBinary(temp);
	sema->sema_handle = (uintptr_t)temp;

	return sema;
}

/**
 *
 * @brief   Takes binary semaphore.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 * @param[in] timeout_ms   timeout for acquiring the lock in milliseconds
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Take(struct pios_semaphore *sema, uint32_t timeout_ms)
{
	PIOS_Assert(sema != NULL);

	portTickType timeout_ticks;
	if (timeout_ms == PIOS_SEMAPHORE_TIMEOUT_MAX)
		timeout_ticks = portMAX_DELAY;
	else
		timeout_ticks = MS2TICKS(timeout_ms);

	return xSemaphoreTake(sema->sema_handle, timeout_ticks) == pdTRUE;
}

/**
 *
 * @brief   Gives binary semaphore.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Give(struct pios_semaphore *sema)
{
	PIOS_Assert(sema != NULL);

	return xSemaphoreGive(sema->sema_handle) == pdTRUE;
}

/* Workaround for simulator version of FreeRTOS. */
#if !defined(SIM_POSIX)
/**
 *
 * @brief   Takes binary semaphore from ISR context.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 * @param[out] woken       pointer to bool which will be set true if a context switch is required
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Take_FromISR(struct pios_semaphore *sema, bool *woken)
{
	PIOS_Assert(sema != NULL);
	PIOS_Assert(woken != NULL);

	signed portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;

	bool result = xSemaphoreTakeFromISR(sema->sema_handle, &xHigherPriorityTaskWoken) == pdTRUE;

	*woken = *woken || xHigherPriorityTaskWoken == pdTRUE;

	return result;
}

/**
 *
 * @brief   Gives binary semaphore from ISR context.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 * @param[out] woken       pointer to bool which will be set true if a context switch is required
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Give_FromISR(struct pios_semaphore *sema, bool *woken)
{
	PIOS_Assert(sema != NULL);
	PIOS_Assert(woken != NULL);

	signed portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;

	bool result = xSemaphoreGiveFromISR(sema->sema_handle, &xHigherPriorityTaskWoken) == pdTRUE;

	*woken = *woken || xHigherPriorityTaskWoken == pdTRUE;

	return result;
}
#endif /* !defined(SIM_POSIX) */

#elif defined(PIOS_INCLUDE_CHIBIOS)

/**
 *
 * @brief   Creates a binary semaphore.
 *
 * @returns instance of @p struct pios_semaphore or NULL on failure
 *
 */
struct pios_semaphore *PIOS_Semaphore_Create(void)
{
	struct pios_semaphore *sema = PIOS_malloc(sizeof(struct pios_semaphore));

	if (sema == NULL)
		return NULL;

	/*
	 * The initial state of a binary semaphore is "given".
	 */
	chBSemInit(&sema->sema, false);

	return sema;
}

/**
 *
 * @brief   Takes binary semaphore.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 * @param[in] timeout_ms   timeout for acquiring the lock in milliseconds
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Take(struct pios_semaphore *sema, uint32_t timeout_ms)
{
	PIOS_Assert(sema != NULL);

	if (timeout_ms == PIOS_SEMAPHORE_TIMEOUT_MAX)
		return chBSemWait(&sema->sema) == RDY_OK;
	else if (timeout_ms == 0)
		return chBSemWaitTimeout(&sema->sema, TIME_IMMEDIATE) == RDY_OK;
	else
		return chBSemWaitTimeout(&sema->sema, MS2ST(timeout_ms)) == RDY_OK;
}

/**
 *
 * @brief   Gives binary semaphore.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Give(struct pios_semaphore *sema)
{
	PIOS_Assert(sema != NULL);

	chBSemSignal(&sema->sema);

	return true;
}

/**
 *
 * @brief   Takes binary semaphore from ISR context.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 * @param[out] woken       pointer to bool which will be set true if a context switch is required
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Take_FromISR(struct pios_semaphore *sema, bool *woken)
{
	/* Waiting on a semaphore within an interrupt is not supported by ChibiOS. */
	PIOS_Assert(false);
	return false;
}

/**
 *
 * @brief   Gives binary semaphore from ISR context.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 * @param[out] woken       pointer to bool which will be set true if a context switch is required
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Give_FromISR(struct pios_semaphore *sema, bool *woken)
{
	PIOS_Assert(sema != NULL);
	PIOS_Assert(woken != NULL);

	chSysLockFromIsr();
	chBSemSignalI(&sema->sema);
	chSysUnlockFromIsr();

	return true;
}

#elif defined(PIOS_INCLUDE_IRQ)

/**
 *
 * @brief   Creates a binary semaphore.
 *
 * @returns instance of @p struct pios_semaphore or NULL on failure
 *
 */
struct pios_semaphore *PIOS_Semaphore_Create(void)
{
	struct pios_semaphore *sema = PIOS_malloc(sizeof(struct pios_semaphore));

	if (sema == NULL)
		return NULL;

	/*
	 * The initial state of a binary semaphore is "given".
	 */
	sema->sema_count = 1;

	return sema;
}

/**
 *
 * @brief   Takes binary semaphore.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 * @param[in] timeout_ms   timeout for acquiring the lock in milliseconds
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Take(struct pios_semaphore *sema, uint32_t timeout_ms)
{
	PIOS_Assert(sema != NULL);

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
}

/**
 *
 * @brief   Gives binary semaphore.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Give(struct pios_semaphore *sema)
{
	PIOS_Assert(sema != NULL);

	bool result = true;

	PIOS_IRQ_Disable();

	if (sema->sema_count == 0)
		++sema->sema_count;
	else
		result = false;

	PIOS_IRQ_Enable();

	return result;
}

/* Workaround for simulator version of FreeRTOS. */
#if !defined(SIM_POSIX)
/**
 *
 * @brief   Takes binary semaphore from ISR context.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 * @param[out] woken       pointer to bool which will be set true if a context switch is required
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Take_FromISR(struct pios_semaphore *sema, bool *woken)
{
	PIOS_Assert(sema != NULL);

	bool result = true;

	PIOS_IRQ_Disable();

	if (sema->sema_count != 0)
		--sema->sema_count;
	else
		result = false;

	PIOS_IRQ_Enable();

	return result;
}

/**
 *
 * @brief   Gives binary semaphore from ISR context.
 *
 * @param[in] sema         pointer to instance of @p struct pios_semaphore
 * @param[out] woken       pointer to bool which will be set true if a context switch is required
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Semaphore_Give_FromISR(struct pios_semaphore *sema, bool *woken)
{
	PIOS_Assert(sema != NULL);

	bool result = true;

	PIOS_IRQ_Disable();

	if (sema->sema_count == 0)
		++sema->sema_count;
	else
		result = false;

	PIOS_IRQ_Enable();

	return result;
}
#endif /* !defined(SIM_POSIX) */

#endif /* defined(PIOS_INCLUDE_IRQ) */
