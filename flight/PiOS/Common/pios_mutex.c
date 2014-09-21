/**
 ******************************************************************************
 * @file       pios_mutex.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_Mutex Mutex Abstraction
 * @{
 * @brief Abstracts the concept of a mutex to hide different implementations
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
#include "pios_mutex.h"

#if !defined(PIOS_INCLUDE_FREERTOS)
#error "pios_mutex.c requires PIOS_INCLUDE_FREERTOS"
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
 * @brief   Creates a non recursive mutex.
 *
 * @returns instance of @p struct pios_mutex or NULL on failure
 *
 */
struct pios_mutex *PIOS_Mutex_Create(void)
{
	struct pios_mutex *mtx = PIOS_malloc(sizeof(struct pios_mutex));

	if (mtx == NULL)
		return NULL;

	mtx->mtx_handle = (uintptr_t)xSemaphoreCreateMutex();

	return mtx;
}

/**
 *
 * @brief   Locks a non recursive mutex.
 *
 * @param[in] mtx          pointer to instance of @p struct pios_mutex
 * @param[in] timeout_ms   timeout for acquiring the lock in milliseconds
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Mutex_Lock(struct pios_mutex *mtx, uint32_t timeout_ms)
{
	PIOS_Assert(mtx != NULL);

	portTickType timeout_ticks;
	if (timeout_ms == PIOS_MUTEX_TIMEOUT_MAX)
		timeout_ticks = portMAX_DELAY;
	else
		timeout_ticks = MS2TICKS(timeout_ms);

	return xSemaphoreTake(mtx->mtx_handle, timeout_ticks) == pdTRUE;
}

/**
 *
 * @brief   Unlocks a non recursive mutex.
 *
 * @param[in] mtx          pointer to instance of @p struct pios_mutex
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Mutex_Unlock(struct pios_mutex *mtx)
{
	PIOS_Assert(mtx != NULL);

	return xSemaphoreGive(mtx->mtx_handle) == pdTRUE;
}

/**
 *
 * @brief   Creates a recursive mutex.
 *
 * @returns instance of @p struct pios_recursive mutex or NULL on failure
 *
 */
struct pios_recursive_mutex *PIOS_Recursive_Mutex_Create(void)
{
	struct pios_recursive_mutex *mtx = PIOS_malloc(sizeof(struct pios_recursive_mutex));

	if (mtx == NULL)
		return NULL;

	mtx->mtx_handle = (uintptr_t)xSemaphoreCreateRecursiveMutex();

	return mtx;
}

/**
 *
 * @brief   Locks a recursive mutex.
 *
 * @param[in] mtx          pointer to instance of @p struct pios_recursive_mutex
 * @param[in] timeout_ms   timeout for acquiring the lock in milliseconds
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Recursive_Mutex_Lock(struct pios_recursive_mutex *mtx, uint32_t timeout_ms)
{
	PIOS_Assert(mtx != NULL);

	portTickType timeout_ticks;
	if (timeout_ms == PIOS_MUTEX_TIMEOUT_MAX)
		timeout_ticks = portMAX_DELAY;
	else
		timeout_ticks = MS2TICKS(timeout_ms);

	return xSemaphoreTakeRecursive((xSemaphoreHandle)mtx->mtx_handle, timeout_ticks) == pdTRUE;
}

/**
 *
 * @brief   Unlocks a recursive mutex.
 *
 * @param[in] mtx          pointer to instance of @p struct pios_recursive_mutex
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Recursive_Mutex_Unlock(struct pios_recursive_mutex *mtx)
{
	PIOS_Assert(mtx != NULL);

	return xSemaphoreGiveRecursive((xSemaphoreHandle)mtx->mtx_handle) == pdTRUE;
}

#endif
