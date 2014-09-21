/**
 ******************************************************************************
 * @file       pios_queue.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_Queue Queue Abstraction
 * @{
 * @brief Abstracts the concept of a queue to hide different implementations
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
#include "pios_queue.h"

#if !defined(PIOS_INCLUDE_FREERTOS)
#error "pios_queue.c requires PIOS_INCLUDE_FREERTOS"
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
 * @brief   Creates a queue.
 *
 * @returns instance of @p struct pios_queue or NULL on failure
 *
 */
struct pios_queue *PIOS_Queue_Create(size_t queue_length, size_t item_size)
{
	struct pios_queue *queuep = PIOS_malloc(sizeof(struct pios_queue));

	if (queuep == NULL)
		return NULL;

	queuep->queue_handle = (uintptr_t)NULL;

	if ((queuep->queue_handle = (uintptr_t)xQueueCreate(queue_length, item_size)) == (uintptr_t)NULL)
	{
		PIOS_free(queuep);
		return NULL;
	}

	return queuep;
}

/**
 *
 * @brief   Destroys an instance of @p struct pios_queue
 *
 * @param[in] queuep       pointer to instance of @p struct pios_queue
 *
 */
void PIOS_Queue_Delete(struct pios_queue *queuep)
{
	vQueueDelete((xQueueHandle)queuep->queue_handle);
	PIOS_free(queuep);
}

/**
 *
 * @brief   Appends an item to a queue.
 *
 * @param[in] queuep       pointer to instance of @p struct pios_queue
 * @param[in] itemp        pointer to item which will be appended to the queue
 * @param[in] timeout_ms   timeout for appending item to queue in milliseconds
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Queue_Send(struct pios_queue *queuep, const void *itemp, uint32_t timeout_ms)
{
	return xQueueSendToBack((xQueueHandle)queuep->queue_handle, itemp, MS2TICKS(timeout_ms)) == pdTRUE;
}

/**
 *
 * @brief   Appends an item to a queue from ISR context.
 *
 * @param[in] queuep       pointer to instance of @p struct pios_queue
 * @param[in] itemp        pointer to item which will be appended to the queue
 * @param[in] timeout_ms   timeout for appending item to queue in milliseconds
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Queue_Send_FromISR(struct pios_queue *queuep, const void *itemp, bool *wokenp)
{
	portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;
	portBASE_TYPE result = xQueueSendToBackFromISR((xQueueHandle)queuep->queue_handle, itemp, &xHigherPriorityTaskWoken);
	*wokenp = *wokenp || xHigherPriorityTaskWoken == pdTRUE;
	return result == pdTRUE;
}

/**
 *
 * @brief   Retrieves an item from the front of a queue.
 *
 * @param[in] queuep       pointer to instance of @p struct pios_queue
 * @param[in] itemp        pointer to item which will be retrieved
 * @param[in] timeout_ms   timeout for retrieving item from queue in milliseconds
 *
 * @returns true on success or false on timeout or failure
 *
 */
bool PIOS_Queue_Receive(struct pios_queue *queuep, void *itemp, uint32_t timeout_ms)
{
	return xQueueReceive((xQueueHandle)queuep->queue_handle, itemp, MS2TICKS(timeout_ms)) == pdTRUE;
}

#endif /* defined(PIOS_INCLUDE_FREERTOS) */
