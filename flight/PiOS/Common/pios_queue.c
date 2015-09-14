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

#if !defined(PIOS_INCLUDE_FREERTOS) && !defined(PIOS_INCLUDE_CHIBIOS)
#error "pios_queue.c requires PIOS_INCLUDE_FREERTOS or PIOS_INCLUDE_CHIBIOS"
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
	struct pios_queue *queuep = PIOS_malloc_no_dma(sizeof(struct pios_queue));

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

#elif defined(PIOS_INCLUDE_CHIBIOS)

#if !defined(PIOS_QUEUE_MAX_WAITERS)
#define PIOS_QUEUE_MAX_WAITERS 2
#endif /* !defined(PIOS_QUEUE_MAX_WAITERS) */

/**
 *
 * @brief   Creates a queue.
 *
 * @returns instance of @p struct pios_queue or NULL on failure
 *
 */
struct pios_queue *PIOS_Queue_Create(size_t queue_length, size_t item_size)
{
	struct pios_queue *queuep = PIOS_malloc_no_dma(sizeof(struct pios_queue));
	if (queuep == NULL)
		return NULL;

	/* Create the memory pool. */
	queuep->mpb = PIOS_malloc_no_dma(item_size * (queue_length + PIOS_QUEUE_MAX_WAITERS));
	if (queuep->mpb == NULL) {
		PIOS_free(queuep);
		return NULL;
	}
	chPoolInit(&queuep->mp, item_size, NULL);
	chPoolLoadArray(&queuep->mp, queuep->mpb, queue_length + PIOS_QUEUE_MAX_WAITERS);

	/* Create the mailbox. */
	msg_t *mb_buf = PIOS_malloc_no_dma(sizeof(msg_t) * queue_length);
	chMBInit(&queuep->mb, mb_buf, queue_length);

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
	PIOS_free(queuep->mpb);
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
	void *buf = chPoolAlloc(&queuep->mp);
	if (buf == NULL)
		return false;

	memcpy(buf, itemp, queuep->mp.mp_object_size);

	systime_t timeout;
	if (timeout_ms == PIOS_QUEUE_TIMEOUT_MAX)
		timeout = TIME_INFINITE;
	else if (timeout_ms == 0)
		timeout = TIME_IMMEDIATE;
	else
		timeout = MS2ST(timeout_ms);

	msg_t result = chMBPost(&queuep->mb, (msg_t)buf, timeout);

	if (result != RDY_OK)
	{
		chPoolFree(&queuep->mp, buf);
		return false;
	}

	return true;
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
	chSysLockFromIsr();
	void *buf = chPoolAllocI(&queuep->mp);
	if (buf == NULL)
	{
		chSysUnlockFromIsr();
		return false;
	}

	memcpy(buf, itemp, queuep->mp.mp_object_size);

	msg_t result = chMBPostI(&queuep->mb, (msg_t)buf);

	if (result != RDY_OK)
	{
		chPoolFreeI(&queuep->mp, buf);
		chSysUnlockFromIsr();
		return false;
	}

	chSysUnlockFromIsr();

	return true;
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
	msg_t buf;

	systime_t timeout;
	if (timeout_ms == PIOS_QUEUE_TIMEOUT_MAX)
		timeout = TIME_INFINITE;
	else if (timeout_ms == 0)
		timeout = TIME_IMMEDIATE;
	else
		timeout = MS2ST(timeout_ms);

	msg_t result = chMBFetch(&queuep->mb, &buf, timeout);

	if (result != RDY_OK)
		return false;

	memcpy(itemp, (void*)buf, queuep->mp.mp_object_size);

	chPoolFree(&queuep->mp, (void*)buf);

	return true;
}

#endif /* defined(PIOS_INCLUDE_CHIBIOS) */
