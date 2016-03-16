/**
 ******************************************************************************
 * @addtogroup TauLabsCore Tau Labs Core components
 * @{
 * @addtogroup UAVObjectHandling UAVObject handling code
 * @{
 *
 * @file       eventdispatcher.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      Event dispatcher, distributes object events as callbacks. Alternative
 * 	           to using tasks and queues. All callbacks are invoked from the event task.
 * @see        The GNU Public License (GPL) Version 3
 *
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

#include "openpilot.h"
#include "pios_mutex.h"
#include "pios_thread.h"
#include "pios_queue.h"
#include "misc_math.h"

// Private constants
#if defined(PIOS_EVENTDISAPTCHER_QUEUE)
#define MAX_QUEUE_SIZE PIOS_EVENTDISAPTCHER_QUEUE
#else
#define MAX_QUEUE_SIZE 20
#endif

#if defined(PIOS_EVENTDISPATCHER_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_EVENTDISPATCHER_STACK_SIZE
#else
#define STACK_SIZE_BYTES PIOS_THREAD_STACK_SIZE_MIN
#endif /* PIOS_EVENTDISPATCHER_STACK_SIZE */

#define TASK_PRIORITY PIOS_THREAD_PRIO_HIGH
#define MAX_UPDATE_PERIOD_MS 1000

// Private types

/**
 * Event callback information
 */
typedef struct {
	UAVObjEvent ev; /** The actual event */
	UAVObjEventCallback cb; /** The callback function, or zero if none */
	struct pios_queue *queue; /** The queue or zero if none */
} EventCallbackInfo;

/**
 * List of object properties that are needed for the periodic updates.
 */
struct PeriodicObjectListStruct {
	EventCallbackInfo evInfo; /** Event callback information */
    uint16_t updatePeriodMs; /** Update period in ms or 0 if no periodic updates are needed */
    int32_t timeToNextUpdateMs; /** Time delay to the next update */
    struct PeriodicObjectListStruct* next; /** Needed by linked list library (utlist.h) */
};
typedef struct PeriodicObjectListStruct PeriodicObjectList;

// Private variables
static PeriodicObjectList* objList;
static struct pios_queue *queue;
static struct pios_thread *eventTaskHandle;
static struct pios_recursive_mutex *mutex;
static EventStats stats;

// Private functions
static int32_t processPeriodicUpdates();
static void eventTask();
static int32_t eventPeriodicCreate(UAVObjEvent* ev, UAVObjEventCallback cb, struct pios_queue *queue, uint16_t periodMs);
static int32_t eventPeriodicUpdate(UAVObjEvent* ev, UAVObjEventCallback cb, struct pios_queue *queue, uint16_t periodMs);

/**
 * Initialize the dispatcher
 * \return Success (0), failure (-1)
 */
int32_t EventDispatcherInitialize()
{
	// Initialize variables
	objList = NULL;
	memset(&stats, 0, sizeof(EventStats));

	// Create mutex
	mutex = PIOS_Recursive_Mutex_Create();
	if (mutex == NULL)
		return -1;

	// Create event queue
	queue = PIOS_Queue_Create(MAX_QUEUE_SIZE, sizeof(EventCallbackInfo));

	// Create task
	eventTaskHandle = PIOS_Thread_Create(eventTask, "event", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);

	// Done
	return 0;
}

/**
 * Get the statistics counters
 * @param[out] statsOut The statistics counters will be copied there
 */
void EventGetStats(EventStats* statsOut)
{
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	memcpy(statsOut, &stats, sizeof(EventStats));
	PIOS_Recursive_Mutex_Unlock(mutex);
}

/**
 * Clear the statistics counters
 */
void EventClearStats()
{
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	memset(&stats, 0, sizeof(EventStats));
	PIOS_Recursive_Mutex_Unlock(mutex);
}

/**
 * Dispatch an event at periodic intervals.
 * \param[in] ev The event to be dispatched
 * \param[in] cb The callback to be invoked
 * \param[in] periodMs The period the event is generated
 * \return Success (0), failure (-1)
 */
int32_t EventPeriodicCallbackCreate(UAVObjEvent* ev, UAVObjEventCallback cb, uint16_t periodMs)
{
	return eventPeriodicCreate(ev, cb, 0, periodMs);
}

/**
 * Update the period of a periodic event.
 * \param[in] ev The event to be dispatched
 * \param[in] cb The callback to be invoked
 * \param[in] periodMs The period the event is generated
 * \return Success (0), failure (-1)
 */
int32_t EventPeriodicCallbackUpdate(UAVObjEvent* ev, UAVObjEventCallback cb, uint16_t periodMs)
{
	return eventPeriodicUpdate(ev, cb, 0, periodMs);
}

/**
 * Dispatch an event at periodic intervals.
 * \param[in] ev The event to be dispatched
 * \param[in] queue The queue that the event will be pushed in
 * \param[in] periodMs The period the event is generated
 * \return Success (0), failure (-1)
 */
int32_t EventPeriodicQueueCreate(UAVObjEvent* ev, struct pios_queue *queue, uint16_t periodMs)
{
	return eventPeriodicCreate(ev, 0, queue, periodMs);
}

/**
 * Update the period of a periodic event.
 * \param[in] ev The event to be dispatched
 * \param[in] queue The queue
 * \param[in] periodMs The period the event is generated
 * \return Success (0), failure (-1)
 */
int32_t EventPeriodicQueueUpdate(UAVObjEvent* ev, struct pios_queue *queue, uint16_t periodMs)
{
	return eventPeriodicUpdate(ev, 0, queue, periodMs);
}

/**
 * Dispatch an event through a callback at periodic intervals.
 * \param[in] ev The event to be dispatched
 * \param[in] cb The callback to be invoked or zero if none
 * \param[in] queue The queue or zero if none
 * \param[in] periodMs The period the event is generated
 * \return Success (0), failure (-1)
 */
static int32_t eventPeriodicCreate(UAVObjEvent* ev, UAVObjEventCallback cb, struct pios_queue *queue, uint16_t periodMs)
{
	PeriodicObjectList* objEntry;
	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	// Check that the object is not already connected
	LL_FOREACH(objList, objEntry)
	{
		if (objEntry->evInfo.cb == cb &&
			objEntry->evInfo.queue == queue &&
			objEntry->evInfo.ev.obj == ev->obj &&
			objEntry->evInfo.ev.instId == ev->instId &&
			objEntry->evInfo.ev.event == ev->event)
		{
			// Already registered, do nothing
			PIOS_Recursive_Mutex_Unlock(mutex);
			return -1;
		}
	}
    // Create handle
	objEntry = (PeriodicObjectList*)PIOS_malloc_no_dma(sizeof(PeriodicObjectList));
	if (objEntry == NULL) return -1;
	objEntry->evInfo.ev.obj = ev->obj;
	objEntry->evInfo.ev.instId = ev->instId;
	objEntry->evInfo.ev.event = ev->event;
	objEntry->evInfo.cb = cb;
	objEntry->evInfo.queue = queue;
    objEntry->updatePeriodMs = periodMs;
    objEntry->timeToNextUpdateMs = randomize_int(periodMs); // avoid bunching of updates
    // Add to list
    LL_APPEND(objList, objEntry);
	// Release lock
	PIOS_Recursive_Mutex_Unlock(mutex);
    return 0;
}

/**
 * Update the period of a periodic event.
 * \param[in] ev The event to be dispatched
 * \param[in] cb The callback to be invoked or zero if none
 * \param[in] queue The queue or zero if none
 * \param[in] periodMs The period the event is generated
 * \return Success (0), failure (-1)
 */
static int32_t eventPeriodicUpdate(UAVObjEvent* ev, UAVObjEventCallback cb, struct pios_queue *queue, uint16_t periodMs)
{
	PeriodicObjectList* objEntry;
	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	// Find object
	LL_FOREACH(objList, objEntry)
	{
		if (objEntry->evInfo.cb == cb &&
			objEntry->evInfo.queue == queue &&
			objEntry->evInfo.ev.obj == ev->obj &&
			objEntry->evInfo.ev.instId == ev->instId &&
			objEntry->evInfo.ev.event == ev->event)
		{
			// Object found, update period
			objEntry->updatePeriodMs = periodMs;
			objEntry->timeToNextUpdateMs = randomize_int(periodMs); // avoid bunching of updates
			// Release lock
			PIOS_Recursive_Mutex_Unlock(mutex);
			return 0;
		}
	}
    // If this point is reached the object was not found
	PIOS_Recursive_Mutex_Unlock(mutex);
    return -1;
}

/**
 * Event task, responsible of invoking callbacks.
 */
static void eventTask()
{
	int32_t timeToNextUpdateMs;
	int32_t delayMs;

	/* Must do this in task context to ensure that TaskMonitor has already finished its init */
	TaskMonitorAdd(TASKINFO_RUNNING_EVENTDISPATCHER, eventTaskHandle);

	// Initialize time
	timeToNextUpdateMs = PIOS_Thread_Systime();

	// Loop forever
	while (1)
	{
		// Calculate delay time
		delayMs = timeToNextUpdateMs - PIOS_Thread_Systime();
		if (delayMs > 0) {
			PIOS_Thread_Sleep(delayMs);
		}

		timeToNextUpdateMs = processPeriodicUpdates();
	}
}

/**
 * Handle periodic updates for all objects.
 * \return The system time until the next update (in ms) or -1 if failed
 */
static int32_t processPeriodicUpdates()
{
	PeriodicObjectList* objEntry;
	int32_t timeNow;
    int32_t timeToNextUpdate;
    int32_t offset;

	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

    // Iterate through each object and update its timer, if zero then transmit object.
    // Also calculate smallest delay to next update.
    timeToNextUpdate = PIOS_Thread_Systime() + MAX_UPDATE_PERIOD_MS;
    LL_FOREACH(objList, objEntry)
    {
        // If object is configured for periodic updates
        if (objEntry->updatePeriodMs > 0)
        {
            // Check if time for the next update
        	timeNow = PIOS_Thread_Systime();
            if (objEntry->timeToNextUpdateMs <= timeNow)
            {
                // Reset timer
            	offset = ( timeNow - objEntry->timeToNextUpdateMs ) % objEntry->updatePeriodMs;
            	objEntry->timeToNextUpdateMs = timeNow + objEntry->updatePeriodMs - offset;
    			// Invoke callback, if one
    			if ( objEntry->evInfo.cb != 0)
    			{
				objEntry->evInfo.cb(&objEntry->evInfo.ev, NULL, NULL, 0); // the function is expected to copy the event information
    			}
    			// Push event to queue, if one
    			if ( objEntry->evInfo.queue != 0)
    			{
    				if (PIOS_Queue_Send(objEntry->evInfo.queue, &objEntry->evInfo.ev, 0) != true ) // do not block if queue is full
    				{
						if (objEntry->evInfo.ev.obj != NULL)
							stats.lastErrorID = UAVObjGetID(objEntry->evInfo.ev.obj);
    					++stats.eventErrors;
    				}
    			}
            }
            // Update minimum delay
            if (objEntry->timeToNextUpdateMs < timeToNextUpdate)
            {
            	timeToNextUpdate = objEntry->timeToNextUpdateMs;
            }
        }
    }

    // Done
    PIOS_Recursive_Mutex_Unlock(mutex);
    return timeToNextUpdate;
}

/**
 * @}
 * @}
 */
