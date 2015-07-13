/**
 ******************************************************************************
 * @file       pios_thread.c
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

#include "pios.h"
#include "pios_thread.h"

#if !defined(PIOS_INCLUDE_FREERTOS) && !defined(PIOS_INCLUDE_CHIBIOS)
#error "pios_thread.c requires PIOS_INCLUDE_FREERTOS or PIOS_INCLUDE_CHIBIOS"
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
 * @brief   Creates a thread.
 *
 * @param[in] fp           pointer to thread function
 * @param[in] namep        pointer to thread name
 * @param[in] stack_bytes  stack size in bytes
 * @param[in] argp         pointer to argument which will be passed to thread function
 * @param[in] prio         thread priority
 *
 * @returns instance of @p struct pios_thread or NULL on failure
 *
 */
struct pios_thread *PIOS_Thread_Create(void (*fp)(void *), const char *namep, size_t stack_bytes, void *argp, enum pios_thread_prio_e prio)
{
	struct pios_thread *thread = PIOS_malloc(sizeof(struct pios_thread));

	if (thread == NULL)
		return NULL;

	thread->task_handle = (uintptr_t)NULL;

	if (xTaskCreate(fp, (signed char*)namep, stack_bytes / 4, argp, prio, (xTaskHandle*)&thread->task_handle) != pdPASS)
	{
		PIOS_free(thread);
		return NULL;
	}

	return thread;
}

#if (INCLUDE_vTaskDelete == 1)
/**
 *
 * @brief   Destroys an instance of @p struct pios_thread.
 *
 * @param[in] threadp      pointer to instance of @p struct pios_thread
 *
 */
void PIOS_Thread_Delete(struct pios_thread *threadp)
{
	if (threadp == NULL)
		vTaskDelete(NULL);
	else
		vTaskDelete((xTaskHandle)threadp->task_handle);
}
#else
#error "PIOS_Thread_Delete requires INCLUDE_vTaskDelete to be defined 1"
#endif /* (INCLUDE_vTaskDelete == 1) */

/**
 *
 * @brief   Returns the current system time.
 *
 * @returns current system time
 *
 */
uint32_t PIOS_Thread_Systime(void)
{
	return (uint32_t)TICKS2MS(xTaskGetTickCount());
}

#if (INCLUDE_vTaskDelay == 1)
/**
 *
 * @brief   Suspends execution of current thread at least for specified time.
 *
 * @param[in] time_ms      time in milliseconds to suspend thread execution
 *
 */
void PIOS_Thread_Sleep(uint32_t time_ms)
{
	if (time_ms == PIOS_THREAD_TIMEOUT_MAX)
		vTaskDelay(portMAX_DELAY);
	else
		vTaskDelay((portTickType)MS2TICKS(time_ms));
}
#else
#error "PIOS_Thread_Sleep requires INCLUDE_vTaskDelay to be defined 1"
#endif /* (INCLUDE_vTaskDelay == 1) */

#if (INCLUDE_vTaskDelayUntil == 1)
/**
 *
 * @brief   Suspends execution of current thread for a regular interval.
 *
 * @param[in] previous_ms  pointer to system time of last execution,
 *                         must have been initialized with PIOS_Thread_Systime() on first invocation
 * @param[in] increment_ms time of regular interval in milliseconds
 *
 */
void PIOS_Thread_Sleep_Until(uint32_t *previous_ms, uint32_t increment_ms)
{
	portTickType temp = MS2TICKS(*previous_ms);
	vTaskDelayUntil(&temp, (portTickType)MS2TICKS(increment_ms));
	*previous_ms = TICKS2MS(temp);
}
#else
#error "PIOS_Thread_Sleep requires INCLUDE_vTaskDelayUntil to be defined 1"
#endif /* (INCLUDE_vTaskDelayUntil == 1) */

/**
 *
 * @brief   Returns stack usage of a thread.
 *
 * @param[in] threadp      pointer to instance of @p struct pios_thread
 *
 * @return stack usage in bytes
 *
 */
uint32_t PIOS_Thread_Get_Stack_Usage(struct pios_thread *threadp)
{
#if (INCLUDE_uxTaskGetStackHighWaterMark == 1)
	/* @note: This will fail when FreeRTOS TCB structure changes. */
	return uxTaskGetStackHighWaterMark((xTaskHandle)threadp->task_handle) * 4;
#else
	return 1024;
#endif /* (INCLUDE_uxTaskGetStackHighWaterMark == 1) */
}

/**
 *
 * @brief   Returns runtime of a thread.
 *
 * @param[in] threadp      pointer to instance of @p struct pios_thread
 *
 * @return runtime in milliseconds
 *
 */
uint32_t PIOS_Thread_Get_Runtime(struct pios_thread *threadp)
{
#if (INCLUDE_uxTaskGetRunTime == 1)
	return uxTaskGetRunTime((xTaskHandle)threadp->task_handle);
#else
	return 0;
#endif /* (INCLUDE_uxTaskGetRunTime == 1) */
}

/**
 *
 * @brief   Suspends execution of all threads.
 *
 */
void PIOS_Thread_Scheduler_Suspend(void)
{
	vTaskSuspendAll();
}

/**
 *
 * @brief   Resumes execution of all threads.
 *
 */
void PIOS_Thread_Scheduler_Resume(void)
{
	xTaskResumeAll();
}

#elif defined(PIOS_INCLUDE_CHIBIOS)

#define ST2MS(n) (((((n) - 1UL) * 1000UL) / CH_FREQUENCY) + 1UL)

/**
 * Compute size that is at rounded up to the nearest
 * multiple of 8
 */ 
static uint32_t ceil_size(uint32_t size)
{
	const uint32_t a = sizeof(stkalign_t);
	size = size + (a - size % a);
	return size;
}
/**
 * ChibiOS stack expects alignment (both start and end)
 * to 8 byte boundaries. This makes sure to allocate enough
 * memory and return an address that has the requested size
 * or more with these constraints.
 */
static uint8_t * align8_alloc(uint32_t size)
{
	// round size up to at nearest multiple of 8 + 4 bytes to guarantee
	// sufficient size within. This is because PIOS_malloc only guarantees
	// uintptr_t alignment which is 4 bytes.
	size = size + sizeof(uintptr_t);
	uint8_t *wap = PIOS_malloc(size);

	// shift start point to nearest 8 byte boundary.
	uint32_t pad = ((uint32_t) wap) % sizeof(stkalign_t);
	wap = wap + pad;

	return wap;
}

/**
 *
 * @brief   Creates a thread.
 *
 * @param[in] fp           pointer to thread function
 * @param[in] namep        pointer to thread name
 * @param[in] stack_bytes  stack size in bytes
 * @param[in] argp         pointer to argument which will be passed to thread function
 * @param[in] prio         thread priority
 *
 * @returns instance of @p struct pios_thread or NULL on failure
 *
 */
struct pios_thread *PIOS_Thread_Create(void (*fp)(void *), const char *namep, size_t stack_bytes, void *argp, enum pios_thread_prio_e prio)
{
	struct pios_thread *thread = PIOS_malloc(sizeof(struct pios_thread));
	if (thread == NULL)
		return NULL;

#ifdef SIM_POSIX
	if (stack_bytes < PIOS_THREAD_STACK_SIZE_MIN) {
		stack_bytes = PIOS_THREAD_STACK_SIZE_MIN;
	}
#endif

	// Use special functions to ensure ChibiOS stack requirements
	stack_bytes = ceil_size(stack_bytes);
	uint8_t *wap = align8_alloc(stack_bytes);
	if (wap == NULL)
	{
		PIOS_free(thread);
		return NULL;
	}

	thread->threadp = chThdCreateStatic(wap, stack_bytes, prio, (msg_t (*)(void *))fp, argp);
	if (thread->threadp == NULL)
	{
		PIOS_free(thread);
		PIOS_free(wap);
		return NULL;
	}

#if CH_USE_REGISTRY
	thread->threadp->p_name = namep;
#endif /* CH_USE_REGISTRY */

	return thread;
}

#if (CH_USE_WAITEXIT == TRUE)
/**
 *
 * @brief   Destroys an instance of @p struct pios_thread.
 *
 * @param[in] threadp      pointer to instance of @p struct pios_thread
 *
 */
void PIOS_Thread_Delete(struct pios_thread *threadp)
{
	if (threadp == NULL)
	{
		chThdExit(0);
	}
	else
	{
		chThdTerminate(threadp->threadp);
		chThdWait(threadp->threadp);
	}
}
#else
#error "PIOS_Thread_Delete requires CH_USE_WAITEXIT to be defined TRUE"
#endif /* (CH_USE_WAITEXIT == TRUE) */

/**
 *
 * @brief   Returns the current system time.
 *
 * @returns current system time
 *
 */
uint32_t PIOS_Thread_Systime(void)
{
	return (uint32_t)ST2MS(chTimeNow());
}

/**
 *
 * @brief   Suspends execution of current thread at least for specified time.
 *
 * @param[in] time_ms      time in milliseconds to suspend thread execution
 *
 */
void PIOS_Thread_Sleep(uint32_t time_ms)
{
	if (time_ms == PIOS_THREAD_TIMEOUT_MAX)
		chThdSleep(TIME_INFINITE);
	else
		chThdSleep(MS2ST(time_ms));
}

/**
 *
 * @brief   Suspends execution of current thread for a regular interval.
 *
 * @param[in] previous_ms  pointer to system time of last execution,
 *                         must have been initialized with PIOS_Thread_Systime() on first invocation
 * @param[in] increment_ms time of regular interval in milliseconds
 *
 */
void PIOS_Thread_Sleep_Until(uint32_t *previous_ms, uint32_t increment_ms)
{
	systime_t future = MS2ST(*previous_ms) + MS2ST(increment_ms);
	chSysLock();
	systime_t now = chTimeNow();
	int mustDelay =
		now < MS2ST(*previous_ms) ?
		(now < future && future < MS2ST(*previous_ms)) :
		(now < future || future < MS2ST(*previous_ms));
	if (mustDelay)
		chThdSleepS(future - now);
	chSysUnlock();
	*previous_ms = ST2MS(future);
}

/**
 *
 * @brief   Returns stack usage of a thread.
 *
 * @param[in] threadp      pointer to instance of @p struct pios_thread
 *
 * @return stack usage in bytes
 */
uint32_t PIOS_Thread_Get_Stack_Usage(struct pios_thread *threadp)
{
#if CH_DBG_FILL_THREADS
	uint32_t *stack = (uint32_t*)((size_t)threadp->threadp + sizeof(*threadp->threadp));
	uint32_t *stklimit = stack;
	while (*stack ==
			((CH_STACK_FILL_VALUE << 24) |
			(CH_STACK_FILL_VALUE << 16) |
			(CH_STACK_FILL_VALUE << 8) |
			(CH_STACK_FILL_VALUE << 0)))
		++stack;
	return (stack - stklimit) * 4;
#else
	return 0;
#endif /* CH_DBG_FILL_THREADS */
}

/**
 *
 * @brief   Returns runtime of a thread.
 *
 * @param[in] threadp      pointer to instance of @p struct pios_thread
 *
 * @return runtime in milliseconds
 *
 */
 uint32_t PIOS_Thread_Get_Runtime(struct pios_thread *threadp)
{
	chSysLock();

	uint32_t result = threadp->threadp->ticks_total;
	threadp->threadp->ticks_total = 0;

	chSysUnlock();

	return result;
}

/**
 *
 * @brief   Suspends execution of all threads.
 *
 */
void PIOS_Thread_Scheduler_Suspend(void)
{
	chSysLock();
}

/**
 *
 * @brief   Resumes execution of all threads.
 *
 */
void PIOS_Thread_Scheduler_Resume(void)
{
	chSysUnlock();
}

#endif /* defined(PIOS_INCLUDE_CHIBIOS) */
