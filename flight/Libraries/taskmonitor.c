/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 *
 * @file       taskmonitor.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      Task monitoring library
 * @see        The GNU Public License (GPL) Version 3
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
//#include "taskmonitor.h"
#include "pios_mutex.h"

// Private constants

// Private types

// Private variables
static struct pios_mutex *lock;
static struct pios_thread *handles[TASKINFO_RUNNING_NUMELEM];
static uint32_t lastMonitorTime;

// Private functions

/**
 * Initialize library
 */
int32_t TaskMonitorInitialize(void)
{
	lock = PIOS_Mutex_Create();
	PIOS_Assert(lock != NULL);
	memset(handles, 0, sizeof(struct pios_thread) * TASKINFO_RUNNING_NUMELEM);
	lastMonitorTime = 0;
#if defined(DIAG_TASKS)
#if defined(PIOS_INCLUDE_FREERTOS)
	lastMonitorTime = portGET_RUN_TIME_COUNTER_VALUE();
#elif defined(PIOS_INCLUDE_CHIBIOS)
	lastMonitorTime = halGetCounterValue();
#endif /* defined(PIOS_INCLUDE_CHIBIOS) */
#endif
	return 0;
}

/**
 * Register a task handle with the library
 */
int32_t TaskMonitorAdd(TaskInfoRunningElem task, struct pios_thread *threadp)
{
	uint32_t task_idx = (uint32_t) task;
	if (task_idx < TASKINFO_RUNNING_NUMELEM)
	{
		PIOS_Mutex_Lock(lock, PIOS_MUTEX_TIMEOUT_MAX);
		handles[task_idx] = threadp;
		PIOS_Mutex_Unlock(lock);
		return 0;
	}
	else
	{
		return -1;
	}
}

/**
 * Remove a task handle from the library
 */
int32_t TaskMonitorRemove(TaskInfoRunningElem task)
{
	uint32_t task_idx = (uint32_t) task;
	if (task_idx < TASKINFO_RUNNING_NUMELEM)
	{
		PIOS_Mutex_Lock(lock, PIOS_MUTEX_TIMEOUT_MAX);
		handles[task_idx] = 0;
		PIOS_Mutex_Unlock(lock);
		return 0;
	}
	else
	{
		return -1;
	}
}

/**
 * Query if a task is running
 */
bool TaskMonitorQueryRunning(TaskInfoRunningElem task)
{
	uint32_t task_idx = (uint32_t) task;
	if (task_idx < TASKINFO_RUNNING_NUMELEM && handles[task_idx] != 0)
		return true;
	return false;
}

/**
 * Update the status of all tasks
 */
void TaskMonitorUpdateAll(void)
{
#if defined(DIAG_TASKS)
	TaskInfoData data;
	int n;

	// Lock
	PIOS_Mutex_Lock(lock, PIOS_MUTEX_TIMEOUT_MAX);

	uint32_t currentTime;
	uint32_t deltaTime;
	
	/*
	 * Calculate the amount of elapsed run time between the last time we
	 * measured and now. Scale so that we can convert task run times
	 * directly to percentages.
	 */
#if defined(PIOS_INCLUDE_FREERTOS)
	currentTime = portGET_RUN_TIME_COUNTER_VALUE();
#elif defined(PIOS_INCLUDE_CHIBIOS)
	currentTime = hal_lld_get_counter_value();
#endif /* defined(PIOS_INCLUDE_CHIBIOS) */
	deltaTime = ((currentTime - lastMonitorTime) / 100) ? : 1; /* avoid divide-by-zero if the interval is too small */
	lastMonitorTime = currentTime;
	
	// Update all task information
	for (n = 0; n < TASKINFO_RUNNING_NUMELEM; ++n)
	{
		if (handles[n] != 0)
		{
			data.Running[n] = TASKINFO_RUNNING_TRUE;
			data.StackRemaining[n] = PIOS_Thread_Get_Stack_Usage(handles[n]);
			/* Generate run time stats */
			data.RunningTime[n] = PIOS_Thread_Get_Runtime(handles[n]) / deltaTime;
		}
		else
		{
			data.Running[n] = TASKINFO_RUNNING_FALSE;
			data.StackRemaining[n] = 0;
			data.RunningTime[n] = 0;
		}
	}

	// Update object
	TaskInfoSet(&data);

	// Done
	PIOS_Mutex_Unlock(lock);
#endif
}

/**
 * @}
 */
