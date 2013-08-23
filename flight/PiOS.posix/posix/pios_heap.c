/**
 ******************************************************************************
 * @file       pios_heap.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_HEAP Heap Allocation Abstraction
 * @{
 * @brief Heap allocation abstraction to hide details of allocation from SRAM and CCM RAM
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

/* Project Includes */
#include "pios.h"		/* PIOS_INCLUDE_* */

#include "pios_heap.h"		/* External API declaration */
#include <stdbool.h>		/* bool */

#define DEBUG_MALLOC_FAILURES 0
static volatile bool malloc_failed_flag = false;
static void malloc_failed_hook(void)
{
	malloc_failed_flag = true;
#if DEBUG_MALLOC_FAILURES
	static volatile bool wait_here = true;
	while(wait_here);
	wait_here = true;
#endif
}

bool PIOS_heap_malloc_failed_p(void)
{
	return malloc_failed_flag;
}

#if defined(PIOS_INCLUDE_FREERTOS)

/*
 * Defining MPU_WRAPPERS_INCLUDED_FROM_API_FILE prevents task.h from redefining
 * all the API functions to use the MPU wrappers.  That should only be done when
 * task.h is included from an application file.
 * */
#define MPU_WRAPPERS_INCLUDED_FROM_API_FILE
#include "FreeRTOS.h"		/* needed by task.h */
#include "task.h"		/* vTaskSuspendAll, xTaskResumeAll */
#undef MPU_WRAPPERS_INCLUDED_FROM_API_FILE

#endif	/* PIOS_INCLUDE_FREERTOS */

void * PIOS_malloc(size_t size)
{
	void *buf = pvPortMalloc(size);

	if (buf == NULL)
		malloc_failed_hook();

	return buf;
}

void * PIOS_malloc_no_dma(size_t size)
{
	return PIOS_malloc(size);
}

void PIOS_free(void * buf)
{
	vPortFree(buf);
}

/**
 * @}
 * @}
 */
