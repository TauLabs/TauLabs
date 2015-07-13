/**
 ******************************************************************************
 * @file       pios_heap.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
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

void * PIOS_malloc(size_t size)
{
#if defined(PIOS_INCLUDE_FREERTOS)
	void *buf = pvPortMalloc(size);
#elif defined(PIOS_INCLUDE_CHIBIOS)
	void *buf = chHeapAlloc(NULL, size);
#else
#error "pios_heap requires either PIOS_INCLUDE_FREERTOS or PIOS_INCLUDE_CHIBIOS"
#endif

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
#if defined(PIOS_INCLUDE_FREERTOS)
	vPortFree(buf);
#elif defined(PIOS_INCLUDE_CHIBIOS)
	chHeapFree(buf);
#else
#error "pios_heap requires either PIOS_INCLUDE_FREERTOS or PIOS_INCLUDE_CHIBIOS"
#endif
}

void PIOS_heap_initialize_blocks(void)
{
}

size_t PIOS_heap_get_free_size(void)
{
	return 1024;
}

/**
 * @}
 * @}
 */
