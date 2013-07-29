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

#include <stdio.h>		/* NULL */
#include <stdint.h>		/* uintptr_t */
#include <stdbool.h>		/* bool */

#if defined(PIOS_INCLUDE_FREERTOS)
#include "FreeRTOS.h"		/* needed by task.h */
#include "task.h"		/* vTaskSuspendAll, xTaskResumeAll */
#endif	/* PIOS_INCLUDE_FREERTOS */

struct pios_heap {
	const uintptr_t start_addr;
	const uintptr_t end_addr;
	uintptr_t free_addr;
};

#if defined(PIOS_INCLUDE_FASTHEAP) || !defined(PIOS_INCLUDE_FREERTOS)

static bool is_ptr_in_heap_p(const struct pios_heap * heap, void * buf)
{
	uintptr_t buf_addr = (uintptr_t)buf;

	return ((buf_addr >= heap->start_addr) && (buf_addr <= heap->end_addr));
}

static void * simple_malloc(struct pios_heap *heap, size_t size)
{
	if (heap == NULL)
		return NULL;

	void * buf = NULL;
	uint32_t align_pad = (sizeof(uintptr_t) - (size & (sizeof(uintptr_t) - 1))) % sizeof(uintptr_t);

#if defined(PIOS_INCLUDE_FREERTOS)
	vTaskSuspendAll();
#endif	/* PIOS_INCLUDE_FREERTOS */

	if (heap->free_addr + size <= heap->end_addr) {
		buf = (void *)heap->free_addr;
		heap->free_addr += size + align_pad;
	}

#if defined(PIOS_INCLUDE_FREERTOS)
	xTaskResumeAll();
#endif	/* PIOS_INCLUDE_FREERTOS */

	return buf;
}

static void simple_free(struct pios_heap *heap, void *buf)
{
	/* This allocator doesn't support free */
}

#endif	/* PIOS_INCLUDE_FASTHEAP || !PIOS_INCLUDE_FREERTOS */

/*
 * Standard heap.  All memory in this heap is DMA-safe.
 * Note: Uses underlying FreeRTOS heap when available
 */
#if defined(PIOS_INCLUDE_FREERTOS)

void * PIOS_malloc(size_t size)
{
	return pvPortMalloc(size);
}

#else  /* PIOS_INCLUDE_FREERTOS */

extern const void * _eheap;	/* defined in linker script */
extern const void * _sheap;	/* defined in linker script */

static struct pios_heap pios_slow_heap = {
	.start_addr = (const uintptr_t)&_sheap,
	.end_addr   = (const uintptr_t)&_eheap,
	.free_addr  = (uintptr_t)&_sheap,
};
void * PIOS_malloc(size_t size)
{
	return simple_malloc(&pios_slow_heap, size);
}

#endif	/* PIOS_INCLUDE_FREERTOS */

/*
 * Fast heap.  Memory in this heap is NOT DMA-safe.
 * Note: This should not be used to allocate RAM for task stacks since a task may pass
 *       automatic variables into underlying PIOS functions which *may* use DMA.
 * Note: Falls back to using the standard heap when allocations cannot be satisfied.
 */
#if defined(PIOS_INCLUDE_FASTHEAP)

extern const void * _efastheap;	/* defined in linker script */
extern const void * _sfastheap;	/* defined in linker script */
static struct pios_heap pios_fast_heap = {
	.start_addr = (const uintptr_t)&_sfastheap,
	.end_addr   = (const uintptr_t)&_efastheap,
	.free_addr  = (uintptr_t)&_sfastheap,
};
void * PIOS_malloc_no_dma(size_t size)
{
	void * buf = simple_malloc(&pios_fast_heap, size);

	if (buf == NULL)
		return PIOS_malloc(size);

	return buf;
}

#else	/* PIOS_INCLUDE_FASTHEAP */

/* This platform only has a standard heap.  Fall back directly to that */
void * PIOS_malloc_no_dma(size_t size)
{
	return PIOS_malloc(size);
}

#endif	/* PIOS_INCLUDE_FASTHEAP */

void PIOS_free(void * buf)
{
#if defined(PIOS_INCLUDE_FASTHEAP)
	if (is_ptr_in_heap_p(&pios_fast_heap, buf))
		return simple_free(&pios_fast_heap, buf);
#endif	/* PIOS_INCLUDE_FASTHEAP */

#if !defined(PIOS_INCLUDE_FREERTOS)
	if (is_ptr_in_heap_p(&pios_slow_heap, buf))
		return simple_free(&pios_slow_heap, buf);
#else  /* PIOS_INCLUDE_FREERTOS */
	return vPortFree(buf);
#endif	/* PIOS_INCLUDE_FREERTOS */
}

/**
 * @}
 * @}
 */
