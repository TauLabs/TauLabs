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

struct pios_heap {
	const uintptr_t start_addr;
	uintptr_t end_addr;
	uintptr_t free_addr;
};

static bool is_ptr_in_heap_p(const struct pios_heap *heap, void *buf)
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

static size_t simple_get_free_bytes(struct pios_heap *heap)
{
	if (heap->free_addr > heap->end_addr)
		return 0;

	return heap->end_addr - heap->free_addr;
}

static void simple_extend_heap(struct pios_heap *heap, size_t bytes)
{
	heap->end_addr += bytes;
}

/*
 * Standard heap.  All memory in this heap is DMA-safe.
 * Note: Uses underlying FreeRTOS heap when available
 */
extern const void * _eheap;	/* defined in linker script */
extern const void * _sheap;	/* defined in linker script */

static struct pios_heap pios_standard_heap = {
	.start_addr = (const uintptr_t)&_sheap,
	.end_addr   = (const uintptr_t)&_eheap,
	.free_addr  = (uintptr_t)&_sheap,
};


void * pvPortMalloc(size_t size) __attribute__((alias ("PIOS_malloc"), weak));
void * PIOS_malloc(size_t size)
{
	void *buf = simple_malloc(&pios_standard_heap, size);

	if (buf == NULL)
		malloc_failed_hook();

	return buf;
}

/*
 * Fast heap.  Memory in this heap is NOT DMA-safe.
 * Note: This should not be used to allocate RAM for task stacks since a task may pass
 *       automatic variables into underlying PIOS functions which *may* use DMA.
 * Note: Falls back to using the standard heap when allocations cannot be satisfied.
 */
#if defined(PIOS_INCLUDE_FASTHEAP)

extern const void * _efastheap;	/* defined in linker script */
extern const void * _sfastheap;	/* defined in linker script */
static struct pios_heap pios_nodma_heap = {
	.start_addr = (const uintptr_t)&_sfastheap,
	.end_addr   = (const uintptr_t)&_efastheap,
	.free_addr  = (uintptr_t)&_sfastheap,
};
void * PIOS_malloc_no_dma(size_t size)
{
	void * buf = simple_malloc(&pios_nodma_heap, size);

	if (buf == NULL)
		buf = PIOS_malloc(size);

	if (buf == NULL)
		malloc_failed_hook();

	return buf;
}

#else	/* PIOS_INCLUDE_FASTHEAP */

/* This platform only has a standard heap.  Fall back directly to that */
void * PIOS_malloc_no_dma(size_t size)
{
	return PIOS_malloc(size);
}

#endif	/* PIOS_INCLUDE_FASTHEAP */

void vPortFree(void * buf) __attribute__((alias ("PIOS_free")));
void PIOS_free(void * buf)
{
#if defined(PIOS_INCLUDE_FASTHEAP)
	if (is_ptr_in_heap_p(&pios_nodma_heap, buf))
		return simple_free(&pios_nodma_heap, buf);
#endif	/* PIOS_INCLUDE_FASTHEAP */

	if (is_ptr_in_heap_p(&pios_standard_heap, buf))
		return simple_free(&pios_standard_heap, buf);
}

size_t xPortGetFreeHeapSize(void) __attribute__((alias ("PIOS_heap_get_free_size")));
size_t PIOS_heap_get_free_size(void)
{
#if defined(PIOS_INCLUDE_FREERTOS)
	vTaskSuspendAll();
#endif	/* PIOS_INCLUDE_FREERTOS */

	size_t free_bytes = simple_get_free_bytes(&pios_standard_heap);

#if defined(PIOS_INCLUDE_FREERTOS)
	xTaskResumeAll();
#endif	/* PIOS_INCLUDE_FREERTOS */

	return free_bytes;
}

void vPortInitialiseBlocks(void) __attribute__((alias ("PIOS_heap_initialize_blocks")));
void PIOS_heap_initialize_blocks(void)
{
	/* NOP for the simple allocator */
}

void xPortIncreaseHeapSize(size_t bytes) __attribute__((alias ("PIOS_heap_increase_size")));
void PIOS_heap_increase_size(size_t bytes)
{
#if defined(PIOS_INCLUDE_FREERTOS)
	vTaskSuspendAll();
#endif	/* PIOS_INCLUDE_FREERTOS */

	simple_extend_heap(&pios_standard_heap, bytes);

#if defined(PIOS_INCLUDE_FREERTOS)
	xTaskResumeAll();
#endif	/* PIOS_INCLUDE_FREERTOS */
}

/**
 * @}
 * @}
 */
