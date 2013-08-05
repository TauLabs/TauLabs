/**
 ******************************************************************************
 * @file       pios_heap.h
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

#ifndef PIOS_HEAP_H
#define PIOS_HEAP_H

#include <stdlib.h>		/* size_t */
#include <stdbool.h>		/* bool */

extern bool PIOS_heap_malloc_failed_p(void);

extern void * PIOS_malloc_no_dma(size_t size);
extern void * PIOS_malloc(size_t size);

extern void PIOS_free(void * buf);

extern size_t PIOS_heap_get_free_size(void);
extern void PIOS_heap_initialize_blocks(void);
extern void PIOS_heap_increase_size(size_t bytes);

#endif	/* PIOS_HEAP_H */
