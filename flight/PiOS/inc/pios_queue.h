/**
 ******************************************************************************
 * @file       pios_queue.h
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

#ifndef PIOS_QUEUE_H_
#define PIOS_QUEUE_H_

#define PIOS_QUEUE_TIMEOUT_MAX 0xffffffff

#include <stddef.h>
#include <stdbool.h>

#if defined(PIOS_INCLUDE_FREERTOS)

struct pios_queue
{
	uintptr_t queue_handle;
};

#elif defined(PIOS_INCLUDE_CHIBIOS)

#include "ch.h"

struct pios_queue
{
	Mailbox mb;
	MemoryPool mp;
	void *mpb;
};

#endif /* defined(PIOS_INCLUDE_FREERTOS) */

/*
 * The following functions implement the concept of a queue usable
 * with PIOS_INCLUDE_FREERTOS or PIOS_INCLUDE_CHIBIOS.
 *
 * for details see
 * http://www.freertos.org/a00018.html
 * http://chibios.sourceforge.net/html/group__mailboxes.html
 *
 */

struct pios_queue *PIOS_Queue_Create(size_t queue_length, size_t item_size);
void PIOS_Queue_Delete(struct pios_queue *queuep);
bool PIOS_Queue_Send(struct pios_queue *queuep, const void *itemp, uint32_t timeout_ms);
bool PIOS_Queue_Send_FromISR(struct pios_queue *queuep, const void *itemp, bool *wokenp);
bool PIOS_Queue_Receive(struct pios_queue *queuep, void *itemp, uint32_t timeout_ms);

#endif /* PIOS_QUEUE_H_ */

/**
  * @}
  * @}
  */
