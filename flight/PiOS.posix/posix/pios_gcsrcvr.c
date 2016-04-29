/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_GCSRCVR GCS Receiver Input Functions
 * @brief		Code to read the channels within the GCS Receiver UAVObject
 * @{
 *
 * @file       pios_gcsrcvr.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      GCS Input functions (STM32 dependent)
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

/* Project Includes */
#include "pios.h"

#if defined(PIOS_INCLUDE_GCSRCVR)
#include "openpilot.h"
#include "pios_gcsrcvr_priv.h"

static GCSReceiverData gcsreceiverdata;

#define VALID_WHEN_MISSING

/* Provide a RCVR driver */
static int32_t PIOS_GCSRCVR_Get(uintptr_t rcvr_id, uint8_t channel);

const struct pios_rcvr_driver pios_gcsrcvr_rcvr_driver = {
	.read = PIOS_GCSRCVR_Get,
};

/* Local Variables */
enum pios_gcsrcvr_dev_magic {
	PIOS_GCSRCVR_DEV_MAGIC = 0xe9da5c56,
};

struct pios_gcsrcvr_dev {
	enum pios_gcsrcvr_dev_magic magic;

	uint8_t supv_timer;
	bool Fresh;
};

static struct pios_gcsrcvr_dev *global_gcsrcvr_dev;

#if defined(PIOS_INCLUDE_FREERTOS)
static struct pios_gcsrcvr_dev *PIOS_gcsrcvr_alloc(void)
{
	struct pios_gcsrcvr_dev * gcsrcvr_dev;

	gcsrcvr_dev = (struct pios_gcsrcvr_dev *)PIOS_malloc(sizeof(*gcsrcvr_dev));
	if (!gcsrcvr_dev) return(NULL);

	gcsrcvr_dev->magic = PIOS_GCSRCVR_DEV_MAGIC;
	gcsrcvr_dev->Fresh = false;
	gcsrcvr_dev->supv_timer = 0;

	/* The update callback cannot receive the device pointer, so set it in a global */
	global_gcsrcvr_dev = gcsrcvr_dev;

	return(gcsrcvr_dev);
}
#else
static struct pios_gcsrcvr_dev pios_gcsrcvr_devs[PIOS_GCSRCVR_MAX_DEVS];
static uint8_t pios_gcsrcvr_num_devs;
static struct pios_gcsrcvr_dev *PIOS_gcsrcvr_alloc(void)
{
	struct pios_gcsrcvr_dev *gcsrcvr_dev;

	if (pios_gcsrcvr_num_devs >= PIOS_GCSRCVR_MAX_DEVS) {
		return (NULL);
	}

	gcsrcvr_dev = &pios_gcsrcvr_devs[pios_gcsrcvr_num_devs++];
	gcsrcvr_dev->magic = PIOS_GCSRCVR_DEV_MAGIC;
	gcsrcvr_dev->Fresh = false;
	gcsrcvr_dev->supv_timer = 0;

	global_gcsrcvr_dev = gcsrcvr_dev;

	return (gcsrcvr_dev);
}
#endif

static void gcsreceiver_updated(UAVObjEvent * ev, void *ctx, void *obj,
		int len)
{
	(void) ev; (void) ctx; (void) obj; (void) len;

	struct pios_gcsrcvr_dev *gcsrcvr_dev = global_gcsrcvr_dev;
	if (ev->obj == GCSReceiverHandle()) {
		GCSReceiverGet(&gcsreceiverdata);
		gcsrcvr_dev->Fresh = true;
	}
}

extern int32_t PIOS_GCSRCVR_Init(uintptr_t *gcsrcvr_id)
{
	struct pios_gcsrcvr_dev *gcsrcvr_dev;

	/* Allocate the device structure */
	gcsrcvr_dev = (struct pios_gcsrcvr_dev *)PIOS_gcsrcvr_alloc();
	if (!gcsrcvr_dev)
		return -1;

	/* Register uavobj callback */
	GCSReceiverConnectCallback (gcsreceiver_updated);

	return 0;
}

static int32_t PIOS_GCSRCVR_Get(uintptr_t rcvr_id, uint8_t channel)
{
	if (channel >= GCSRECEIVER_CHANNEL_NUMELEM) {
		/* channel is out of range */
		return -1;
	}

#ifdef VALID_WHEN_MISSING
	return (gcsreceiverdata.Channel[channel] != 0) ? (gcsreceiverdata.Channel[channel]) : 1500;
#else 
	return (gcsreceiverdata.Channel[channel]);
#endif

}

#endif	/* PIOS_INCLUDE_GCSRCVR */

/** 
  * @}
  * @}
  */
