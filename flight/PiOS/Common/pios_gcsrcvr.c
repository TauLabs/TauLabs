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
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

#include "pios_gcsrcvr_priv.h"

static GCSReceiverData gcsreceiverdata;

/* Provide a RCVR driver */
static int32_t PIOS_GCSRCVR_Get(uintptr_t rcvr_id, uint8_t channel);
static void PIOS_gcsrcvr_Supervisor(uintptr_t rcvr_id);

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

static bool PIOS_gcsrcvr_validate(struct pios_gcsrcvr_dev *gcsrcvr_dev)
{
	return (gcsrcvr_dev->magic == PIOS_GCSRCVR_DEV_MAGIC);
}

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

static void gcsreceiver_updated(UAVObjEvent * ev)
{
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

	for (uint8_t i = 0; i < GCSRECEIVER_CHANNEL_NUMELEM; i++) {
		/* Flush channels */
		gcsreceiverdata.Channel[i] = PIOS_RCVR_TIMEOUT;
	}

	/* Register uavobj callback */
	GCSReceiverConnectCallback (gcsreceiver_updated);

	/* Register the failsafe timer callback. */
	if (!PIOS_RTC_RegisterTickCallback(PIOS_gcsrcvr_Supervisor, (uintptr_t)gcsrcvr_dev)) {
		PIOS_DEBUG_Assert(0);
	}

	return 0;
}

/**
 * Get the value of an input channel
 * \param[in] channel Number of the channel desired (zero based)
 * \output PIOS_RCVR_INVALID channel not available
 * \output PIOS_RCVR_TIMEOUT failsafe condition or missing receiver
 * \output >=0 channel value
 */
static int32_t PIOS_GCSRCVR_Get(uintptr_t rcvr_id, uint8_t channel)
{
	if (channel >= GCSRECEIVER_CHANNEL_NUMELEM) {
		/* channel is out of range */
		return PIOS_RCVR_INVALID;
	}

	return (gcsreceiverdata.Channel[channel]);
}

static void PIOS_gcsrcvr_Supervisor(uintptr_t gcsrcvr_id) {
	/* Recover our device context */
	struct pios_gcsrcvr_dev * gcsrcvr_dev = (struct pios_gcsrcvr_dev *)gcsrcvr_id;

	if (!PIOS_gcsrcvr_validate(gcsrcvr_dev)) {
		/* Invalid device specified */
		return;
	}

	/* 
	 * RTC runs at 625Hz.
	 */
	if(++(gcsrcvr_dev->supv_timer) < (PIOS_GCSRCVR_TIMEOUT_MS * 1000 / 625)) {
		return;
	}
	gcsrcvr_dev->supv_timer = 0;

	if (!gcsrcvr_dev->Fresh)
		for (int32_t i = 0; i < GCSRECEIVER_CHANNEL_NUMELEM; i++)
			gcsreceiverdata.Channel[i] = PIOS_RCVR_TIMEOUT;

	gcsrcvr_dev->Fresh = false;
}

#endif	/* PIOS_INCLUDE_GCSRCVR */

/** 
  * @}
  * @}
  */
