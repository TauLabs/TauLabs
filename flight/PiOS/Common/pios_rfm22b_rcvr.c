/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_RFM22b RFM22b receiver functions
 * @brief Deals with the RFM22b module
 * @{
 *
 * @file       pios_rfm22b_rcvr.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2013.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      RFM22B Rcvr function. This catches UAVO updates from the
 *             remote RFM22B (normally from PPM) that are in the form of
 *             a UAVO and presents them as a PIOS_RCVR interface. This is
 *             almost identical to PIOS_GCSRCVR but looks at a different
 *             object.
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

#include <pios.h>

#ifdef PIOS_INCLUDE_RFM22B_RCVR

#include <uavobjectmanager.h>
#include <rfm22breceiver.h>
#include <pios_rfm22b_priv.h>
#include <pios_rfm22b_rcvr_priv.h>

#define PIOS_RFM22B_RCVR_TIMEOUT_MS  100

/* Provide a RCVR driver */
static int32_t PIOS_RFM22B_Rcvr_Get(uintptr_t rcvr_id, uint8_t channel);
static void PIOS_RFM22B_Rcvr_Supervisor(uintptr_t ppm_id);

const struct pios_rcvr_driver pios_rfm22b_rcvr_driver = {
	.read = PIOS_RFM22B_Rcvr_Get,
};

/* Local Variables */
enum pios_rfm22b_rcvr_dev_magic {
	PIOS_RFM22B_RCVR_DEV_MAGIC = 0x07ab9e2544cf5029,
};

struct pios_rfm22b_rcvr_dev {
	enum pios_rfm22b_rcvr_dev_magic magic;
	int16_t channels[RFM22B_PPM_NUM_CHANNELS];
	uint8_t supv_timer;
	bool fresh;
};

static bool PIOS_RFM22B_Rcvr_Validate(struct pios_rfm22b_rcvr_dev
				      *rfm22b_rcvr_dev)
{
	return rfm22b_rcvr_dev->magic == PIOS_RFM22B_RCVR_DEV_MAGIC;
}

static struct pios_rfm22b_rcvr_dev *PIOS_RFM22B_Rcvr_alloc(void)
{
	struct pios_rfm22b_rcvr_dev *rfm22b_rcvr_dev;

	rfm22b_rcvr_dev =
	    (struct pios_rfm22b_rcvr_dev *)
	    PIOS_malloc(sizeof(*rfm22b_rcvr_dev));
	if (!rfm22b_rcvr_dev) {
		return NULL;
	}

	rfm22b_rcvr_dev->magic = PIOS_RFM22B_RCVR_DEV_MAGIC;
	rfm22b_rcvr_dev->fresh = false;
	rfm22b_rcvr_dev->supv_timer = 0;

	return rfm22b_rcvr_dev;
}

extern int32_t PIOS_RFM22B_Rcvr_Init(uintptr_t * rfm22b_rcvr_id, uint32_t rfm22b_id)
{
	struct pios_rfm22b_rcvr_dev *rfm22b_rcvr_dev;

	/* Allocate the device structure */
	rfm22b_rcvr_dev =
	    (struct pios_rfm22b_rcvr_dev *)PIOS_RFM22B_Rcvr_alloc();
	if (!rfm22b_rcvr_dev) {
		return -1;
	}

	/* Register uavobj callback */
    RFM22BReceiverInitialize();
    RFM22BReceiverData rcvr;
	for (uint8_t i = 0; i < RFM22BRECEIVER_CHANNEL_NUMELEM; i++) {
		/* Flush channels */
		rcvr.Channel[i] = PIOS_RCVR_TIMEOUT;
	}
    RFM22BReceiverSet(&rcvr);

    *rfm22b_rcvr_id = (uintptr_t) rfm22b_rcvr_dev;
	PIOS_RFM22B_RegisterRcvr(rfm22b_id, *rfm22b_rcvr_id);

	/* Register the failsafe timer callback. */
	if (!PIOS_RTC_RegisterTickCallback
	    (PIOS_RFM22B_Rcvr_Supervisor, *rfm22b_rcvr_id)) {
		PIOS_DEBUG_Assert(0);
	}

	return 0;
}

/**
 * Called from the core driver to set the channel values whenever a
 * PPM packet is received. This method stores the data locally as well
 * as sets the data into the RFM22BReceiver UAVO for visibility
 */
int32_t PIOS_RFM22B_Rcvr_UpdateChannels(uintptr_t rfm22b_rcvr_id, int16_t * channels)
{
	/* Recover our device context */
	struct pios_rfm22b_rcvr_dev *rfm22b_rcvr_dev =
	    (struct pios_rfm22b_rcvr_dev *)rfm22b_rcvr_id;

	if (!PIOS_RFM22B_Rcvr_Validate(rfm22b_rcvr_dev)) {
		/* Invalid device specified */
		return -1;
	}

	for (uint8_t i = 0; i < RFM22B_PPM_NUM_CHANNELS; i++) {
		rfm22b_rcvr_dev->channels[i] = channels[i];
	}

	// Also store the received data in a UAVO for easy
	// debugging. However this is not what is used in
	// ManualControl (it fetches directly from this driver)
    RFM22BReceiverData rcvr;
	for (uint8_t i = 0; i < RFM22B_PPM_NUM_CHANNELS; i++) {
		if (i < RFM22BRECEIVER_CHANNEL_NUMELEM)
			rcvr.Channel[i] = channels[i];
	}
	for (int i = RFM22B_PPM_NUM_CHANNELS - 1; i < RFM22BRECEIVER_CHANNEL_NUMELEM; i++)
		rcvr.Channel[i] = PIOS_RCVR_INVALID;
	RFM22BReceiverSet(&rcvr);

	// let supervisor know we have new data
	rfm22b_rcvr_dev->fresh = true;

	return 0;
}

/**
 * Get the value of an input channel
 * \param[in] channel Number of the channel desired (zero based)
 * \output PIOS_RCVR_INVALID channel not available
 * \output PIOS_RCVR_TIMEOUT failsafe condition or missing receiver
 * \output >=0 channel value
 */
static int32_t PIOS_RFM22B_Rcvr_Get(uintptr_t rfm22b_rcvr_id, uint8_t channel)
{
	if (channel >= RFM22B_PPM_NUM_CHANNELS) {
		/* channel is out of range */
		return PIOS_RCVR_INVALID;
	}

	/* Recover our device context */
	struct pios_rfm22b_rcvr_dev *rfm22b_rcvr_dev =
	    (struct pios_rfm22b_rcvr_dev *)rfm22b_rcvr_id;

	if (!PIOS_RFM22B_Rcvr_Validate(rfm22b_rcvr_dev)) {
		/* Invalid device specified */
		return PIOS_RCVR_INVALID;
	}

	return rfm22b_rcvr_dev->channels[channel];
}

static void PIOS_RFM22B_Rcvr_Supervisor(uintptr_t rfm22b_rcvr_id)
{
	/* Recover our device context */
	struct pios_rfm22b_rcvr_dev *rfm22b_rcvr_dev =
	    (struct pios_rfm22b_rcvr_dev *)rfm22b_rcvr_id;

	if (!PIOS_RFM22B_Rcvr_Validate(rfm22b_rcvr_dev)) {
		/* Invalid device specified */
		return;
	}

	/*
	 * RTC runs at 625Hz.
	 */
	if (++(rfm22b_rcvr_dev->supv_timer) <
	    (PIOS_RFM22B_RCVR_TIMEOUT_MS * 1000 / 625)) {
		return;
	}
	rfm22b_rcvr_dev->supv_timer = 0;

	if (!rfm22b_rcvr_dev->fresh) {
		for (int32_t i = 0; i < RFM22BRECEIVER_CHANNEL_NUMELEM;
		     i++) {
			rfm22b_rcvr_dev->channels[i] = PIOS_RCVR_TIMEOUT;
		}
	}

	rfm22b_rcvr_dev->fresh = false;
}

#endif /* PIOS_INCLUDE_RFM22B_RCVR */

/**
 * @}
 * @}
 */
