/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 * @addtogroup TauLabsMath Tau Labs math support libraries
 * @{
 *
 * @file       pid.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      PID Control algorithms
 *
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

#include <math.h>
#include "misc_math.h"
#include "cic.h"


//! Store the filter parameter
static uint8_t cic_filter_order     = 1;
static uint8_t cic_filter_ddelay    = 1;
static uint8_t cic_filter_decimation= 1;

/**
 * @brief Applying the higher order integration stage of the CIC filter, the lowest order integration must be already done in the function calling this (e.g. during data acquisition)
 * @param[in] filter_data Pointer to the CIC filter structure which stores the data / information
 */
void cic_apply_higher_order_int_stage(struct cic_filter_data *filter_data) {
	if (cic_filter_order  >= 2) {
		filter_data->integrateState1 += filter_data->integrateState0;
	}

	if (cic_filter_order >= 3) {
		filter_data->integrateState2 += filter_data->integrateState1;
	}
}

/**
 * @brief Applying the comb stage of the CIC filter
 * @param[in] filter_data Pointer to the CIC filter structure which stores the data / information
 */
void cic_apply_comb_stage(struct cic_filter_data *filter_data) {
	if (cic_filter_order == 1) {
		filter_data->combState0 = filter_data->integrateState0 - filter_data->integrateState_old;

		if (cic_filter_ddelay == 1) {
			filter_data->integrateState_old = filter_data->integrateState0;
		}
		else if (cic_filter_ddelay == 2) {
			filter_data->integrateState_old      = filter_data->integrateState_old_temp;
			filter_data->integrateState_old_temp = filter_data->integrateState0;
		}
	}

	else if (cic_filter_order == 2) {
		filter_data->combState0 = filter_data->integrateState1 - filter_data->integrateState_old;
		filter_data->combState1 = filter_data->combState0  - filter_data->combState0_old;

		if (cic_filter_ddelay == 1) {
			filter_data->integrateState_old = filter_data->integrateState1;
			filter_data->combState0_old = filter_data->combState0;
		}
		else if (cic_filter_ddelay == 2) {
			filter_data->integrateState_old      = filter_data->integrateState_old_temp;
			filter_data->integrateState_old_temp = filter_data->integrateState1;
			filter_data->combState0_old      = filter_data->combState0_old_temp;
			filter_data->combState0_old_temp = filter_data->combState0;
		}
	}

	else if (cic_filter_order == 3) {
		filter_data->combState0 = filter_data->integrateState2 - filter_data->integrateState_old;
		filter_data->combState1 = filter_data->combState0  - filter_data->combState0_old;
		filter_data->combState2 = filter_data->combState1  - filter_data->combState1_old;

		if (cic_filter_ddelay == 1) {
			filter_data->integrateState_old = filter_data->integrateState2;
			filter_data->combState0_old = filter_data->combState0;
			filter_data->combState1_old = filter_data->combState1;
		}
		else if (cic_filter_ddelay == 2) {
			filter_data->integrateState_old      = filter_data->integrateState_old_temp;
			filter_data->integrateState_old_temp = filter_data->integrateState2;
			filter_data->combState0_old      = filter_data->combState0_old_temp;
			filter_data->combState0_old_temp = filter_data->combState0;
			filter_data->combState1_old      = filter_data->combState1_old_temp;
			filter_data->combState1_old_temp = filter_data->combState1;
		}
	}
}

/**
 * @brief Get the decimated / down-sampled output of the CIC filter
 * @param[in] filter_data Pointer to the CIC filter structure which stores the data / information
 * @returns Output the computed gain of the filter
 */
int32_t cic_get_decimation_output(struct cic_filter_data *filter_data) {
	if (cic_filter_order == 1)
		return filter_data->combState0;

	else if (cic_filter_order == 2)
		return filter_data->combState1;

	else if (cic_filter_order == 3)
		return filter_data->combState2;

	else // (cic_filter_order == 0)
		return filter_data->integrateState0;
}


/**
 * @brief Get the gain of the CIC filter, e.g. for normalisation
 * @returns Output the computed gain of the filter
 */
float cic_get_gain(void) {
	// CIC:
	// Differential Delay N = D/R = Delay / Downsampling factor
	// Gain: (N*R)^M
	if (cic_filter_order == 0) // simple boxcar averaging
		return (cic_filter_decimation);
	else
		return (pow((cic_filter_decimation * cic_filter_ddelay), cic_filter_order));
}


/**
 * @brief Configure the common parameters of the CIC filter
 * @param[in] cic_filter_order The order of the CIC filter
 * @param[in] cic_filter_ddelay  The differential delay of the CIC filter
 * @param[in] cic_filter_ddelay  The decimation factor of the CIC filter
 */
void cic_configure(uint8_t filter_order, uint8_t filter_ddelay,  uint8_t filter_decimation) {
	cic_filter_order     = bound_min_max(filter_order, 0, 3);
	cic_filter_ddelay    = bound_min_max(filter_ddelay, 1, 2);
	cic_filter_decimation= bound_min_max(filter_decimation, 1, 255);
}

/**
 * @}
 * @}
 */
