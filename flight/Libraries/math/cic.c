/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 * @addtogroup TauLabsMath Tau Labs math support libraries
 * @{
 *
 * @file       cic.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
 * @brief      CIC Filter algorithms
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


/**
 * @brief Applying the higher order integration stage of the CIC filter, the lowest order integration must be already done in the function calling this (e.g. during data acquisition)
 * @param[in] filter_data Pointer to the CIC filter structure which stores the data / information
 */
void cic_apply_higher_order_int_stage(struct cic_filter_data * const filter_data) {
	if (filter_data->filter_order  >= 2) {
		filter_data->integrateState1 += filter_data->integrateState0;
	}

	if (filter_data->filter_order >= 3) {
		filter_data->integrateState2 += filter_data->integrateState1;
	}
}

/**
 * @brief Applying the comb stage of the CIC filter
 * @param[in] filter_data Pointer to the CIC filter structure which stores the data / information
 */
void cic_apply_comb_stage(struct cic_filter_data * const filter_data) {
	if (filter_data->filter_order == 1) {
		filter_data->combState0 = filter_data->integrateState0 - filter_data->integrateState_old;

		if (filter_data->filter_ddelay == 1) {
			filter_data->integrateState_old = filter_data->integrateState0;
		}
		else if (filter_data->filter_ddelay == 2) {
			filter_data->integrateState_old      = filter_data->integrateState_old_temp;
			filter_data->integrateState_old_temp = filter_data->integrateState0;
		}
	}

	else if (filter_data->filter_order == 2) {
		filter_data->combState0 = filter_data->integrateState1 - filter_data->integrateState_old;
		filter_data->combState1 = filter_data->combState0  - filter_data->combState0_old;

		if (filter_data->filter_ddelay == 1) {
			filter_data->integrateState_old = filter_data->integrateState1;
			filter_data->combState0_old = filter_data->combState0;
		}
		else if (filter_data->filter_ddelay == 2) {
			filter_data->integrateState_old      = filter_data->integrateState_old_temp;
			filter_data->integrateState_old_temp = filter_data->integrateState1;
			filter_data->combState0_old      = filter_data->combState0_old_temp;
			filter_data->combState0_old_temp = filter_data->combState0;
		}
	}

	else if (filter_data->filter_order == 3) {
		filter_data->combState0 = filter_data->integrateState2 - filter_data->integrateState_old;
		filter_data->combState1 = filter_data->combState0  - filter_data->combState0_old;
		filter_data->combState2 = filter_data->combState1  - filter_data->combState1_old;

		if (filter_data->filter_ddelay == 1) {
			filter_data->integrateState_old = filter_data->integrateState2;
			filter_data->combState0_old = filter_data->combState0;
			filter_data->combState1_old = filter_data->combState1;
		}
		else if (filter_data->filter_ddelay == 2) {
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
int32_t cic_get_decimation_output(struct cic_filter_data * const filter_data) {
	if (filter_data->filter_order == 1)
		return filter_data->combState0;

	else if (filter_data->filter_order == 2)
		return filter_data->combState1;

	else if (filter_data->filter_order == 3)
		return filter_data->combState2;

	else // (filter_data->filter_order == 0)
		return filter_data->integrateState0;
}


/**
 * @brief Get the gain of the CIC filter, e.g. for normalisation
 * @param[in] filter_data Pointer to the CIC filter structure which stores the data / information
 * @returns Output the computed gain of the filter
 */
float cic_get_gain(struct cic_filter_data * const filter_data) {
	// CIC:
	// Differential Delay N = D/R = Delay / Down Sampling factor
	// Gain: (N*R)^M
	if (filter_data->filter_order == 0) // simple boxcar averaging
		return (filter_data->filter_decimation);
	else
		return (pow((filter_data->filter_decimation * filter_data->filter_ddelay), filter_data->filter_order));
}


/**
 * @brief Configure the common parameters of the CIC filter and check if they are allowed, otherwise change them
 * @param[in] filter_data Pointer to the CIC filter structure which stores the data / information
 * @param[in] filter_order The order of the CIC filter
 * @param[in] filter_ddelay  The differential delay of the CIC filter
 * @param[in] filter_ddelay  The decimation factor of the CIC filter
 */
void cic_configure(struct cic_filter_data * const filter_data, uint8_t filter_order, uint8_t filter_ddelay,  uint8_t filter_decimation) {
	float needed_bit_width = 0;
	// formula for calculating the variable bit width to prevent unwanted overflow:
	// bit_width_output = bit_width_input + filter_order * log2(differential_delay * decimation_factor)
	//                  = bit_width_input + M * log2(N*R)
	// e.g. 16 bit raw data, 32 bit variables for M=3 and N=2 20x decimation should work (16 + 3 * log2(2 * 20) = 31.97)
	filter_data->filter_order     = bound_min_max(filter_order, 0, 3); // M
	filter_data->filter_ddelay    = bound_min_max(filter_ddelay, 1, 2); // N
	filter_data->filter_decimation= bound_min_max(filter_decimation, 1, 255); // R

	//TODO: if the infrastructure for alarms will be more advanced in the future,
	//      we should assert a alarm here when bit width is to large and parameter of the filter must be reduced

	// we get 16 bit data from the mpu, and have 32bit variables for the filter states
	// check if the needed bit width is larger than 32, if yes at first reduce the differential delay, than the filter order
	needed_bit_width = (float)(16.0 + filter_data->filter_order * log2(filter_data->filter_ddelay * filter_data->filter_decimation));
	while (needed_bit_width > 32.0f) {
		if (filter_data->filter_ddelay > 1) {
			filter_data->filter_ddelay--;
			needed_bit_width = (float)(16.0 + filter_data->filter_order * log2(filter_data->filter_ddelay * filter_data->filter_decimation));
		}
		else if (filter_data->filter_order > 0) {
				filter_data->filter_order--;
				needed_bit_width = (float)(16.0 + filter_data->filter_order * log2(filter_data->filter_ddelay * filter_data->filter_decimation));
		}
	}
}

/**
 * @}
 * @}
 */
