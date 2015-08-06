/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 * @addtogroup TauLabsMath Tau Labs math support libraries
 * @{
 *
 * @file       pid.h
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

#ifndef CIC_H
#define CIC_H



// prepared for CIC filter up to order M=3 & differential delay up to N=2
// when order = 0 simple boxcar filter / averaging is done
// data variables should be max int16 to prevent overflow; for state variables 32bit variables are used

struct cic_filter_data {
	int32_t  integrateState0;
	int32_t  integrateState1;
	int32_t  integrateState2;
	int32_t  combState0;
	int32_t  combState1;
	int32_t  combState2;
	int32_t  combState0_old;
	int32_t  combState1_old;
	int32_t  integrateState_old;
	int32_t  combState0_old_temp;
	int32_t  combState1_old_temp;
	int32_t  integrateState_old_temp;
};

//! Methods to use the cic structures
void cic_apply_higher_order_int_stage(struct cic_filter_data *filter_data);
void cic_apply_comb_stage(struct cic_filter_data *filter_data);
int32_t cic_get_decimation_output(struct cic_filter_data *filter_data);
float cic_get_gain(void);
void cic_configure(uint8_t filter_order, uint8_t filter_ddelay,  uint8_t filter_decimation);

#endif /* CIC_H */

/**
 * @}
 * @}
 */
