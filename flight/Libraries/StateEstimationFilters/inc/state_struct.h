/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 * @addtogroup StateEstimationFilters
 * @{
 *
 * @file       state_struct.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Common state structure used by @ref StateModule
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
#ifndef STATE_STRUCT_H
#define STATE_STRUCT_H

#include "openpilot.h"
#include "attitudesettings.h"
#include "sensorsettings.h"
#include "gyrosbias.h"

struct GlobalAttitudeVariables {
	//! How fast to estimate the gyro bias
	float accelKi;
	//! How much to change the attitude based on accelerometer data
	float accelKp;
	//! Sets the rate when assuming that average yaw is zero to null out yaw error
	float yawBiasRate;
	//! Current etimate of the gyro bias
	float gyro_correct_int[3];
	//! Local copy of current estimate of the attitude
	float q[4];
	//! Transformation from LLA to NED
	float T[3];
	//! Whether to rotate data from the sensor board frame to the body frame (if the frames are different)
	bool rotate;
	//! Rotation matrix which transforms from the body frame to the sensor board frame
	float Rsb[3][3];
	//! Indicates to estimate the gyro bias again while arming (faster convergence)
	bool zero_during_arming;
	//! Whether to apply the estimate of gyro bias to the data in the UAVO
	bool bias_correct_gyro;
	//! The currently active attitude estimator
	//! Indicates trimming is in progress and data is being accumulated
	bool trim_requested;
	//! Accumulator for the accel data during trimming
	float trim_accels[3];
	//! Counter of how many accel samples have been accumulated
	int32_t trim_samples;
};

typedef struct GlobalAttitudeVariables GlobalAttitudeVariables;


#endif /* STATE_STRUCT_H */
