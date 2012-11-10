/**
 ******************************************************************************
 * @addtogroup OpenPilotModules OpenPilot Modules
 * @{ 
 * @addtogroup State State Module
 * @{ 
 *
 * @file       state.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @brief      Acquires sensor data and fuses it into attitude estimate for CC
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
#ifndef STATE_H
#define STATE_H

#include "openpilot.h"

struct GlobalAttitudeVariables {
	//! How fast to estimate the gyro bias
	float accelKi;
	//! How much to change the attitude based on accelerometer data
	float accelKp;
	//! Sets the rate when assuming that average yaw is zero to null out yaw error
	float yawBiasRate;
	//! Pre channel calibration of the gyro gain
	float gyroGain[3];
	//! Nominal scale which is different for CC3D and CC
	float gyroGain_ref;
	//! Estimate of the accel bias (in board frame)
	float accelbias[3];
	//! Estimate of the accel scale (in board frame)
	float accelscale[3];
	//! Current etimate of the gyro bias
	float gyro_correct_int[3];
	//! Local copy of current estimate of the attitude
	float q[4];
	//! Transformation from LLA to NED
	float T[3];
	//! Whether to rotate data from the board frame to the body frame (if they are different)
	bool rotate;
	//! Rotation matrix that transforms from the sensor frame to the body frame
	float Rsb[3][3]; 
	//! Indicates to estimate the gyro bias again while arming (faster convergence)
	bool zero_during_arming;
	//! Whether to apply the estimate of gyro bias to the data in the UAVO
	bool bias_correct_gyro;
	//! The currently active attitude estimator
	uint8_t filter_choice;
	//! Indicates trimming is in progress and data is being accumulated
	bool trim_requested;
	//! Accumulator for the accel data during trimming
	float trim_accels[3];
	//! Counter of how many accel samples have been accumulated
	int32_t trim_samples;
};


int32_t AttitudeInitialize(void);


#endif // STATE_H
