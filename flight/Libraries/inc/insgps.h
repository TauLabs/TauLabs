/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 *
 * @file       insgps.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Include file of the INSGPS exposed functionality.
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

#ifndef INSGPS_H_
#define INSGPS_H_

#include "stdint.h"

/**
  * @addtogroup Constants
  * @{
  */
#define POS_SENSORS 0x007
#define HORIZ_POS_SENSORS 0x003
#define VERT_POS_SENSORS 0x004
#define HORIZ_VEL_SENSORS 0x018
#define VERT_VEL_SENSORS  0x020
#define MAG_SENSORS 0x1C0
#define BARO_SENSOR 0x200

#define FULL_SENSORS 0x3FF

/**
  * @}
  */

/****************************************************/
/**  Main interface for running the filter         **/
/****************************************************/

//! Reset the internal state variables and variances
void INSGPSInit();

//! Compute an update of the state estimate
void INSStatePrediction(const float gyro_data[3], const float accel_data[3], float dT);

//! Compute an update of the state covariance
void INSCovariancePrediction(float dT);

//! Correct the state and covariance estimate based on the sensors that were updated
void INSCorrection(const float mag_data[3], const float Pos[3], const float Vel[3], float BaroAlt, uint16_t SensorsUsed);

//! Get the current state estimate
void INSGetState(float *pos, float *vel, float *attitude, float *gyro_bias, float *accel_bias);

/****************************************************/
/** These methods alter the behavior of the filter **/
/****************************************************/

void INSResetP(const float *PDiag);
void INSSetState(const float pos[3], const float vel[3], const float q[4], const float gyro_bias[3], const float accel_bias[3]);
void INSSetPosVelVar(float PosVar, float VelVar, float VertPosVar);
void INSSetGyroBias(const float gyro_bias[3]);
void INSSetAccelBias(const float gyro_bias[3]);
void INSSetAccelVar(const float accel_var[3]);
void INSSetGyroVar(const float gyro_var[3]);
void INSSetMagNorth(const float B[3]);
void INSSetMagVar(const float scaled_mag_var[3]);
void INSSetBaroVar(float baro_var);
void INSPosVelReset(const float pos[3], const float vel[3]);

void INSGetVariance(float *p);

uint16_t ins_get_num_states();

#endif /* INSGPS_H_ */

/**
 * @}
 */
 