/**
 ******************************************************************************
 * @addtogroup AHRS 
 * @{
 * @addtogroup INSGPS
 * @{
 * @brief INSGPS is a joint attitude and position estimation EKF
 *
 * @file       insgps.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://github.com/TauLabs Copyright (C) 2012-2013.
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
#define HORIZ_SENSORS 0x018
#define VERT_SENSORS  0x020
#define MAG_SENSORS 0x1C0
#define BARO_SENSOR 0x200

#define FULL_SENSORS 0x3FF

/**
  * @}
  */

//  Exposed Function Prototypes
void INSGPSInit();
struct NavStruct *INSGPSGetNav();
void INSStatePrediction(const float gyro_data[3], const float accel_data[3], float dT);
void INSCovariancePrediction(float dT);
void INSCorrection(const float mag_data[3], const float Pos[3], const float Vel[3], float BaroAlt, uint16_t SensorsUsed);

void INSResetP(const float PDiag[13]);
void INSSetState(const float pos[3], const float vel[3], const float q[4], const float gyro_bias[3], const float accel_bias[3]);
void INSSetPosVelVar(float PosVar, float VelVar);
void INSSetGyroBias(const float gyro_bias[3]);
void INSSetAccelVar(const float accel_var[3]);
void INSSetGyroVar(const float gyro_var[3]);
void INSSetMagNorth(const float B[3]);
void INSSetMagVar(const float scaled_mag_var[3]);
void INSSetBaroVar(float baro_var);
void INSPosVelReset(const float pos[3], const float vel[3]);

void MagCorrection(const float mag_data[3]);
void MagVelBaroCorrection(const float mag_data[3], const float Vel[3], float BaroAlt);
void FullCorrection(const float mag_data[3], const float Pos[3], const float Vel[3],
		    float BaroAlt);
void GpsBaroCorrection(const float Pos[3], const float Vel[3], float BaroAlt);
void GpsMagCorrection(const float mag_data[3], const float Pos[3], const float Vel[2]);
void VelBaroCorrection(const float Vel[3], float BaroAlt);

uint16_t ins_get_num_states();

//  Nav structure containing current solution
struct NavStruct {
	float Pos[3];		// Position in meters and relative to a local NED frame
	float Vel[3];		// Velocity in meters and in NED
	float q[4];		// unit quaternion rotation relative to NED
	float gyro_bias[3];
	float accel_bias[3];
};

/**
 * @}
 * @}
 */

#endif /* EKF_H_ */
