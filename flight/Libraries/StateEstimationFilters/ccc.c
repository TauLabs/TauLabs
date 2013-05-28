/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 * @addtogroup StateEstimationFilters
 * @{
 *
 * @file       ccc.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Complementary filter implementation used by @ref StateModule
 *
 * @see        The GNU Public License (GPL) Version 3
 *
 ******************************************************************************/
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

#include "pios.h"
#include "ccc.h"
#include <pios_board_info.h>
#include "coordinate_conversions.h"

#include "accels.h"
#include "gyros.h"

//Global variables

// Private constants

// Private types

// Private variables

// Private functions

/*
 * Correct sensor drift, using the 3C approach by J. Cotton
 */
void CottonComplementaryCorrection(float *accels, float *gyros, const float delT, GlobalAttitudeVariables *glblAtt, float *accel_err_b)
{
	float grav_b[3];
	
	// Rotate normalized gravity reference vector to body frame and cross with measured acceleration
	grav_b[0] = -(2 * (glblAtt->q[1] * glblAtt->q[3] - glblAtt->q[0] * glblAtt->q[2]));
	grav_b[1] = -(2 * (glblAtt->q[2] * glblAtt->q[3] + glblAtt->q[0] * glblAtt->q[1]));
	grav_b[2] = -(glblAtt->q[0] * glblAtt->q[0] - glblAtt->q[1] * glblAtt->q[1] -
					  glblAtt->q[2] * glblAtt->q[2] + glblAtt->q[3] * glblAtt->q[3]);
	CrossProduct((const float *)accels, (const float *)grav_b, accel_err_b);
	
	// Account for accel magnitude
	float accel_mag = VectorMagnitude(accels);
	if (accel_mag < 1.0e-3f)
		return;
	
	// Normalize error vector
	accel_err_b[0] /= accel_mag;
	accel_err_b[1] /= accel_mag;
	accel_err_b[2] /= accel_mag;
	
	// Accumulate integral of error.  Scale here so that units are (deg/s) but accelKi has units of s
	glblAtt->gyro_correct_int[0] += accel_err_b[0] * glblAtt->accelKi;
	glblAtt->gyro_correct_int[1] += accel_err_b[1] * glblAtt->accelKi;
	
	// Because most crafts wont get enough information from gravity to zero yaw gyro, we try
	// and make it average zero (weakly)
	glblAtt->gyro_correct_int[2] += -gyros[2] * glblAtt->yawBiasRate;
	
	// In this step, correct rates based on proportional error. The integral component is applied in updateSensors
	gyros[0] += accel_err_b[0] * glblAtt->accelKp / delT;
	gyros[1] += accel_err_b[1] * glblAtt->accelKp / delT;
	gyros[2] += accel_err_b[2] * glblAtt->accelKp / delT;
}

/**
 * @}
 * @}
 */
