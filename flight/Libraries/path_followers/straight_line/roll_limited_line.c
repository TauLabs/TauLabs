/**
 ******************************************************************************
 * @file       roll_limited_line.c
 * @author     Tau Labs, http://www.taulabs.org, Copyright (C) 2013
 * @addtogroup Path Followers
 * @{
 * @brief Miscellaneous math support
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

#include "path_followers.h"
#include "misc_math.h"
#include "physical_constants.h"

#include "positionactual.h"
#include "velocityactual.h"
#include "pathdesired.h"

/**
 * 
 */
/**
 * @brief This heading controller computes a roll command command based on
 * a line-following vector field, taking into account the vehicle's roll angle
 * contraints. A full description of parameters as well as a proof of convergence
 * is given in "Fixed Wing UAV Path Following in Wind with Input Constraints",
 * R. Beard, 2012.
 * @param positionActual Vehicle's current position vector
 * @param velocityActual Vehicle's current velocity vector
 * @param pathDesired Structure containing initial and final path points
 * @param true_airspeed TAS
 * @param true_airspeed_desired TAS setpoint
 * @param headingActual_R Current heading in [rad]
 * @param gamma_max Maximum flight path angle that can be commanded by autonomous flight modules
 * @param phi_max Maximum roll angle that can be commanded by autonomous flight modules
 * @return
 */

float roll_limited_line_follower(PositionActualData *positionActual, VelocityActualData *velocityActual, 
											PathDesiredData *pathDesired, float true_airspeed, float true_airspeed_desired,
											float headingActual_R, float gamma_max, float phi_max)
{
	float gamma;
	float err_xt;
	float err_xt_dot;

	float psi = headingActual_R;
	float psi_tilde_thresh = PI/4; // Beyond 45 degrees off course, full roll is applied. Fixme: This should not be hard-coded

	float p[3]={positionActual->North, positionActual->East, positionActual->Down};
	float *c = pathDesired->End;
	float *r = pathDesired->Start;
	float q[3] = {c[0]-r[0], c[1]-r[1], c[2]-r[2]};

	float V = true_airspeed;

	gamma = atan2f(velocityActual->Down, sqrtf(powf(velocityActual->North, 2) + powf(velocityActual->East, 2)));


	// Roll command
	float roll_c_R;

	float k1 = 3.9/true_airspeed_desired; // Dividing by airspeed ensures that roll rates stay constant with increasing scale
	float k2 = 1;

	float chi_q=atan2f(q[1], q[0]);
	float psi_tilde = circular_modulus_rad(psi - chi_q); // This is the difference between the vehicle heading and the path heading

	err_xt = -sinf(chi_q)*(p[0] - r[0]) + cosf(chi_q)*(p[1] - r[1]); // Compute cross-track error
	err_xt_dot = V * sinf(psi_tilde) * cosf(gamma);// + wind_y //TODO: add wind estimate in the local reference frame

	if (psi_tilde < -psi_tilde_thresh)
		roll_c_R = phi_max;
	else if (psi_tilde > psi_tilde_thresh)
		roll_c_R = -phi_max;
	else
	{
		float M1 = tanf(phi_max);
		float M2 = GRAVITY/2.0f * M1 * cosf(psi_tilde_thresh) * cosf(gamma_max);
		roll_c_R = -bound_sym((k1*err_xt_dot + bound_sym(k2*(k1*err_xt + err_xt_dot), M2))/(GRAVITY*cosf(psi_tilde)*cosf(gamma)), M1);
	}

	return roll_c_R;
}
