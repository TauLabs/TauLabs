/**
 ******************************************************************************
 * @file       simple_arc.c
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
#include "physical_constants.h"

/**
 * Calculate command for following simple vector based orbit. Taken from R. Beard at BYU.
 */
float simple_arc_follower(PositionActualData *positionActual, float c[2], float rho, float curvature, float k_orbit, float k_psi_int, float delT, Integral *integral)
{
	float p[2]={positionActual->North, positionActual->East};

	float pncn = p[0] - c[0];
	float pece = p[1] - c[1];
	float d = sqrtf(pncn*pncn + pece*pece);

	float err_orbit = d - rho;
	integral->circle_error += err_orbit*delT;

	float phi = atan2f(pece, pncn);

	float psi_command = (curvature > 0) ?
		phi + (PI/2.0f + atanf(k_orbit*err_orbit) + k_psi_int*integral->circle_error): // Turn clockwise
		phi - (PI/2.0f + atanf(k_orbit*err_orbit) + k_psi_int*integral->circle_error); // Turn counter-clockwise

	return psi_command;
}
