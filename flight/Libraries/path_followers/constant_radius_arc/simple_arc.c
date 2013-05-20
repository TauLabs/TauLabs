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
 * @brief simple_line_follower Calculate command for following simple vector-based orbit. A
 * full description of parameters as well as a proof of convergence is given in
 * "Small Unmanned Aircraft-- Theory and Practice", R. Beard and T. McLain, 2011.
 * @param positionActual Current vehicle position
 * @param c[2] Center of arc in North-East coordinates
 * @param rho Arcradius
 * @param curvature_sign Sense of the arc
 * @param k_orbit Gain on radial distance error
 * @param k_psi_int Gain on radial-distance error integral
 * @param delT Time step between iterations
 * @param arc_error_accum Radial-distance error integral
 * @return psi_command course command
 */
float simple_arc_follower(PositionActualData *positionActual, float c[2], 
								  float rho, int8_t curvature_sign, 
								  float k_orbit, float k_psi_int, 
								  float *arc_error_accum, float delT)
{
	float p[2]={positionActual->North, positionActual->East};

	float pncn = p[0] - c[0];
	float pece = p[1] - c[1];
	float d = sqrtf(pncn*pncn + pece*pece);

	float err_orbit = d - rho;
	*arc_error_accum += err_orbit*delT;

	float phi = atan2f(pece, pncn);

	float psi_command = (curvature_sign > 0) ?
		phi + (PI/2.0f + atanf(k_orbit*err_orbit) + k_psi_int*(*arc_error_accum)): // Turn clockwise
		phi - (PI/2.0f + atanf(k_orbit*err_orbit) + k_psi_int*(*arc_error_accum)); // Turn counter-clockwise

	return psi_command;
}
