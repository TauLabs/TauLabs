/**
 ******************************************************************************
 * @file       simple_line.c
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
 * Calculate command for following simple vector based line. Taken from R. Beard at BYU.
 */
float simple_line_follower(PositionActualData *positionActual, PathDesiredData *pathDesired, float chi_inf, float k_path, float k_psi_int, float delT, Integral *integral)
{
	float p[2]={positionActual->North, positionActual->East};

	float *c = pathDesired->End;
	float *r = pathDesired->Start;
	float q[2] = {c[0]-r[0], c[1]-r[1]};


	float chi_q=atan2f(q[1], q[0]);

	float err_xt=-sinf(chi_q)*(p[0]-r[0])+cosf(chi_q)*(p[1]-r[1]); // Compute cross-track error
	integral->line_error+=delT*err_xt;
	float psi_command = chi_q-chi_inf*2.0f/PI*atanf(k_path*err_xt)-k_psi_int*integral->line_error;

	return psi_command;
}
