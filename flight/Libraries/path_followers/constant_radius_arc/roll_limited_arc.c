/**
 ******************************************************************************
 * @file       roll_limited_arc.c
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

#include "misc_math.h"
#include "path_followers.h"
#include "physical_constants.h"

#include "positionactual.h"
#include "velocityactual.h"


/**
 * @brief This heading controller computes a roll command command based on
 * an arc-following vector field, taking into account the vehicle's roll angle
 * contraints. A full description of parameters as well as a proof of convergence
 * is given in "Fixed Wing UAV Path Following in Wind with Input Constraints"
 * @param positionActual Vehicle's current position vector
 * @param velocityActual Vehicle's current velocity vector
 * @param arc_center_NE Arc center in North-East coordinates
 * @param curvature_sign Sense of arc
 * @param arc_radius Arc radius (strictly positive)
 * @param true_airspeed TAS
 * @param true_airspeed_desired TAS setpoint
 * @param headingActual_R Current heading in [rad]
 * @param gamma_max Maximum flight path angle that can be commanded by autonomous flight modules
 * @param phi_max Maximum roll angle that can be commanded by autonomous flight modules
 * @return roll_c_R Constrained roll command
 */
float roll_limited_arc_follower(PositionActualData *positionActual, VelocityActualData *velocityActual,
									  float arc_center_NE[2], int8_t curvature_sign, float arc_radius,
									  float true_airspeed, float true_airspeed_desired,
								      float headingActual_R, float gamma_max, float phi_max)
{
	float psi = headingActual_R;
	float psi_tilde_thresh = PI/4; // Beyond 45 degrees of course, full roll is applied. FIXME: This shouldn't be hard coded, but it needs to be strictly positive.

	float p[2]={positionActual->North, positionActual->East};
	float *c = arc_center_NE;

	float V = true_airspeed;

	// This is not strictly correct, as gamma is the "air-mass" referenced flight path, whereas using velocityActual
	// leads to the ground-referenced flight path. However, the difference is not critical as the flight paths will
	// tend to be relatively shallow.
	float gamma = atan2f(velocityActual->Down, sqrtf(powf(velocityActual->North, 2) + powf(velocityActual->East, 2)));

	// Fixme: Tuning values shouldn't be hard coded
	float k4 = 7.5/true_airspeed_desired;
	float k5 = 0.4;

	// Determine sense of arc path
	int8_t lambda;
	if (curvature_sign < 0)
		lambda = -1;
	else
		lambda = 1;

	float psi_wind;
	float W;
#ifdef WIND_ESTIMATION
	W = sqrtf(powf(windActual->North, 2) + powf(windActual->East, 2));
	psi_wind = atan2f(windActual->East, windActual->North);
#else
	W = 0;
	psi_wind = 0;
#endif

	float Phi = atan2f(p[1]-c[1], p[0]-c[0]);
	float psi_d = Phi + lambda * PI/2.0f;
	float psi_tilde = circular_modulus_rad(psi - psi_d); // This is the difference between the vehicle heading and the path heading

	float d = sqrtf(powf(p[0]-c[0], 2) + powf(p[1]-c[1], 2));
	float d_min = bound_min_max(arc_radius * 0.50f, V*(V + W)/(GRAVITY * tanf(phi_max)), arc_radius); // Try for 50% of the arc radius

	float d_tilde = d - arc_radius;
	float d_tilde_dot = -lambda*V*sinf(psi_tilde)*cosf(gamma) + W*cosf(psi - psi_wind);

	float M4;
	if (d_min > 1e3)
		M4 = tanf(phi_max) - V*V/(d_min*GRAVITY)*cosf(gamma_max)*cosf(psi_tilde_thresh);
	else
		M4 = tanf(phi_max)*(1 - cosf(gamma_max)*cosf(psi_tilde_thresh));

	float M5 = 1/2.0f*M4*GRAVITY*fabsf(cosf(psi_tilde_thresh)*cosf(gamma_max) - W/V);

	float z2 = k4*d_tilde + d_tilde_dot;
	float zeta = k5*z2;

	// Test for navigation errors that come from limits being exceeded.
	if (W >= V * cosf(psi_tilde_thresh) * cosf(gamma_max))
		AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER, SYSTEMALARMS_ALARM_WARNING);
	else if	(d_min/arc_radius < 0.90f) // Check if the d_min is greater than 90% of arc_radius
		AlarmsSet(SYSTEMALARMS_ALARM_PATHFOLLOWER, SYSTEMALARMS_ALARM_WARNING);


	// Calculate roll command
	float roll_c_R;

	if (d < d_min)
		roll_c_R = 0;
	else if (lambda*psi_tilde <= -psi_tilde_thresh)
		roll_c_R =  lambda * phi_max;
	else if (lambda*psi_tilde >=  psi_tilde_thresh)
		roll_c_R = -lambda * phi_max;
	else {
		roll_c_R = atanf(lambda*V*V/(GRAVITY*d)*cosf(gamma)*cosf(psi_tilde) +
						 bound_sym((k4*d_tilde_dot + bound_sym(zeta, M5)) / (lambda*GRAVITY*cosf(psi_tilde)*cosf(gamma) +
																	 GRAVITY*W/V*sinf(psi - psi_wind)), M4));
	}

	return roll_c_R;
}
