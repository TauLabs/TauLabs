/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 * @addtogroup TauLabsMath Tau Labs math support libraries
 * @{
 *
 * @file       misc_math.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Miscellaneous math support
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

#include "misc_math.h" 		/* API declarations */
#include "physical_constants.h"
#include "math.h"

/**
 * Bound input value between min and max
 */
float bound_min_max(float val, float min, float max)
{
	if (val < min)
		return min;
	if (val > max)
		return max;
	return val;
}

/**
 * Bound input value within range (plus or minus)
 */
float bound_sym(float val, float range)
{
	return (bound_min_max(val, -range, range));
}


/**
 * Circular modulus [degrees].  Compute the equivalent angle between [-180,180]
 * for the input angle.  This is useful taking the difference between
 * two headings and working out the relative rotation to get there quickest.
 * @param[in] err input value in degrees.
 * @returns The equivalent angle between -180 and 180
 */
float circular_modulus_deg(float err)
{
	float val = fmodf(err + 180.0f, 360.0f);

	// fmodf converts negative values into the negative remainder
	// so we must add 360 to make sure this ends up correct and
	// behaves like positive output modulus
	if (val < 0)
		val += 180;
	else
		val -= 180;

	return val;

}


/**
 * Circular modulus [radians].  Compute the equivalent angle between [-pi,pi]
 * for the input angle.  This is useful taking the difference between
 * two headings and working out the relative rotation to get there quickest.
 * @param[in] err input value in radians.
 * @returns The equivalent angle between -pi and pi
 */
float circular_modulus_rad(float err)
{
	float val = fmodf(err + PI, 2*PI);

	// fmodf converts negative values into the negative remainder
	// so we must add 360 to make sure this ends up correct and
	// behaves like positive output modulus
	if (val < 0)
		val += PI;
	else
		val -= PI;

	return val;

}


/**
 * @brief Compute the center of curvature of the arc, by calculating the intersection
 * of the two circles of radius R around the two points. Inspired by
 * http://www.mathworks.com/matlabcentral/newsreader/view_thread/255121
 * @param[in] start_point Starting point, in North-East coordinates
 * @param[in] end_point Ending point, in North-East coordinates
 * @param[in] radius Radius of the curve segment
 * @param[in] clockwise true if clockwise is the positive sense of the arc, false if otherwise
 * @param[in] minor true if minor arc, false if major arc
 * @param[out] center Center of circle formed by two points, in North-East coordinates
 * @return
 */
enum arc_center_results find_arc_center(float start_point[2], float end_point[2], float radius, bool clockwise, bool minor, float center[2])
{
	// Sanity check
	if(fabsf(start_point[0] - end_point[0]) < 1e-6f && fabsf(start_point[1] - end_point[1]) < 1e-6f){
		// This means that the start point and end point are directly on top of each other. In the
		// case of coincident points, there is not enough information to define the circle
		center[0]=NAN;
		center[1]=NAN;
		return ARC_COINCIDENT_POINTS;
	}

	float m_n, m_e, p_n, p_e, d, d2;

	// Center between start and end
	m_n = (start_point[0] + end_point[0]) / 2;
	m_e = (start_point[1] + end_point[1]) / 2;

	// Normal vector to the line between start and end points
	if ((clockwise == true && minor == true) ||
			(clockwise == false && minor == false)) { //clockwise minor OR counterclockwise major
		p_n = -(end_point[1] - start_point[1]);
		p_e =  (end_point[0] - start_point[0]);
	} else { //counterclockwise minor OR clockwise major
		p_n =  (end_point[1] - start_point[1]);
		p_e = -(end_point[0] - start_point[0]);
	}

	// Work out how far to go along the perpendicular bisector. First check there is a solution.
	d2 = radius*radius / (p_n*p_n + p_e*p_e) - 0.25f;
	if (d2 < 0) {
		if (d2 > -powf(radius*0.01f, 2)) // Make a 1% allowance for roundoff error
			d2 = 0;
		else {
			center[0]=NAN;
			center[1]=NAN;
			return ARC_INSUFFICIENT_RADIUS; // In this case, the radius wasn't big enough to connect the two points
		}
	}

	d = sqrtf(d2);

	if (fabsf(p_n) < 1e-3f && fabsf(p_e) < 1e-3f) {
		center[0] = m_n;
		center[1] = m_e;
	} else {
		center[0] = m_n + p_n * d;
		center[1] = m_e + p_e * d;
	}

	return ARC_CENTER_FOUND;
}


/**
 * @brief measure_arc_rad Measure angle between two points on a circular arc
 * @param oldPosition_NE
 * @param newPosition_NE
 * @param arcCenter_NE
 * @return theta The angle between the two points on the circluar arc
 */
float measure_arc_rad(float oldPosition_NE[2], float newPosition_NE[2], float arcCenter_NE[2])
{
	float a[2] = {oldPosition_NE[0] - arcCenter_NE[0], oldPosition_NE[1] - arcCenter_NE[1]};
	float b[2] = {newPosition_NE[0] - arcCenter_NE[0], newPosition_NE[1] - arcCenter_NE[1]};

	float theta = angle_between_2d_vectors(a, b);
	return theta;
}


/**
 * @brief angle_between_2d_vectors Using simple vector calculus, calculate the angle between two 2D vectors
 * @param a
 * @param b
 * @return theta The angle between two vectors
 */
float angle_between_2d_vectors(float a[2], float b[2])
{
	// We cannot directly use the vector calculus formula for cos(theta) and sin(theta) because each
	// is only unique on half the circle. Instead, we combine the two because tangent is unique across
	// [-pi,pi]. Use the definition of the cross-product for 2-D vectors, a x b = |a||b| sin(theta), and
	// the definition of the dot product, a.b = |a||b| cos(theta), and divide the first by the second,
	// yielding a x b / (a.b) = sin(theta)/cos(theta) == tan(theta)
	float theta = atan2f(a[0]*b[1] - a[1]*b[0],(a[0]*b[0] + a[1]*b[1]));
	return theta;
}

/**
 * @}
 * @}
 */
