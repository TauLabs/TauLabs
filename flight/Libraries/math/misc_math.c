/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 * @addtogroup TauLabsMath Tau Labs math support libraries
 * @{
 *
 * @file       math_misc.c
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

#include <math.h>
#include "misc_math.h" 		/* API declarations */
#include "physical_constants.h"

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
 * Approximation an exponential scale curve
 * @param[in] x   input from [-1,1]
 * @param[in] g   sets the exponential amount [0,100]
 * @return  rescaled input
 */
float expo3(float x, int32_t g)
{
	return (x * ((100 - g) / 100.0f) + powf(x, 3) * (g / 100.0f));
}


/**
 * @}
 * @}
 */
