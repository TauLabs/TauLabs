/**
 ******************************************************************************
 * @file       math_misc.c
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @addtogroup OpenPilot Math Utilities
 * @{
 * @addtogroup MiscellaneousMath Math Various mathematical routines
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

#include <math.h>

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
	if(val < -range) {
		val = -range;
	} else if(val > range) {
		val = range;
	}
	return val;
}

/**
 * Circular modulus.  Compute the equivalent angle between [-180,180]
 * for an input from [-360,360].  This is useful taking the difference
 * between two headings and working out the relative rotation to get
 * there quickest.
 * @param[in] err error in degrees.  Must not be less than -540 degrees
 * @returns The equivalent angle between -180 and 180
 */
float circular_modulus_deg(float err)
{
	return fmodf(err + 360.0f + 180.0f, 360.0f) - 180.0f;
}

