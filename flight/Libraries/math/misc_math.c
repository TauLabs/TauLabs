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

#include <stdbool.h>
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
 *
 * This function is also used for the expo plot in GCS (config/stabilization/advanced).
 * Please adapt changes here also to /ground/gcs/src/plugins/config/expocurve.cpp
 */
float expo3(float x, int32_t g)
{
	return (x * ((100 - g) / 100.0f) + powf(x, 3) * (g / 100.0f));
}

/**
 * Interpolate values (groundspeeds, altitudes) over flight legs
 * @param[in] fraction how far we are through the leg
 * @param[in] beginVal the configured value for the beginning of the leg
 * @param[in] endVal the configured value for the end of the leg
 * @returns the interpolated value
 *
 * Simple linear interpolation with clipping to ends (fraction>=0, <=1).
 */
float interpolate_value(const float fraction, const float beginVal,
	const float endVal) {
	return beginVal + (endVal - beginVal) * bound_min_max(fraction, 0, 1);
}

/**
 * Calculate pythagorean magnitude of a vector.
 * @param[in] v pointer to a float array
 * @param[in] n length of the amount to take the magnitude of
 * @returns Root of sum of squares of the vector
 *
 * Note that sometimes we only take the magnitude of the first 2 elements
 * of a 3-vector to get the horizontal component.
 */
float vectorn_magnitude(const float *v, int n)
{
	float sum=0;

	for (int i=0; i<n; i++) {
		sum += powf(v[i], 2);
	}

	return sqrtf(sum);
}

/**
 * Subtract two 3-vectors, and optionally normalize to return an error value.
 * @param[in] actual the measured process value
 * @param[in] desired the setpoint of the system
 * @param[out] the resultant error vector desired-actual
 * @param[in] normalize True if it's desired to normalize the output vector
 * @returns the norm, for fun.
 */
float vector3_distances(const float *actual,
	const float *desired, float *out, bool normalize)
{
	out[0] = desired[0] - actual[0];
	out[1] = desired[1] - actual[1];
	out[2] = desired[2] - actual[2];

	if (normalize) {
		float mag=vectorn_magnitude(out, 3);

		if (mag < 0.00001f) {
			/* Just pick a direction. */
			out[0] = 1.0; out[1] = 0.0; out[2] = 0.0;
		} else {
			out[0] /= mag;  out[1] /= mag;  out[2] /= mag;
		}

		return mag;
	}

	return 0.0;
}

/**
 * Clip a velocity 2-vector while maintaining vector direction.
 * @param[in,out] vels velocity to clip
 * @param[in] limit desired limit magnitude.
 *
 * if mag(vels) > limit, vels=vels / mag(vels) * limit
 */
void vector2_clip(float *vels, float limit)
{
	float mag = vectorn_magnitude(vels, 2);    // only horiz component

	if (mag > limit && mag != 0) {
		float scale = limit / mag;
		vels[0] *= scale;
		vels[1] *= scale;
	}
}

/**
 * Rotate a 2-vector by a specified angle.
 * @param[in] original the input vector
 * @param[out] out the rotated output vector
 * @param[in] angle in degrees
 */
void vector2_rotate(const float *original, float *out, float angle) {
	angle *= DEG2RAD;

	out[0] = original[0] * cosf(angle) - original[1] * sinf(angle);
	out[1] = original[0] * sinf(angle) + original[1] * cosf(angle);
}

/**
 * Apply a "cubic deadband" to the input.
 * @param[in] in the value to deadband
 * @param[in] w deadband width
 * @param[in] b slope of deadband at in=0
 * @param[in] m cubic weighting calculated by vtol_deadband_setup
 * @param[in] r integrated response at in=w, calculated by vtol_deadband_setup
 *
 * "Real" deadbands are evil.  Control systems end up fighting the edge.
 * You don't teach your integrator about emerging drift.  Discontinuities
 * in your control inputs cause all kinds of neat stuff.  As a result this
 * calculates a cubic function within the deadband which has a low slope
 * within the middle, but unity slope at the edge.
 */
float cubic_deadband(float in, float w, float b, float m, float r)
{
	// First get the nice linear bits -- outside the deadband-- out of
	// the way.
	if (in <= -w) {
		return in+w-r;
	} else if (in >= w) {
		return in-w+r;
	}

	return m * powf(in, 3) + b * in;
}

/**
 * Calculate the "cubic deadband" system parameters.
 * @param[in] The width of the deadband
 * @param[in] Slope of deadband at in=0, sane values between 0 and 1.
 * @param[out] m cubic weighting of function
 * @param[out] integrated response at in=w
 */
void cubic_deadband_setup(float w, float b, float *m, float *r)
{
	/* So basically.. we want the function to be tangent to the
	 ** linear sections-- have a slope of 1-- at -w and w.  In the
	 ** middle we want a slope of b.   So the cube here does all the
	 ** work b isn't doing. */
	*m = cbrtf((1-b)/(3*powf(w,2)));

	*r = *m * powf(w, 3) + b * w;
}


/**
 * @}
 * @}
 */
