/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 * @addtogroup TauLabsMath Tau Labs math support libraries
 * @{
 *
 * @file       math_misc.h
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

#ifndef MISC_MATH_H
#define MISC_MATH_H

#include "stdint.h"

// Max/Min macros. Taken from http://stackoverflow.com/questions/3437404/min-and-max-in-c
#define MAX(a, b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })
#define MIN(a, b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

//! This is but one definition of sign(.)
#define sign(x) (x < 0 ? -1 : 1)

//! Bound input value within range (plus or minus)
float bound_sym(float val, float range);

//! Bound input value between min and max
float bound_min_max(float val, float min, float max);

//! Circular modulus
float circular_modulus_deg(float err);
float circular_modulus_rad(float err);

//! Approximation an exponential scale curve
float expo3(float x, int32_t g);

#endif /* MISC_MATH_H */

/**
 * @}
 * @}
 */
