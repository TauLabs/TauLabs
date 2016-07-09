/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       dare.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      Solve discrete riccati algebraic equations
 *
 * @see        The GNU Public License (GPL) Version 3
 *
 ******************************************************************************/
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

#ifndef _DARE_H

#include "Eigen/Dense"
#include "stdint.h"

using Eigen::Matrix;

#define NUMX 3
#define NUMU 1

typedef Matrix <float, NUMX, NUMX> MXX;
typedef Matrix <float, NUMX, NUMU> MXU;
typedef Matrix <float, NUMU, NUMX> MUX;
typedef Matrix <float, NUMU, NUMU> MUU;

MXX dare_solve(MXX A, MXU B, MXX Q, MUU R);
MUX lqr_gain_solve(MXX A, MXU B, MXX Q, MUU R);
MXU kalman_gain_solve(MXX A, MXU B, MXX Q, MUU R);

#endif /* _DARE_H */

/**
 * @}
 * @}
 */