/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       rate_torque_kf.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      Kalman filter to estimate rate and torque of motors
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

#ifndef RATE_TORQUE_KF_H
#define RATE_TORQUE_KF_H

#include "stdbool.h"

// Methods to configure RTKF parameters
void rtkf_set_qw(const float qw_new);
void rtkf_set_qu(const float qu_new);
void rtkf_set_qbias(const float qbias_new);
void rtkf_set_sa(const float sa_new);
void rtkf_set_gains(const float gains_new[3]);
void rtkf_set_tau(const float tau_new);

void rtkf_get_rate(const float *X, float rate[3]);
void rtkf_get_torque(const float *X, float torque[3]);
void rtkf_get_bias(const float *X, float bias[3]);
void rtkf_predict(float *X, float *P, const float u_in[3], const float gyro[3], const float dT_s);
bool rtkf_init(float **X_in, float **P_in);

#endif /* RATE_TORQUE_KF_H */

/**
 * @}
 * @}
 */
 