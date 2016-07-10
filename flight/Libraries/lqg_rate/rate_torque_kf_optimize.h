/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       rate_torque_kf_optimize.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      Calculate static kalman gains using system properties measured
 *             by system identification
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

void rtkfo_init(float new_Ts);
void rtkfo_set_tau(float tau);
void rtkfo_set_gains(const float gain[4]);
void rtkfo_set_noise(float *q, float g);
void rtkfo_solver();
void rtkfo_get_roll_gain(float g[3]);
void rtkfo_get_pitch_gain(float g[3]);
void rtkfo_get_yaw_gain(float g[3]);

/**
 * @}
 * @}
 */