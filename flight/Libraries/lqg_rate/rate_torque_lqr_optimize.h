/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       rate_torque_lqr_optimize.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      Optimize discrete time LQR controller gain matrix using
 *             system properties measured by system identification
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

void rtlqro_init(float new_Ts);
void rtlqro_set_tau(float tau);
void rtlqro_set_gains(const float gain[4]);
void rtlqro_set_costs(float rate_error,
	float torque_error,
	float integral_error,
	float roll_pitch_input,
	float yaw_input);
void rtlqro_solver();
void rtlqro_get_roll_gain(float g[3]);
void rtlqro_get_pitch_gain(float g[3]);
evoid rtlqro_get_yaw_gain(float g[3]);

/**
 * @}
 * @}
 */