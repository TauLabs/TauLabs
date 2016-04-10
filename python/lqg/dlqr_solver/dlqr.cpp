/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       dlqr.c
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

#include "rate_torque_lqr_optimize.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using std::cout;

extern MatrixXd K_dlqr;

int main()
{
	static float g[4] = {9.67f, 9.84f, 5.2f, 8.29f};

	rtlqro_init(1.0f/400.0f);
	rtlqro_set_tau(-3.39f);
	rtlqro_set_gains(g);
	rtlqro_set_costs(10, 1, 10000, 1e4, 1e5);

	rtlqro_solver();

	cout << "K_dlqr" << std::endl << K_dlqr << std::endl;
}

/**
 * @}
 * @}
 */