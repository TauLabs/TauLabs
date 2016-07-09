/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       dare.c
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

#include "dare.h"

#define CONVERGE_ITERATIONS 10000
#define CONVERGENCE_TOLERANCE 0.0000001f

void *__dso_handle = (void *)NULL;

/*
 Pseudoinverse code taken from:
 http://eigen.tuxfamily.org/bz/show_bug.cgi?id=257
*/

bool pseudoInverseUU(const MUU &a, MUU &result, float epsilon = std::numeric_limits<MUU::Scalar>::epsilon())
{
	if(a.rows()<a.cols())
		return false;

	Eigen::JacobiSVD<MUU> svd = a.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

	MUU::Scalar tolerance = epsilon;
	tolerance *= (MUU::Scalar) std::max(a.cols(), a.rows());
	tolerance *= (MUU::Scalar) svd.singularValues().array().abs().maxCoeff();

	result = svd.matrixV() * Matrix<float,NUMU,1>( (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().
	         array().inverse(), 0) ).asDiagonal() * svd.matrixU().adjoint();

	return true;
}


bool pseudoInverseXX(const MXX &a, MXX &result, float epsilon = std::numeric_limits<MXX::Scalar>::epsilon())
{
	if(a.rows()<a.cols())
		return false;

	Eigen::JacobiSVD<MXX> svd = a.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

	MXX::Scalar tolerance = epsilon;
	tolerance *= (MXX::Scalar) std::max(a.cols(), a.rows());
	tolerance *= (MXX::Scalar) svd.singularValues().array().abs().maxCoeff();

	result = svd.matrixV() *  Matrix<float,NUMX,1>( (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().
	         array().inverse(), 0) ).asDiagonal() * svd.matrixU().adjoint();

	return true;
}


/**
 * @brief dare_solve solves teh discrete algebraic riccati equations
 * @param[in] A dynamic system matrix
 * @param[in] B control matrix
 * @param[in] Q process noise
 * @param[in] R observation/control noise
 * @param[out] K_dlqr the computed gain matrix
 *
 * The algorithm for solving the riccati eqn. comes from 
 * "NUMERICAL SOLUTION OF DISCRETE-TIME ALGEBRAIC RICCATI EQUATION" 
 * by Tan K. Nguyen
 *
 * adopted from an implementation written in CV by Kenn Sebestra that iterates:
 * X = Q+A_t*(B*R^-1*B_t)^-1*A 
 * for i=1:10000
 *     X=Q+A_t*(X^-1+B*R^-1*B_t)^-1*A
 */

MXX dare_solve(MXX A, MXU B, MXX Q, MUU R)
{
	MUX K_dlqr;

	MXX X, X_1;
	
	MXX  X_inv;
	MXX  A_trnsp;
	MUX  B_trnsp;
	MXU  BR;
	MXX  BRB;
	MXX  tmp_inv;
	MUU  R_inv;

	// Precalculate Matrices
	A_trnsp = A.transpose();
	B_trnsp = B.transpose();
	pseudoInverseUU(R,R_inv);
	BR = B * R_inv;
	BRB = BR * B_trnsp;

	// Calculate X_1, the seeding value for the riccati eqn. solution.
	pseudoInverseXX(BRB, tmp_inv);
	tmp_inv = A_trnsp * tmp_inv;
	tmp_inv = Q + (tmp_inv * A);
	X = tmp_inv;

	X_1 = X;

	// Calculate X_n, the convergent value for the riccati eqn. solution.
	for (int i=0; i<CONVERGE_ITERATIONS; i++)
	{
		pseudoInverseXX(X, X_inv);
		tmp_inv = BRB + X_inv;
		pseudoInverseXX(tmp_inv, tmp_inv);
		tmp_inv = A_trnsp * tmp_inv;
		X = Q + (tmp_inv * A);

		// early stopping
		if ( (X - X_1).squaredNorm() < CONVERGENCE_TOLERANCE) {
			break;
		}
		X_1 = X;
	}
	
	return X;
}

/**
 * Calculate the LQR feedback gains for a dynamical system
 * defined by A and B with cost matricies Q and R
 * @param[in] A the discrete time dynamical model of the system
 * @param[in] B the forcing matrix for the input
 * @param[in] Q the cost of state errors
 * @param[in] R the cost of control signals
 * @returns the calculated feedback gain
 */
MUX lqr_gain_solve(MXX A, MXU B, MXX Q, MUU R)
{
	// Get the Riccati solution
	MXX X = dare_solve(A,B,Q,R);

	MUU inv;
	pseudoInverseUU(R + B.transpose() * X * B, inv);
	MUX K = inv * (B.transpose() * X * A);

	return K;
}

/**
 * Calculate the steady state gain for a kalman filter this 
 * is for time invariant linear systems that have a covariance
 * that reaches a constant level
 * @param[in] A the discrete time dynamical model of the system
 * @param[in] B the forcing matrix for the input
 * @param[in] Q the process noise model
 * @param[in] R the observation noise
 * @returns the calculated feedback gain
 */
MXU kalman_gain_solve(MXX A, MXU B, MXX Q, MUU R)
{
	MUX C = MUX::Constant(0.0f);
	C(0,0) = 1;

	// Calculate intermediate steps required to compute kalman estimation
	MXX qe = MXX::Identity(); // This assumes Q is diagonal to save some calculations
	// Create a slightly different qe for the next calculation
	qe(0,0) = 1-(Q(0,0)/R(0,0))/(1+Q(0,0)/R(0,0));
	MXX fe = qe * A;

	qe(0,0) = Q(0,0)-(Q(0,0)*Q(0,0)/R(0,0))/(1+Q(0,0)/R(0,0));
	qe(1,1) = Q(1,1);
	qe(2,2) = Q(2,2);

	MXU he = C*A; //A.row(0); //C * A;
	MUU re = R + Q.block(0,0,1,1); //R + C * Q * C.transpose();

	MXX P = dare_solve(fe.transpose(),he.transpose(),qe,re);
	MXU L = P*C.transpose() / (P(0,0) + R(0,0)); //P*C.transpose() * pseudoInverseUU(C*P*C.transpose() + R);

	return L;
}
/**
 * @}
 * @}
 */