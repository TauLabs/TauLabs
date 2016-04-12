/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       rate_torque_lqr_optimize.c
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

#include "Eigen/Dense"

#define CONVERGE_ITERATIONS 10000
#define CONVERGENCE_TOLERANCE 10.0f

void *__dso_handle = (void *)NULL;

using Eigen::MatrixXd;
using Eigen::Matrix;

#define NUMX 9
#define NUMU 3

typedef Matrix <float, NUMX, NUMX> MXX;
typedef Matrix <float, NUMX, NUMU> MXU;
typedef Matrix <float, NUMU, NUMX> MUX;
typedef Matrix <float, NUMU, NUMU> MUU;

static MXX A;
static MXU B;
static MXX Q;
static MUU R;
static MUX K_dlqr;

// // Tracked for if we iteratively update the model
static Matrix <float, NUMX, NUMX> X_1;
static float Ts;

/**
 * @brief rtlqr_init prepare the solver
 * this still reqires the @ref set_tau and @ref set_gains
 * methods to be set at a minimum
 * @param new_Ts the time step this is called at (in seconds)
 */
extern "C" void rtlqro_init(float new_Ts)
{
	A = MXX::Identity();
	B = MXU::Constant(0.0f);
	Q = MXX::Identity();
	R = MUU::Identity();

	X_1 = MXX::Constant(0.0f);

	Ts = new_Ts;

	A(6,0) = Ts;
	A(7,1) = Ts;
	A(8,2) = Ts;
}

/**
 * @brief rtlqr_set_tau set the time constant from system
 * identification
 * @param[in] tau the time constant ln(seconds)
 */
extern "C" void rtlqro_set_tau(float tau)
{
	float t1 = Ts/(expf(tau) + Ts);

	B(3,0) = t1;
	B(4,1) = t1;
	B(5,2) = t1;
}

/**
 * @brief rtlqr_set_gains the channel gains including two
 * for yaw
 * @param[in] gains to be used
 */
extern "C" void rtlqro_set_gains(const float gain[4])
{
	A(0,3) = expf(gain[0])*Ts;
	A(1,4) = expf(gain[1])*Ts;
	A(2,5) = expf(gain[2])*Ts;

	B(2,2) = expf(gain[3])*Ts;
}

/**
 * @brief rtlqr_set_costs set the state and output costs for optimized LQR
 * @param[in] rate_error cost for static rate error
 * @param[in] torque_error cost for having static torque error
 * @param[in] integral_error cost for having accumulated error
 * @param[in] roll_pitch_input cost of using roll or pitch control
 * @param[in] yaw_input cost of using yaw control
 */
extern "C" void rtlqro_set_costs(float rate_error,
	float torque_error,
	float integral_error,
	float roll_pitch_input,
	float yaw_input)
{
	Q(0,0) = rate_error;
	Q(1,1) = rate_error;
	Q(2,2) = rate_error;
	Q(3,3) = torque_error;
	Q(4,4) = torque_error;
	Q(5,5) = torque_error;
	Q(6,6) = integral_error;
	Q(7,7) = integral_error;
	Q(8,8) = integral_error;
	R(0,0) = roll_pitch_input;
	R(1,1) = roll_pitch_input;
	R(2,2) = yaw_input;
}

/*
 Pseudoinverse code taken from:
 http://eigen.tuxfamily.org/bz/show_bug.cgi?id=257
*/
template<typename _Matrix_Type_>
bool pseudoInverse(const _Matrix_Type_ &a, _Matrix_Type_ &result, double epsilon = std::numeric_limits<typename _Matrix_Type_::Scalar>::epsilon())
{
  if(a.rows()<a.cols())
      return false;

  Eigen::JacobiSVD< _Matrix_Type_ > svd = a.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

  typename _Matrix_Type_::Scalar tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs().maxCoeff();
  
  result = svd.matrixV() * _Matrix_Type_( (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().
      array().inverse(), 0) ).asDiagonal() * svd.matrixU().adjoint();

  return true;
}

bool pseudoInverseUU(const MUU &a, MUU &result, double epsilon = std::numeric_limits<MUU::Scalar>::epsilon())
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


bool pseudoInverseXX(const MXX &a, MXX &result, double epsilon = std::numeric_limits<MXX::Scalar>::epsilon())
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
 * @brief rtlqr_solver calculate the LQR gain matrix for the above settings
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
extern "C" void rtlqro_solver()
{
	MXX X;
	
	// Steal the previous value
	X = X_1;

	MXX  X_inv;
	MXX  A_trnsp;
	MUX  B_trnsp;
	MXU  BR;
	MXX  BRB;
	MXX  tmp_inv;
	MUU  R_inv;

	// Precalculate Matrices
	A_trnsp = A.transpose();         //cvTranspose( A, A_trnsp );
	B_trnsp = B.transpose();         //cvTranspose( B, B_trnsp );
	pseudoInverseUU(R,R_inv);          //cvInvert( R, R_inv, CV_SVD_SYM );
	BR = B * R_inv;                  //cvMatMul( B, R_inv, BR );
	BRB = BR * B_trnsp;              //cvMatMul( BR, B_trnsp, BRB );

	// Calculate X_1, the seeding value for the riccati eqn. solution.
	pseudoInverseXX(BRB, tmp_inv);     //cvInvert( BRB, tmp_inv, CV_SVD );
	tmp_inv = A_trnsp * tmp_inv;     //cvMatMul( A_trnsp, tmp_inv, tmp_inv );
	tmp_inv = Q + (tmp_inv * A);     //cvMatMulAdd( tmp_inv, A, Q, tmp_inv );
	X = tmp_inv;

	// Calculate X_n, the convergent value for the riccati eqn. solution.
	for (int i=0; i<CONVERGE_ITERATIONS; i++)
	{
		pseudoInverseXX(X, X_inv);            //cvInvert(X, X_inv, CV_SVD);
		tmp_inv = BRB + X_inv;              //cvAdd(X_inv,BRB,tmp_inv);
		pseudoInverseXX(tmp_inv, tmp_inv);    //cvInvert(tmp_inv, tmp_inv, CV_SVD);
		tmp_inv = A_trnsp * tmp_inv;        //cvMatMul( A_trnsp, tmp_inv, tmp_inv );
		X = Q + (tmp_inv * A);              //cvMatMulAdd( tmp_inv, A, Q, X );

		// early stopping
		if ( (X - X_1).squaredNorm() < CONVERGENCE_TOLERANCE) {
			break;
		}
		X_1 = X;
	}
	
	// Calculate K_dlqr
	tmp_inv = X * A;              //cvMatMul(X,A,tmp_inv);
	K_dlqr = B_trnsp * tmp_inv;   //cvMatMul(B_trnsp,tmp_inv,K_dlqr);
	BR = X * B;                   //cvMatMul(X,B,BR);
	R_inv = R + (B_trnsp * BR);   //cvMatMulAdd(B_trnsp,BR,R,R_inv);
	pseudoInverseUU(R_inv, R_inv);  //cvInvert(R_inv,R_inv, CV_SVD);
	K_dlqr = R_inv * K_dlqr;      //cvMatMul(R_inv,K_dlqr,K_dlqr);
}

extern "C" void rtlqro_get_roll_gain(float g[3])
{
	g[0] = K_dlqr(0,0);
	g[1] = K_dlqr(0,3);
	g[2] = K_dlqr(0,6);
}

extern "C" void rtlqro_get_pitch_gain(float g[3])
{
	g[0] = K_dlqr(1,1);
	g[1] = K_dlqr(1,4);
	g[2] = K_dlqr(1,7);
}

extern "C" void rtlqro_get_yaw_gain(float g[3])
{
	g[0] = K_dlqr(2,2);
	g[1] = K_dlqr(2,5);
	g[2] = K_dlqr(2,8);
}

/**
 * @}
 * @}
 */