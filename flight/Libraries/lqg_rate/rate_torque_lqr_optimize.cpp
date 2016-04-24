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
#include "stdint.h"

#define CONVERGE_ITERATIONS 10000
#define CONVERGENCE_TOLERANCE 10.0f

void *__dso_handle = (void *)NULL;

using Eigen::MatrixXd;
using Eigen::Matrix;

#define NUMX 3
#define NUMU 1

typedef Matrix <float, NUMX, NUMX> MXX;
typedef Matrix <float, NUMX, NUMU> MXU;
typedef Matrix <float, NUMU, NUMX> MUX;
typedef Matrix <float, NUMU, NUMU> MUU;

static MXX A;
static MXU B;
static MXX Q;
static MUU R;

// one gain for roll and pitch and two for yaw
static float gains[4];
static float roll_pitch_cost;
static float yaw_cost;

// params for rate controller
static float integral_cost;
static float rate_cost;
static float torque_cost;

// this is for attitude controller
static float attitude_cost;
static float attitude_rate_cost;

static struct computed_gains {
	float roll_attitude_gains[3];
	float roll_rate_gains[3];
	float pitch_attitude_gains[3];
	float pitch_rate_gains[3];
	float yaw_attitude_gains[3];
	float yaw_rate_gains[3];
} computed_gains;

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

	Ts = new_Ts;

	A(0,1) = Ts;
}

/**
 * @brief rtlqr_set_tau set the time constant from system
 * identification
 * @param[in] tau the time constant ln(seconds)
 */
extern "C" void rtlqro_set_tau(float tau)
{
	float t1 = Ts/(expf(tau) + Ts);

	B(2,0) = t1;
}

/**
 * @brief rtlqr_set_gains the channel gains including two
 * for yaw
 * @param[in] gains to be used
 */
extern "C" void rtlqro_set_gains(const float new_gains[4])
{
	for (uint32_t i = 0; i < 4; i++)
		gains[i] = new_gains[i];
}

/**
 * @brief rtlqr_set_costs set the state and output costs for optimized LQR
 * @param[in] rate_error cost for static rate error
 * @param[in] torque_error cost for having static torque error
 * @param[in] integral_error cost for having accumulated error
 * @param[in] roll_pitch_input cost of using roll or pitch control
 * @param[in] yaw_input cost of using yaw control
 */
extern "C" void rtlqro_set_costs(float attitude_error,
	float attitude_rate_error,
	float rate_error,
	float torque_error,
	float integral_error,
	float roll_pitch_input,
	float yaw_input)
{
	roll_pitch_cost = roll_pitch_input;
	yaw_cost = yaw_input;

	attitude_cost = attitude_error;
	attitude_rate_cost = attitude_rate_error;

	rate_cost = rate_error;
	torque_cost = torque_error;
	integral_cost = integral_error;
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
MUX rtlqro_gains_calculate()
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

	X_1 = X;

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

	return K_dlqr;
}

static void rtlqro_solver_roll()
{
	MUX K_dlqr;

	// Set up dynamics with roll parameters
	A(1,2) = expf(gains[0])*Ts;
	B(1,0) = 0;
	R(0,0) = roll_pitch_cost;

	// Solve for the rate controller
	Q(0,0) = integral_cost;
	Q(1,1) = rate_cost;
	Q(2,2) = torque_cost;

	K_dlqr = rtlqro_gains_calculate();
	computed_gains.roll_rate_gains[0] = K_dlqr(0,1); // rate term
	computed_gains.roll_rate_gains[1] = K_dlqr(0,2); // torque term
	computed_gains.roll_rate_gains[2] = K_dlqr(0,0); // integral term

	// Solve for the attitude controller
	Q(0,0) = attitude_cost;
	Q(1,1) = attitude_rate_cost;
	Q(2,2) = torque_cost;

	K_dlqr = rtlqro_gains_calculate();
	computed_gains.roll_attitude_gains[0] = K_dlqr(0,0); // attitude term
	computed_gains.roll_attitude_gains[1] = K_dlqr(0,1); // rate term
	computed_gains.roll_attitude_gains[2] = K_dlqr(0,2); // torque term
}

static void rtlqro_solver_pitch()
{
	MUX K_dlqr;

	// Set up dynamics with roll parameters
	A(1,2) = expf(gains[1])*Ts;
	B(1,0) = 0;
	R(0,0) = roll_pitch_cost;

	// Solve for the rate controller
	Q(0,0) = integral_cost;
	Q(1,1) = rate_cost;
	Q(2,2) = torque_cost;

	K_dlqr = rtlqro_gains_calculate();
	computed_gains.pitch_rate_gains[0] = K_dlqr(0,1); // rate term
	computed_gains.pitch_rate_gains[1] = K_dlqr(0,2); // torque term
	computed_gains.pitch_rate_gains[2] = K_dlqr(0,0); // integral term

	// Solve for the attitude controller
	Q(0,0) = attitude_cost;
	Q(1,1) = attitude_rate_cost;
	Q(2,2) = torque_cost;

	K_dlqr = rtlqro_gains_calculate();
	computed_gains.pitch_attitude_gains[0] = K_dlqr(0,0); // attitude term
	computed_gains.pitch_attitude_gains[1] = K_dlqr(0,1); // rate term
	computed_gains.pitch_attitude_gains[2] = K_dlqr(0,2); // torque term
}

static void rtlqro_solver_yaw()
{
	MUX K_dlqr;

	// Set up dynamics with roll parameters
	A(1,2) = expf(gains[2])*Ts;
	B(1,0) = expf(gains[3])*Ts;
	R(0,0) = yaw_cost;

	// Solve for the rate controller
	Q(0,0) = integral_cost;
	Q(1,1) = rate_cost;
	Q(2,2) = torque_cost;

	K_dlqr = rtlqro_gains_calculate();
	computed_gains.yaw_rate_gains[0] = K_dlqr(0,1); // rate term
	computed_gains.yaw_rate_gains[1] = K_dlqr(0,2); // torque term
	computed_gains.yaw_rate_gains[2] = K_dlqr(0,0); // integral term

	// Solve for the attitude controller
	Q(0,0) = attitude_cost;
	Q(1,1) = attitude_rate_cost;
	Q(2,2) = torque_cost;

	K_dlqr = rtlqro_gains_calculate();
	computed_gains.yaw_attitude_gains[0] = K_dlqr(0,0); // attitude term
	computed_gains.yaw_attitude_gains[1] = K_dlqr(0,1); // rate term
	computed_gains.yaw_attitude_gains[2] = K_dlqr(0,2); // torque term
}

extern "C" void rtlqro_solver()
{
	rtlqro_solver_roll();
	rtlqro_solver_pitch();
	rtlqro_solver_yaw();
}

extern "C" void rtlqro_get_roll_rate_gain(float g[3])
{
	for (uint32_t i = 0; i < 3; i++)
		g[i] = computed_gains.roll_rate_gains[i];
}

extern "C" void rtlqro_get_pitch_rate_gain(float g[3])
{
	for (uint32_t i = 0; i < 3; i++)
		g[i] = computed_gains.pitch_rate_gains[i];
}

extern "C" void rtlqro_get_yaw_rate_gain(float g[3])
{
	for (uint32_t i = 0; i < 3; i++)
		g[i] = computed_gains.yaw_rate_gains[i];
}

extern "C" void rtlqro_get_roll_attitude_gain(float g[3])
{
	for (uint32_t i = 0; i < 3; i++)
		g[i] = computed_gains.roll_attitude_gains[i];
}

extern "C" void rtlqro_get_pitch_attitude_gain(float g[3])
{
	for (uint32_t i = 0; i < 3; i++)
		g[i] = computed_gains.pitch_attitude_gains[i];
}

extern "C" void rtlqro_get_yaw_attitude_gain(float g[3])
{
	for (uint32_t i = 0; i < 3; i++)
		g[i] = computed_gains.yaw_attitude_gains[i];
}

/**
 * @}
 * @}
 */