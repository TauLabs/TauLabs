/**
 ******************************************************************************
 * @addtogroup TauLabsMath Tau Labs math support libraries
 * @{
 * @addtogroup TauLabsStatistics Tau Labs statistics math support libraries
 * @{
 *
 * @file       statistics.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 * @brief      Statistics support
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

#include "statistics.h"

#include <math.h>


/**
 * @brief bayes_filter Implements a bayesian testing under the effect of an
 * actuator that can modify what is being sensed.
 *
 * Inspired by Probablistic Robotics, Chap. 2.4, "Bayes Filters"
 * @param belief_H0 The H0 hypothesis (also called the NULL HYPOTHESIS), e.g.
 * the vehicle has enough battery to return home.
 * @param belief_H1 The H1 hypothesis (also called the ALTERNATIVE HYPOTHESIS), e.g.
 * the vehicle does not have enough battery to return home.
 * @param p_sense_condition_H0 The probability of sensing the H0 condition, given the real condition
 * @param p_sense_condition_H1 The probability of sensing the H1 condition, given the real condition
 * @param p_H0_Ux_H0 The probability of state H0 staying H0 after the control is applied
 * @param p_H0_Ux_H1 The probability of state H1 becoming H0 after the control is applied
 * @param p_H1_Ux_H0 The probability of state H0 becoming H1 after the control is applied
 * @param p_H1_Ux_H1 The probability of state H1 staying H1 after the control is applied
 */
void bayes_filter(double *belief_H0, double *belief_H1,
                double p_sense_condition_H0, double p_sense_condition_H1,
                double p_H0_Ux_H0, double p_H0_Ux_H1,
                double p_H1_Ux_H0, double p_H1_Ux_H1)
{
	// --------------------//
	// Process the control //
	// ------------------- //
	// Helper variable to let the function read better
	double old_belief_H0 = *belief_H0;
	double old_belief_H1 = *belief_H1;

	// The median belief is the old belief modified by the likelihood that the
	// control successfully controls the measured state
	double belief_bar_H0 = p_H0_Ux_H0 * old_belief_H0 + p_H0_Ux_H1 * old_belief_H1;
	double belief_bar_H1 = p_H1_Ux_H0 * old_belief_H0 + p_H1_Ux_H1 * old_belief_H1;

	// -------------------//
	// Measurement update //
	// -------------------//
	// Helper variable to let the function read better
	double *new_belief_H0 = belief_H0;
	double *new_belief_H1 = belief_H1;

	// The new belief is the likelihood of the new state occurring times the
	// likelihood of measuring the new state
	*new_belief_H0 = p_sense_condition_H0 * belief_bar_H0;
	*new_belief_H1 = p_sense_condition_H1 * belief_bar_H1;


	// Renormalize by gamma. This keeps the sum of the two values at 1, i.e. 100%
	// of probable cases covered
	double gamma = 1/(*new_belief_H0 + *new_belief_H1);
	*new_belief_H0 *= gamma;
	*new_belief_H1 *= gamma;

}


/**
 * @brief initialize_linear_sums Because floating point values are used,
 * there will be some loss of precision. Thus, it is important to periodically reset the parameters.
 * Cf. https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 * Cf. http://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
 * @param X struct holding the statistical parameters
 * @param window_size desired window size
 * @param num_samples number of samples available in vector
 * @param y vector of samples
 */
void initialize_linear_sums(struct linear_mean_and_std_dev *X, uint16_t window_size, uint16_t num_samples, const float y[])
{
	X->window_size = window_size;
	X->T0 = num_samples;

	// Reinitialize the accumulators
	X->T1 = 0;
	X->T2 = 0;

	for (int i=0; i<num_samples; i++) {
		// Sum of elements
		X->T1 += (double)y[i];

		// Sum of squares of elements
		X->T2 += (double)y[i]*(double)y[i];
	}
}


/**
 * @brief incremental_update_linear_sums  Basically functions like a FIFO, but in the
 * sense that it is adding and removing cumulative data. NOTE: Because this uses floating point values and
 * there will be some loss of precision, it is expected that the parameters must be periodically reset.
 *
 * Cf. https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 * Cf. http://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
 * @param X struct holding the stochastic parameters
 * @param y_oldest The chronologically oldest element in the buffer.
 * @param y_new The chronologically newest element, to be added to the buffer.
 */
void incremental_update_linear_sums(struct linear_mean_and_std_dev *X,
      float y_oldest,
      const float y_new)
{
	// Grow T0, number of samples, until reaching the desired size of the sample set
	if (X->T0 < X->window_size) {
		X->T0++; //

		// Set this equal to 0 so that the earliest element isn't removed when
		// the series hasn't yet reached the window size
		y_oldest = 0;
	}

	// Update T1, sum of elements
	X->T1 += (double)(-y_oldest + y_new);

	// Update T2, sum of squares of elements
	X->T2 += (double)(-y_oldest*y_oldest) + (double)(y_new*y_new);
}


/**
 * @brief get_linear_mean Calculate the mean as the sum of the elements, divided by the
 * number of elements. Classic!
 * @param X struct holding the stochastic parameters
 * @return the mean for the dataset represented by the stochastic parameters
 */
float get_linear_mean(const struct linear_mean_and_std_dev *X)
{
	float mean = X->T1 / X->T0;
	return mean;
}


/**
 * @brief get_linear_standard_deviation This recognizes that the standard deviation for a dataset
 * can be calculated by a rolling sum of number of elements, sum of elements, and sum of
 * elements squared. The decision is made to use the slightly less correct std. deviation
 * calculation which divdes by N, instead of N-1. For large datasets, this makes only a
 * small scalar difference. This results in a slight gain of computational efficiency.
 *
 * Cf. https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 * Cf. http://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
 * @param X struct holding the stochastic parameters
 * @return std_dev the standard deviation for the dataset represented by the stochastic parameters
 */
float get_linear_standard_deviation(const struct linear_mean_and_std_dev *X)
{
	float std_dev;

	// sigma = sqrt(T0*T2 - T1^2)/T0
	double radicand = X->T0*X->T2 - X->T1*X->T1;

	// Check for sanity
	if (radicand > 0 && X->T0 > 0) {
		std_dev = sqrtf(radicand) / X->T0;
	} else {
		std_dev = 0;
	}
	return std_dev;
}


/**
 * @brief get_linear_variance This recognizes that the variance (standard deviation squared)
 * for a dataset can be calculated by a rolling sum of number of elements, sum of elements,
 * and sum of elements squared.
 *
 * Cf. https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 * Cf. http://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
 * @param X struct holding the stochastic parameters
 * @return std_dev the variance (standard deviation squared) for the dataset represented by the stochastic parameters
 */
float get_linear_variance(const struct linear_mean_and_std_dev *X)
{
	// sigma^2 = T2/T0 - (T1/T0)^2
	float variance = (X->T2*X->T0 - X->T1*X->T1) / (X->T0*X->T0);

	return variance;
}


/**
 * @brief pearson_correlation Calculates the Pearson correlation for a covariance between two
 * variables. This is the variance divided by the sqrt of the muliple of the autovariances
 * @param x_autovariance
 * @param y_autovariance
 * @param variance
 * @return
 */
float pearson_correlation(const float x_autovariance, const float y_autovariance, const float variance)
{
	// If either autovariance is 0 (or close to), then we don't know what the correlation is.
	// The other variable could be static or it could be varying. The result is either -Inf
	// or +Inf, depending on from which direction the first autovariance value is
	// approaching 0. Rather than try to calculate this, we simply return an impossible
	// value to indicate which
	if (x_autovariance < 1e-5f) {
		return -2;
	} else if (y_autovariance < 1e-5f) {
		return 2;
	}

	// Precalculate the denominator squared
	float den2 = x_autovariance * y_autovariance;

	// Check that the denominator hasn't suffered an underflow as the result of
	// multiplying the two autovariances together
	if (den2 > 1e-32f) {
		return variance / sqrtf(den2);
	} else {
		return 0;
	}
}


/**
 * @brief initialize_circular_sums Because floating point values are used,
 * there will be some loss of precision. Thus, it is important to periodically reset the parameters.
 *
 * Cf. https://en.wikipedia.org/wiki/Directional_statistics
 * @param X struct holding the stochastic parameters
 * @param num_samples number of samples
 * @param y vector of samples
 */
void initialize_circular_sums(struct circular_mean_and_std_dev *X, uint16_t window_size, uint16_t num_samples, const float y_R[])
{
	X->window_size = window_size;

	X->T0 = num_samples;
	X->T0_2 = num_samples * num_samples;

	// Reinitialize the accumulators
	X->S1 = 0;
	X->C1 = 0;

	for (int i=0; i<num_samples; i++) {
		// Sum of elements
		X->S1 += sin(/*(double)*/y_R[i]);
		X->C1 += cos(/*(double)*/y_R[i]);
	}
}


/**
 * @brief incremental_update_circular_sums  Basically functions like a FIFO, but in the
 * sense that it is adding and removing cumulative data. NOTE: Because this uses float point values and there
 * will be some loss of precision, it is expected that the parameters must be periodically reset.
 *
 * Cf. https://en.wikipedia.org/wiki/Directional_statistics
 * @param X struct holding the stochastic parameters
 * @param x_oldest The chronologically oldest element in the buffer.
 * @param x_new The chronologically newest element, to be added to the buffer.
 */
void incremental_update_circular_sums(struct circular_mean_and_std_dev *X, const float y_oldest, const float y_new)
{
	// Grow T0, number of samples, until reaching the desired size of the sample set
	if (X->T0 < X->window_size) {
		X->T0++; //
		X->T0_2 = X->T0*X->T0; // Squared number of samples

		// If the array isn't at full size yet, then simply add the new values to the sum
		X->S1 += sin(y_new);
		X->C1 += cos(y_new);
	} else {
		X->S1 += -sin(y_oldest) + sin(y_new);
		X->C1 += -cos(y_oldest) + cos(y_new);
	}
}


/**
 * @brief get_circular_mean This recognizes that the mean for a dataset of angles is calculated
 * by a rolling sum of sines and cosines of angles.
 *
 * Cf. https://en.wikipedia.org/wiki/Directional_statistics
 * @param X struct holding the stochastic parameters
 * @return the mean for the dataset represented by the stochastic parameters
 */
float get_circular_mean(const struct circular_mean_and_std_dev *X)
{
	float circular_mean = atan2f(X->S1, X->C1);

	return circular_mean;
}


/**
 * @brief get_circular_standard_deviation This recognizes that the standard deviation for a dataset
 * can be calculated by a rolling sum of number of data points and sum of sines and cosines of angles.
 *
 * Cf. https://en.wikipedia.org/wiki/Directional_statistics
 * @param X struct holding the stochastic parameters
 * @return circular_std_dev the standard deviation for the dataset represented by the stochastic parameters
 */
float get_circular_standard_deviation(const struct circular_mean_and_std_dev *X)
{
	float circular_std_dev;

	// Calculate R^2, the norm of the average vector
	double R2 = (X->S1*X->S1 + X->C1*X->C1) / X->T0_2;

	if (R2 <= 1) {
		circular_std_dev = sqrtf(-log(R2));
	} else {
		circular_std_dev = 0;
	}

	return circular_std_dev;
}


/**
 * @brief get_angular_deviation This recognizes that the angular deviation for a dataset
 * can be calculated by a rolling sum of number of data points and sum of sines and cosines of angles.
 *
 * Cf. "CircStat: A MATLAB Toolbox for Circular Statistics", http://www.kyb.tuebingen.mpg.de/fileadmin/user_upload/files/publications/J-Stat-Softw-2009-Berens_6037[0].pdf
 * @param X struct holding the stochastic parameters
 * @return angular_deviation the standard deviation for the dataset represented by the stochastic parameters
 */
float get_angular_deviation(const struct circular_mean_and_std_dev *X)
{
	float angular_deviation;

	// Calculate R^2, the norm of the average vector
	double R2 = (X->S1*X->S1 + X->C1*X->C1) / X->T0_2;

	if (R2 <= 1) {
		angular_deviation = sqrtf(2*(1 - sqrt(R2)));
	} else {
		angular_deviation = 0;
	}

	return angular_deviation;
}
