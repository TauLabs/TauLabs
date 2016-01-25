/**
 ******************************************************************************
 * @addtogroup TauLabsMath Tau Labs math support libraries
 * @{
 * @addtogroup TauLabsStatistics Tau Labs statistics math support libraries
 * @{
 *
 * @file       statistics.c
 * @author     Kenn Sebesta, Copyright (C) 2015.
 * @brief      Statistics support
 *
 *****************************************************************************/

#include "statistics.h"

#include <math.h>

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
