/**
 ******************************************************************************
 * @addtogroup TauLabsMath Tau Labs math support libraries
 * @{
 * @addtogroup TauLabsStatistics Tau Labs statistics math support libraries
 * @{
 *
 * @file       statistics.h
 * @author     Kenn Sebesta, Copyright (C) 2015.
 * @brief      Statistics support
 *
 *****************************************************************************/

#ifndef STATISTICS_H
#define STATISTICS_H

#include "stdint.h"

// Public types
struct linear_mean_and_std_dev {
	uint16_t T0;
	double T1;
	double T2;
	uint16_t window_size;
};

/* All below functions inspired by
 * http://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
 */
void initialize_linear_sums(struct linear_mean_and_std_dev *X, uint16_t window_size, uint16_t num_samples, const float x[]);
void incremental_update_linear_sums(struct linear_mean_and_std_dev *X, float x_oldest, const float x_new);
float get_linear_mean(const struct linear_mean_and_std_dev *X);
float get_linear_standard_deviation(const struct linear_mean_and_std_dev *X);
float get_linear_variance(const struct linear_mean_and_std_dev *X);

#endif /* STATISTICS_H */
