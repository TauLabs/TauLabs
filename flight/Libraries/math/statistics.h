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

struct circular_mean_and_std_dev {
	uint16_t T0;
	uint32_t T0_2;
	double S1;
	double C1;
	uint16_t window_size;
};

void bayes_filter(double *belief_H0, double *belief_H1,
                double p_sense_condition_H0, double p_sense_condition_H1,
                double p_H0_Ux_H0, double p_H0_Ux_H1,
                double p_H1_Ux_H0, double p_H1_Ux_H1);

/* All below functions inspired by
 * http://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
 */
void initialize_linear_sums(struct linear_mean_and_std_dev *X, uint16_t window_size, uint16_t num_samples, const float x[]);
void incremental_update_linear_sums(struct linear_mean_and_std_dev *X, float x_oldest, const float x_new);
float get_linear_mean(const struct linear_mean_and_std_dev *X);
float get_linear_standard_deviation(const struct linear_mean_and_std_dev *X);
float get_linear_variance(const struct linear_mean_and_std_dev *X);
float pearson_correlation(const float x_autovariance, const float y_autovariance, const float variance);

/* All four below functions inspired by
 * https://en.wikipedia.org/wiki/Directional_statistics
 */
void initialize_circular_sums(struct circular_mean_and_std_dev *X, uint16_t window_size, uint16_t num_samples, const float x[]);
void incremental_update_circular_sums(struct circular_mean_and_std_dev *X, const float x_oldest, const float x_new);
float get_circular_mean(const struct circular_mean_and_std_dev *X);
float get_circular_standard_deviation(const struct circular_mean_and_std_dev *X);

/* Inspired by "CircStat: A MATLAB Toolbox for Circular Statistics",
 * http://www.kyb.tuebingen.mpg.de/fileadmin/user_upload/files/publications/J-Stat-Softw-2009-Berens_6037[0].pdf
 */
float get_angular_deviation(const struct circular_mean_and_std_dev *X);

#endif /* STATISTICS_H */
