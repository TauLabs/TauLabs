/**
 ******************************************************************************
 * @addtogroup TauLabsMath Tau Labs math support libraries
 * @{
 * @addtogroup TauLabsStatistics Tau Labs statistics math support libraries
 * @{
 *
 * @file       statistics.h
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
