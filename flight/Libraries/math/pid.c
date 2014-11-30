/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 * @addtogroup TauLabsMath Tau Labs math support libraries
 * @{
 *
 * @file       pid.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      PID Control algorithms
 *
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

#include "openpilot.h"
#include "physical_constants.h"
#include "misc_math.h"
#include "pid.h"

//! Store the shared time constant for the derivative cutoff.
static float deriv_tau = 7.9577e-3f;

//! Store the setpoint weight to apply for the derivative term
static float deriv_gamma = 1.0;

/**
 * Update the PID computation
 * @param[in] pid The PID struture which stores temporary information
 * @param[in] err The error term
 * @param[in] dT  The time step
 * @returns Output the computed controller value
 */
float pid_apply(struct pid *pid, const float err, float dT)
{	
	if (pid->i == 0) {
		// If Ki is zero, do not change the integrator. We do not reset to zero
		// because sometimes the accumulator term is set externally
	} else {
		// Scale up accumulator by 1000 while computing to avoid losing precision
		pid->iAccumulator += err * (pid->i * dT * 1000.0f);
		pid->iAccumulator = bound_sym(pid->iAccumulator, pid->iLim * 1000.0f);
	}

	// Calculate DT1 term
	float diff = (err - pid->lastErr);
	float dterm = 0;
	pid->lastErr = err;
	if(pid->d && dT)
	{
		dterm = pid->lastDer +  dT / ( dT + deriv_tau) * ((diff * pid->d / dT) - pid->lastDer);
		pid->lastDer = dterm;            //   ^ set constant to 1/(2*pi*f_cutoff)
	}	                                 //   7.9577e-3  means 20 Hz f_cutoff
 
	return ((err * pid->p) + pid->iAccumulator / 1000.0f + dterm);
}

/**
 * Update the PID computation and apply anti windup limit
 * @param[in] pid The PID struture which stores temporary information
 * @param[in] err The error term
 * @param[in] min_bound The minimum output
 * @param[in] max_bound The maximum output
 * @param[in] dT  The time step
 * @returns Output the computed controller value
 *
 * @Note based on "Feedback Systems" by Astrom and Murray, PID control
 *  chapter.
 */
float pid_apply_antiwindup(struct pid *pid, const float err,
	float min_bound, float max_bound, float dT)
{	
	if (pid->i == 0) {
		// If Ki is zero, do not change the integrator. We do not reset to zero
		// because sometimes the accumulator term is set externally
	} else {
		// Scale up accumulator by 1000 while computing to avoid losing precision
		pid->iAccumulator += err * (pid->i * dT * 1000.0f);
	}

	// Calculate DT1 term
	float diff = (err - pid->lastErr);
	float dterm = 0;
	pid->lastErr = err;
	if(pid->d && dT)
	{
		dterm = pid->lastDer +  dT / ( dT + deriv_tau) * ((diff * pid->d / dT) - pid->lastDer);
		pid->lastDer = dterm;            //   ^ set constant to 1/(2*pi*f_cutoff)
	}	                                 //   7.9577e-3  means 20 Hz f_cutoff
 
 	// Compute how much (if at all) the output is saturating
	float ideal_output = ((err * pid->p) + pid->iAccumulator / 1000.0f + dterm);
	float saturation = 0;
	if (ideal_output > max_bound) {
		saturation = max_bound - ideal_output;
		ideal_output = max_bound;
	} else if (ideal_output < min_bound) {
		saturation = min_bound - ideal_output;
		ideal_output = min_bound;
	}
	// Use Kt 10x Ki
	pid->iAccumulator += saturation * (pid->i * 10.0f * dT * 1000.0f);
	pid->iAccumulator = bound_sym(pid->iAccumulator, pid->iLim * 1000.0f);

	return ideal_output;
}

/**
 * Update the PID computation with setpoint weighting on the derivative
 * @param[in] pid The PID struture which stores temporary information
 * @param[in] setpoint The setpoint to use
 * @param[in] measured The measured value of output
 * @param[in] dT  The time step
 * @returns Output the computed controller value
 *
 * This version of apply uses setpoint weighting for the derivative component so the gain
 * on the gyro derivative can be different than the gain on the setpoint derivative
 */
float pid_apply_setpoint(struct pid *pid, const float setpoint, const float measured, float dT)
{
	float err = setpoint - measured;
	
	if (pid->i == 0) {
		// If Ki is zero, do not change the integrator. We do not reset to zero
		// because sometimes the accumulator term is set externally
	} else {
		// Scale up accumulator by 1000 while computing to avoid losing precision
		pid->iAccumulator += err * (pid->i * dT * 1000.0f);
		pid->iAccumulator = bound_sym(pid->iAccumulator, pid->iLim * 1000.0f);
	}

	// Calculate DT1 term,
	float dterm = 0;
	float diff = ((deriv_gamma * setpoint - measured) - pid->lastErr);
	pid->lastErr = (deriv_gamma * setpoint - measured);
	if(pid->d && dT)
	{
		dterm = pid->lastDer +  dT / ( dT + deriv_tau) * ((diff * pid->d / dT) - pid->lastDer);
		pid->lastDer = dterm;            //   ^ set constant to 1/(2*pi*f_cutoff)
	}	                                 //   7.9577e-3  means 20 Hz f_cutoff
 
	return ((err * pid->p) + pid->iAccumulator / 1000.0f + dterm);
}

/**
 * Reset a bit
 * @param[in] pid The pid to reset
 */
void pid_zero(struct pid *pid)
{
	if (!pid)
		return;

	pid->iAccumulator = 0;
	pid->lastErr = 0;
	pid->lastDer = 0;
}

/**
 * @brief Configure the common terms that alter ther derivative
 * @param[in] cutoff The cutoff frequency (in Hz)
 * @param[in] gamma The gamma term for setpoint shaping (unsused now)
 */
void pid_configure_derivative(float cutoff, float g)
{
	deriv_tau = 1.0f / (2 * PI * cutoff);
	deriv_gamma = g;
}

/**
 * Configure the settings for a pid structure
 * @param[out] pid The PID structure to configure
 * @param[in] p The proportional term
 * @param[in] i The integral term
 * @param[in] d The derivative term
 */
void pid_configure(struct pid *pid, float p, float i, float d, float iLim)
{
	if (!pid)
		return;

	pid->p = p;
	pid->i = i;
	pid->d = d;
	pid->iLim = iLim;
}

/**
 * @}
 * @}
 */
