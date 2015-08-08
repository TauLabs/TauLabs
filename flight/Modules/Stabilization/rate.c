/**
******************************************************************************
* @addtogroup TauLabsModules Tau Labs Modules
* @{
* @addtogroup StabilizationModule Stabilization Module
* @{
*
* @file       rate.c
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
* @author     AJ Christensen <aj@junglistheavy.industries>
* @brief      Rate mode
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

#include "pid.h"
#include "stabilization.h"
#include "stabilizationsettings.h"
#include "misc_math.h"

int stabilization_rate(float gyro,
                       float command,
                       float *output,
                       float dT,
                       bool reinit,
                       uint32_t axis,
                       struct pid *pid,
                       const StabilizationSettingsData *settings)
{
    if(reinit)
        pid->iAccumulator = 0;
    command = bound_sym(command, settings->ManualRate[axis]);
    // Compute the inner loop
    *output = pid_apply_setpoint(pid, command, gyro, dT);
    *output = bound_sym(*output, 1.0f);
    return 0;
}
