/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Control Control Module
 * @{
 * @file       crash_commands.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Crash commands
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
#include "crash_commands.h"

#include "stabilizationdesired.h"

//! Sets the stabilization values in such a way that the model will gently crash
void nice_crash()
{
	// Pick default values that will roughly cause a plane to circle down
	// and a quad to fall straight down
	StabilizationDesiredData stabilization_desired;
	StabilizationDesiredGet(&stabilization_desired);
	stabilization_desired.Throttle = -1;
	stabilization_desired.Roll = -10;
	stabilization_desired.Pitch = 0;
	stabilization_desired.Yaw = -5;
	stabilization_desired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	stabilization_desired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
	stabilization_desired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK;
	StabilizationDesiredSet(&stabilization_desired);
}

//! Sets the stabilization values in such a way that the model will violently crash
void nasty_crash()
{
	// Pick default values that will cause a plane to spin down
	// and a quad to plummet
	StabilizationDesiredData stabilization_desired;
	StabilizationDesiredGet(&stabilization_desired);
	stabilization_desired.Throttle = -1;
	stabilization_desired.Roll = -1;
	stabilization_desired.Pitch = 1;
	stabilization_desired.Yaw = 1;
	stabilization_desired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] = STABILIZATIONDESIRED_STABILIZATIONMODE_NONE;
	stabilization_desired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_NONE;
	stabilization_desired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_NONE;
	StabilizationDesiredSet(&stabilization_desired);
}
