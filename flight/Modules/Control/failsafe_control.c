/**
 ******************************************************************************
 * @addtogroup Modules Tau Labs Modules
 * @{
 * @addtogroup Control Control Module
 * @{
 *
 * @file       failsafe_control.c
 * @author     Tau Labs, http://github.org/TauLabs Copyright (C) 2013.
 * @brief      Failsafe controller when transmitter control is lost
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
#include "failsafe_control.h"

#include "flightstatus.h"
#include "stabilizationdesired.h"

//! Initialize the failsafe controller
int32_t failsafe_control_initialize()
{
	return 0;
}

//! Perform any updates to the failsafe controller
int32_t failsafe_control_update()
{
	return 0;
}

//! Use failsafe mode
int32_t failsafe_control_select()
{
	FlightStatusData flight_status;
	FlightStatusGet(&flight_status);
	flight_status.Armed = FLIGHTSTATUS_ARMED_DISARMED;
	flight_status.FlightMode = FLIGHTSTATUS_FLIGHTMODE_STABILIZED1;
	FlightStatusSet(&flight_status);

	StabilizationDesiredData stabilization_desired;
	StabilizationDesiredGet(&stabilization_desired);
	stabilization_desired.Throttle = -1;
	stabilization_desired.Roll = 0;
	stabilization_desired.Pitch = 0;
	stabilization_desired.Yaw = 0;
	StabilizationDesiredSet(&stabilization_desired);

	return 0;
}
