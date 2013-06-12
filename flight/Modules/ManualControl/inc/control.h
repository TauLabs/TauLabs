/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Control Control Module
 * @{
 *
 * @file       manualcontrol.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Control module. Handles safety R/C link and flight mode.
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
#ifndef CONTROL_H
#define CONTROL_H

// The enum from here is used to determine the flight mode
#include "flightstatus.h"

enum control_events {
	CONTROL_EVENTS_NONE,
	CONTROL_EVENTS_ARM,
	CONTROL_EVENTS_ARMING,
	CONTROL_EVENTS_DISARM
};

#endif /* CONTROL_H */

/**
 * @}
 * @}
 */
