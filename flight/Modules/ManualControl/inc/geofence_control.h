/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Control Control Module
 * @{
 *
 * @file       geofence_control.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Geofence controller that is enabled when out of zone
 *
 * Currently this implements a simple controller that disables motors
 * whenever outside of the zone
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

 #ifndef GEOFENCE_CONTROL_H
 #define GEOFENCE_CONTROL_H

//! Initialize the geofence controller
int32_t geofence_control_initialize();

//! Perform any updates to the geofence controller
int32_t geofence_control_update();

//! Use geofence control mode
int32_t geofence_control_select(bool reset_controller);

//! Query if out of bounds and the geofence controller should take over
bool geofence_control_activate();

//! Get any control events
enum control_events geofence_control_get_events();

 #endif /* GEOFENCE_CONTROL_H */

/**
 * @}
 * @}
 */
