/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Control Control Module
 * @{
 *
 * @file       failsafe_control.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

 #ifndef FAILSAFE_CONTROL_H
 #define FAILSAFE_CONTROL_H

//! Initialize the failsafe controller
int32_t failsafe_control_initialize();

//! Perform any updates to the failsafe controller
int32_t failsafe_control_update();

//! Use failsafe mode
int32_t failsafe_control_select(bool reset_controller);

//! Get any control events
enum control_events failsafe_control_get_events();

 #endif /* FAILSAFE_CONTROL_H */

/**
 * @}
 * @}
 */
