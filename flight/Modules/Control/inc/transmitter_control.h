/**
 ******************************************************************************
 * @addtogroup Modules Tau Labs Modules
 * @{
 * @addtogroup Control Control Module
 * @{
 *
 * @file       transmitter_control.h
 * @author     Tau Labs, http://github.org/TauLabs Copyright (C) 2013.
 * @brief      Process transmitter inputs and use as control source
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

 #ifndef TRANSMITTER_CONTROL_H
 #define TRANSMITTER_CONTROL_H

enum control_selection {
	TRANMITTER_MISSING,
	TRANMITTER_PRESENT_AND_USED,
	TRANSMITTER_PRESENT_SELECT_TABLET
};

//! Initialize the transmitter control mode
int32_t transmitter_control_initialize();

//! Process inputs and arming
int32_t transmitter_control_update();

//! Select and use transmitter control
int32_t transmitter_control_select();

//! Choose the control source based on transmitter status
enum control_selection transmitter_control_selected_controller();

 #endif /* TRANSMITTER_CONTROL_H */
