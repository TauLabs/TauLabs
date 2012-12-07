/**
 ******************************************************************************
 * @file       visualization.h
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @addtogroup OpenPilotModules OpenPilot Modules
 * @{
 * @addtogroup Visualization
 * @{
 * @brief Sends the state of the UAV out a UDP port to be visualized in a
 * standalone 
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
#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "openpilot.h"

int32_t VisualizationInitialize(void);

#endif // VISUALIZATION_H
