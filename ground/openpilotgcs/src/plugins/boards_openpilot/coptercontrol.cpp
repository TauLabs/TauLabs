/**
 ******************************************************************************
 *
 * @file       coptercontrol.cpp
 * @author     The PhoenixPilot Team, http://github.com/PhoenixPilot Copyright (C) 2012.
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_OpenPilotPlugin OpenPilot boards support Plugin
 * @{
 * @brief Plugin to support boards by the OP project
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

#include "coptercontrol.h"

/**
 * @brief CopterControl::CopterControl
 *  This is the CopterControl (3D) board definition
 */
CopterControl::CopterControl(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

}

CopterControl::~CopterControl()
{

}


QString CopterControl::shortName()
{
    return QString("CopterControl");
}

QString CopterControl::boardDescription()
{
    return QString("The OpenPilot project CopterControl and CopterControl 3D boards");
}

/**
 * @brief CopterControl::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList CopterControl::getSupportedProtocols()
{

    return QStringList("uavtalk");
}
