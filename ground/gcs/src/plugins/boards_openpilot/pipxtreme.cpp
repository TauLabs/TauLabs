/**
 ******************************************************************************
 *
 * @file       pipxtreme.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

#include "pipxtreme.h"

/**
 * @brief PipXtreme::PipXtreme
 *  This is the PipXtreme radio modem definition
 */
PipXtreme::PipXtreme(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415c;

    setUSBInfo(board);

    boardType = 0x03;
}

PipXtreme::~PipXtreme()
{

}


QString PipXtreme::shortName()
{
    return QString("PipXtreme");
}

QString PipXtreme::boardDescription()
{
    return QString("The OpenPilot project PipXtreme RF radio modem");
}

//! Return which capabilities this board has
bool PipXtreme::queryCapabilities(BoardCapabilities capability)
{
    switch(capability) {
    case BOARD_CAPABILITIES_GYROS:
        return false;
    case BOARD_CAPABILITIES_ACCELS:
        return false;
    case BOARD_CAPABILITIES_MAGS:
        return false;
    case BOARD_CAPABILITIES_BAROS:
        return false;
    case BOARD_CAPABILITIES_RADIO:
        return true;
    }
    return false;
}

/**
 * @brief PipXtreme::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 *  TODO: for the PipXtreme, depending on its configuration, it might offer several protocols (uavtalk and raw)
 * @return
 */
QStringList PipXtreme::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap PipXtreme::getBoardPicture()
{
    return QPixmap(":/images/pipx.png");
}

QString PipXtreme::getHwUAVO()
{
    return "PipXtreme";
}
