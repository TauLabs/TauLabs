/**
 ******************************************************************************
 *
 * @file       naze64.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_Stm STM boards support Plugin
 * @{
 * @brief Plugin to support boards by STM
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

#include "vrbrain.h"

/**
 * @brief Naze64::Naze64
 *  This is the Naze64 board definition
 */
Naze64::Naze64(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x41d0;

    setUSBInfo(board);

    boardType = 0x7a;
}

Naze64::~Naze64()
{

}

QString Naze64::shortName()
{
    return QString("Naze64");
}

QString Naze64::boardDescription()
{
    return QString("Naze64");
}

//! Return which capabilities this board has
bool Naze64::queryCapabilities(BoardCapabilities capability)
{
    switch(capability) {
    case BOARD_CAPABILITIES_GYROS:
        return true;
    case BOARD_CAPABILITIES_ACCELS:
        return true;
    case BOARD_CAPABILITIES_MAGS:
        return true;
    case BOARD_CAPABILITIES_BAROS:
        return true;
    case BOARD_CAPABILITIES_RADIO:
        return false;
    }
    return false;
}

/**
 * @brief Naze64::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList Naze64::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap Naze64::getBoardPicture()
{
    return QPixmap(":/stm/images/naze64.png");
}

QString Naze64::getHwUAVO()
{
    return "HwNaze64";
}

