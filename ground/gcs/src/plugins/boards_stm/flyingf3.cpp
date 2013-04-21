/**
 ******************************************************************************
 *
 * @file       flyingf3.cpp
 * @author     Tau Labs, http://github.com/TauLabs, Copyright (C) 2013
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

#include "flyingf3.h"

/**
 * @brief FlyingF3::FlyingF3
 *  This is the Flying F3 board definition
 */
FlyingF3::FlyingF3(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x83;
}

FlyingF3::~FlyingF3()
{

}

QString FlyingF3::shortName()
{
    return QString("flyingf3");
}

QString FlyingF3::boardDescription()
{
    return QString("FlyingF3");
}

//! Return which capabilities this board has
bool FlyingF3::queryCapabilities(BoardCapabilities capability)
{
    switch(capability) {
    case BOARD_CAPABILITIES_GYROS:
        return true;
    case BOARD_CAPABILITIES_ACCELS:
        return true;
    case BOARD_CAPABILITIES_MAGS:
        return true;
    case BOARD_CAPABILITIES_BAROS:
        return false;
    case BOARD_CAPABILITIES_RADIO:
        return false;
    }
    return false;
}

/**
 * @brief FlyingF3::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList FlyingF3::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap FlyingF3::getBoardPicture()
{
    return QPixmap(":/stm/images/flyingf3.png");
}

QString FlyingF3::getHwUAVO()
{
    return "HwFlyingF3";
}
