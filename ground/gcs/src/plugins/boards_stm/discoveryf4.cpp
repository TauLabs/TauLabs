/**
 ******************************************************************************
 *
 * @file       discoveryf4.cpp
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

#include "discoveryf4.h"

/**
 * @brief DiscoveryF4::DiscoveryF4
 *  This is the DiscoveryF4 board definition
 */
DiscoveryF4::DiscoveryF4(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x85;
}

DiscoveryF4::~DiscoveryF4()
{

}

QString DiscoveryF4::shortName()
{
    return QString("discoveryf4");
}

QString DiscoveryF4::boardDescription()
{
    return QString("DiscoveryF4");
}

//! Return which capabilities this board has
bool DiscoveryF4::queryCapabilities(BoardCapabilities capability)
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
        return false;
    case BOARD_CAPABILITIES_OSD:
        return false;
    }
    return false;
}

/**
 * @brief DiscoveryF4::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList DiscoveryF4::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap DiscoveryF4::getBoardPicture()
{
    return QPixmap();
}

QString DiscoveryF4::getHwUAVO()
{
    return "HwDiscoveryF4";
}
