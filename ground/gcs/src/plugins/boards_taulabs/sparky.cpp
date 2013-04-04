/**
 ******************************************************************************
 * @file       sparky.cpp
 * @author     Tau Labs, http://github.com/TauLabs, Copyright (C) 2013.
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_TauLabsPlugin Tau Labs boards support Plugin
 * @{
 * @brief Plugin to support boards by the Tau Labs project
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

#include "sparky.h"

/**
 * @brief Sparky::Sparky
 *  This is the Sparky board definition
 */
Sparky::Sparky(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x88;
}

Sparky::~Sparky()
{

}


QString Sparky::shortName()
{
    return QString("Sparky");
}

QString Sparky::boardDescription()
{
    return QString("The Tau Labs project Sparky boards");
}

//! Return which capabilities this board has
bool Sparky::queryCapabilities(BoardCapabilities capability)
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
 * @brief Sparky::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList Sparky::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap Sparky::getBoardPicture()
{
    return QPixmap(":/images/sparky.png");
}
