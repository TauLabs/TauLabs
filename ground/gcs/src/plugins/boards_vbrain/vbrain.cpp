/**
 ******************************************************************************
 *
 * @file       vbrain.cpp
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2013
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_Vbrain Vbrain boards support Plugin
 * @{
 * @brief Plugin to support boards by Vbrain
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

#include "vbrain.h"

/**
 * @brief Vbrain::Vbrain
 *  This is the Vbrain board definition
 */
Vbrain::Vbrain(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20a0;
    board.productID = 0x415a;

    setUSBInfo(board);

}

Vbrain::~Vbrain()
{

}

QString Vbrain::shortName()
{
    return QString("vbrain");
}

QString Vbrain::boardDescription()
{
    return QString("vbrain flight control rev. 1");
}

/**
 * @brief Vbrain::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList Vbrain::getSupportedProtocols()
{

    return QStringList("uavtalk");
}
