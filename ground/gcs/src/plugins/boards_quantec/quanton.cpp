/**
 ******************************************************************************
 *
 * @file       quanton.cpp
 * @author     The PhoenixPilot Team, http://github.com/PhoenixPilot Copyright (C) 2013.
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_Quantec Quantec boards support Plugin
 * @{
 * @brief Plugin to support boards by Quantec
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

#include "quanton.h"

/**
 * @brief Quanton::Quanton
 *  This is the Quanton board definition
 */
Quanton::Quanton(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x0fda;
    board.productID = 0x0100;

    setUSBInfo(board);

}

Quanton::~Quanton()
{

}

QString Quanton::shortName()
{
    return QString("quanton");
}

QString Quanton::boardDescription()
{
    return QString("quanton flight control rev. 1 by Quantec Networks GmbH");
}

/**
 * @brief Quanton::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList Quanton::getSupportedProtocols()
{

    return QStringList("uavtalk");
}
