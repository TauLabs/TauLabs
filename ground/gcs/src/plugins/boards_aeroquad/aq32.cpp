/**
 ******************************************************************************
 * @file       aq32.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_AeroQuadPlugin AeroQuad boards support Plugin
 * @{
 * @brief Plugin to support AeroQuad boards
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

#include "aq32.h"

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>

#include "hwaq32.h"

/**
 * @brief AQ32::AQ32
 *  This is the AQ32 board definition
 */
AQ32::AQ32(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID  = 0x0FDA;
    board.productID = 0x0100;

    setUSBInfo(board);

    boardType = 0x94;

    // Define the bank of channels that are connected to a given timer
    channelBanks.resize(5);
    channelBanks[0] = QVector<int> () << 1 << 2;
    channelBanks[1] = QVector<int> () << 3 << 4;
    channelBanks[2] = QVector<int> () << 5 << 6;
    channelBanks[3] = QVector<int> () << 7 << 8 << 9 <<10;
}

AQ32::~AQ32()
{

}


QString AQ32::shortName()
{
    return QString("AQ32");
}

QString AQ32::boardDescription()
{
    return QString("The AQ32 board");
}

//! Return which capabilities this board has
bool AQ32::queryCapabilities(BoardCapabilities capability)
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
 * @brief AQ32::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList AQ32::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap AQ32::getBoardPicture()
{
    return QPixmap(":/aq32/images/aq32.png");
}

QString AQ32::getHwUAVO()
{
    return "HwAQ32";
}

int AQ32::queryMaxGyroRate()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwAQ32 *hwAQ32 = HwAQ32::GetInstance(uavoManager);
    Q_ASSERT(hwAQ32);
    if (!hwAQ32)
        return 0;

    HwAQ32::DataFields settings = hwAQ32->getData();

    switch(settings.GyroRange) {
    case HwAQ32::GYRORANGE_250:
        return 250;
    case HwAQ32::GYRORANGE_500:
        return 500;
    case HwAQ32::GYRORANGE_1000:
        return 1000;
    case HwAQ32::GYRORANGE_2000:
        return 2000;
    default:
        return 500;
    }
}
