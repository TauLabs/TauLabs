/**
 ******************************************************************************
 *
 * @file       flyingf4.cpp
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

#include "flyingf4.h"

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>

#include "hwflyingf4.h"
/**
 * @brief Quanton::Quanton
 *  This is the Quanton board definition
 */
FlyingF4::FlyingF4(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x84;

    // Define the bank of channels that are connected to a given timer
    channelBanks.resize(6);
    channelBanks[0] = QVector<int> () << 1 << 2 << 3 << 4;
    channelBanks[1] = QVector<int> () << 5 << 6 << 7 << 8;
}

FlyingF4::~FlyingF4()
{

}

QString FlyingF4::shortName()
{
    return QString("flyingf4");
}

QString FlyingF4::boardDescription()
{
    return QString("FlyingF4");
}

//! Return which capabilities this board has
bool FlyingF4::queryCapabilities(BoardCapabilities capability)
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
 * @brief FlyingF4::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList FlyingF4::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap FlyingF4::getBoardPicture()
{
    return QPixmap();
}

QString FlyingF4::getHwUAVO()
{
    return "HwFlyingF4";
}

int FlyingF4::queryMaxGyroRate()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwFlyingF4 *hwFlyingF4 = HwFlyingF4::GetInstance(uavoManager);
    Q_ASSERT(hwFlyingF4);
    if (!hwFlyingF4)
        return 0;

    HwFlyingF4::DataFields settings = hwFlyingF4->getData();

    switch(settings.GyroRange) {
    case HwFlyingF4::GYRORANGE_250:
        return 250;
    case HwFlyingF4::GYRORANGE_500:
        return 500;
    case HwFlyingF4::GYRORANGE_1000:
        return 1000;
    case HwFlyingF4::GYRORANGE_2000:
        return 2000;
    default:
        return 500;
    }
}
