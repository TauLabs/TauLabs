/**
 ******************************************************************************
 *
 * @file       revolution.cpp
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

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>
#include "revolution.h"
#include "hwrevolution.h"

/**
 * @brief Revolution::Revolution
 *  This is the Revolution board definition
 */
Revolution::Revolution(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x7f;

    // Define the bank of channels that are connected to a given timer
    channelBanks.resize(6);
    channelBanks[0] = QVector<int> () << 1 << 2;
    channelBanks[1] = QVector<int> () << 3;
    channelBanks[2] = QVector<int> () << 4;
    channelBanks[3] = QVector<int> () << 5 << 6;
}

Revolution::~Revolution()
{

}


QString Revolution::shortName()
{
    return QString("Revolution");
}

QString Revolution::boardDescription()
{
    return QString("The OpenPilot project Revolution boards");
}

//! Return which capabilities this board has
bool Revolution::queryCapabilities(BoardCapabilities capability)
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
 * @brief Revolution::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList Revolution::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap Revolution::getBoardPicture()
{
    return QPixmap();
}

QString Revolution::getHwUAVO()
{
    return "HwRevolution";
}

int Revolution::queryMaxGyroRate()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwRevolution *hwRevolution = HwRevolution::GetInstance(uavoManager);
    Q_ASSERT(hwRevolution);
    if (!hwRevolution)
        return 0;

    HwRevolution::DataFields settings = hwRevolution->getData();

    switch(settings.GyroRange) {
    case HwRevolution::GYRORANGE_250:
        return 250;
    case HwRevolution::GYRORANGE_500:
        return 500;
    case HwRevolution::GYRORANGE_1000:
        return 1000;
    case HwRevolution::GYRORANGE_2000:
        return 2000;
    default:
        return 500;
    }
}
