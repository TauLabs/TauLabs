/**
 ******************************************************************************
 *
 * @file       colibri.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_TBS TBS boards support Plugin
 * @{
 * @brief Plugin to support boards by TBS
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

#include "colibri.h"

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>

#include "hwcolibri.h"

/**
 * @brief Colibri::Colibri
 *  This is the Colibri board definition
 */
Colibri::Colibri(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x0fda;
    board.productID = 0x0100;

    setUSBInfo(board);

    boardType = 0x86;

    // Define the bank of channels that are connected to a given timer
    channelBanks.resize(6);
    channelBanks[0] = QVector<int> () << 1 << 2 << 3 << 4;
    channelBanks[1] = QVector<int> () << 5 << 6;
    channelBanks[2] = QVector<int> () << 7;
    channelBanks[3] = QVector<int> () << 8;
}

Colibri::~Colibri()
{

}

QString Colibri::shortName()
{
    return QString("colibri");
}

QString Colibri::boardDescription()
{
    return QString("colibri flight control rev. 1 by TBS");
}

//! Return which capabilities this board has
bool Colibri::queryCapabilities(BoardCapabilities capability)
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
 * @brief Colibri::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList Colibri::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap Colibri::getBoardPicture()
{
    return QPixmap(":/TBS/images/colibri.png");
}

QString Colibri::getHwUAVO()
{
    return "HwColibri";
}

int Colibri::queryMaxGyroRate()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwColibri *hwColibri = HwColibri::GetInstance(uavoManager);
    Q_ASSERT(hwColibri);
    if (!hwColibri)
        return 0;

    HwColibri::DataFields settings = hwColibri->getData();

    switch(settings.GyroRange) {
    case HwColibri::GYRORANGE_250:
        return 250;
    case HwColibri::GYRORANGE_500:
        return 500;
    case HwColibri::GYRORANGE_1000:
        return 1000;
    case HwColibri::GYRORANGE_2000:
        return 2000;
    default:
        return 500;
    }
}
