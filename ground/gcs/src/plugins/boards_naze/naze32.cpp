/**
 ******************************************************************************
 *
 * @file       naze32.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_Naze Naze boards support Plugin
 * @{
 * @brief Plugin to support Naze boards
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

#include "naze32.h"

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>

#include "hwnaze.h"

/**
 * @brief Naze:Naze
 *  This is the Naze board definition
 */
Naze::Naze(void)
{
    boardType = 0xA0;

    // Define the bank of channels that are connected to a given timer
    channelBanks.resize(6);
    channelBanks[0] = QVector<int> () << 1 << 2; // T1
    channelBanks[1] = QVector<int> () << 3 << 4 << 5 << 6; // T4
    channelBanks[2] = QVector<int> () << 7 << 8 << 9 << 10; // T2
    channelBanks[3] = QVector<int> () << 11 << 12 << 13 << 14; // T3
}

Naze::~Naze()
{

}

QString Naze::shortName()
{
    return QString("naze");
}

QString Naze::boardDescription()
{
    return QString("Naze32 flight controller by timecop");
}

//! Return which capabilities this board has
bool Naze::queryCapabilities(BoardCapabilities capability)
{
    switch(capability) {
    case BOARD_CAPABILITIES_GYROS:
        return true;
    case BOARD_CAPABILITIES_ACCELS:
        return true;
    case BOARD_CAPABILITIES_MAGS:
        return false;
    case BOARD_CAPABILITIES_BAROS:
        return false;
    case BOARD_CAPABILITIES_RADIO:
        return false;
    }
    return false;
}


/**
 * @brief Naze::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList Naze::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap Naze::getBoardPicture()
{
    return QPixmap(":/naze/images/naze32.png");
}

QString Naze::getHwUAVO()
{
    return "HwNaze";
}

int Naze::queryMaxGyroRate()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwNaze *hwNaze = HwNaze::GetInstance(uavoManager);
    Q_ASSERT(hwNaze);
    if (!hwNaze)
        return 0;

    HwNaze::DataFields settings = hwNaze->getData();

    switch(settings.GyroRange) {
    case HwNaze::GYRORANGE_250:
        return 250;
    case HwNaze::GYRORANGE_500:
        return 500;
    case HwNaze::GYRORANGE_1000:
        return 1000;
    case HwNaze::GYRORANGE_2000:
        return 2000;
    default:
        return 500;
    }
}
