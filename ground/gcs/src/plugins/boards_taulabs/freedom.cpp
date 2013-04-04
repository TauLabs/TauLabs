/**
 ******************************************************************************
 * @file       freedom.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

#include "freedom.h"

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>

#include "hwfreedom.h"

/**
 * @brief Freedom::Freedom
 *  This is the Freedom board definition
 */
Freedom::Freedom(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x81;

    // Define the bank of channels that are connected to a given timer
    channelBanks.resize(6);
    channelBanks[0] = QVector<int> () << 1 << 2;
    channelBanks[1] = QVector<int> () << 3 << 4;
    channelBanks[2] = QVector<int> () << 6 << 7;
}

Freedom::~Freedom()
{

}

QString Freedom::shortName()
{
    return QString("Freedom");
}

QString Freedom::boardDescription()
{
    return QString("The Tau Labs project Freedom boards");
}

//! Return which capabilities this board has
bool Freedom::queryCapabilities(BoardCapabilities capability)
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
        return true;
    }
    return false;
}


/**
 * @brief Freedom::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList Freedom::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap Freedom::getBoardPicture()
{
    return QPixmap(":/taulabs/images/freedom.png");
}

QString Freedom::getHwUAVO()
{
    return "HwFreedom";
}

int Freedom::queryMaxGyroRate()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwFreedom *hwFreedom = HwFreedom::GetInstance(uavoManager);
    Q_ASSERT(hwFreedom);
    if (!hwFreedom)
        return 0;

    HwFreedom::DataFields settings = hwFreedom->getData();

    switch(settings.GyroRange) {
    case HwFreedom::GYRORANGE_250:
        return 250;
    case HwFreedom::GYRORANGE_500:
        return 500;
    case HwFreedom::GYRORANGE_1000:
        return 1000;
    case HwFreedom::GYRORANGE_2000:
        return 2000;
    default:
        return 500;
    }
}
