/**
 ******************************************************************************
 * @file       tauosd.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
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

#include "tauosd.h"

#include <uavobjectmanager.h>
#include <extensionsystem/pluginmanager.h>


/**
 * @brief TauOsd::TauOsd
 *  This is the TauOsd board definition
 */
TauOsd::TauOsd(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x05;

    // Define the bank of channels that are connected to a given timer
    //   Ch1 TIM3
    //   Ch2 TIM3
    //   Ch3 TIM9
    //   Ch4 TIM9
    //   Ch5 TIM5
    //   Ch6 TIM5
    //  LED1 TIM12
    //  LED2 TIM12
    //  LED3 TIM8
    //  LED4 TIM8
    channelBanks.resize(0);

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    uavoUtilManager = pm->getObject<UAVObjectUtilManager>();
}

TauOsd::~TauOsd()
{

}

QString TauOsd::shortName()
{
    return QString("TauOsd");
}

QString TauOsd::boardDescription()
{
    return QString("The Tau Labs project TauOsd boards");
}

/**
 * @brief TauOsd::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList TauOsd::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap TauOsd::getBoardPicture()
{
    return QPixmap(":/taulabs/images/tauosd.png");
}

QString TauOsd::getHwUAVO()
{
    return "HwTauOsd";
}

//! Get the settings object
HwTauOsd * TauOsd::getSettings()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();

    HwTauOsd *hwTauOsd = HwTauOsd::GetInstance(uavoManager);
    Q_ASSERT(hwTauOsd);

    return hwTauOsd;
}

//! Return which capabilities this board has
bool TauOsd::queryCapabilities(BoardCapabilities capability)
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
        return true;
    }
    return false;
}
