/**
 ******************************************************************************
 * @file       sparkybgc.cpp
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

#include "sparkybgc.h"

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>

#include "hwsparky.h"

/**
 * @brief Sparky::Sparky
 *  This is the Sparky board definition
 */
SparkyBGC::SparkyBGC(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x89;

    // Define the bank of channels that are connected to a given timer
    channelBanks.resize(6);
    channelBanks[0] = QVector<int> () << 1 << 2;
    channelBanks[1] = QVector<int> () << 3;
    channelBanks[2] = QVector<int> () << 4 << 7 << 9;
    channelBanks[3] = QVector<int> () << 5;
    channelBanks[4] = QVector<int> () << 6 << 10;
    channelBanks[5] = QVector<int> () << 8;
}

SparkyBGC::~SparkyBGC()
{

}


QString SparkyBGC::shortName()
{
    return QString("SparkyBGC");
}

QString SparkyBGC::boardDescription()
{
    return QString("The Tau Labs project Sparky BGC boards");
}

//! Return which capabilities this board has
bool SparkyBGC::queryCapabilities(BoardCapabilities capability)
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
 * @brief SparkyBGC::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList SparkyBGC::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap SparkyBGC::getBoardPicture()
{
    return QPixmap(":/taulabs/images/sparky.png");
}

QString SparkyBGC::getHwUAVO()
{
    return "HwSparky";
}

//! Determine if this board supports configuring the receiver
bool SparkyBGC::isInputConfigurationSupported()
{
    return true;
}

/**
 * Configure the board to use a receiver input type on a port number
 * @param type the type of receiver to use
 * @param port_num which input port to configure (board specific numbering)
 * @return true if successfully configured or false otherwise
 */
bool SparkyBGC::setInputOnPort(enum InputType type, int port_num)
{
    if (port_num != 0)
        return false;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwSparky *hwSparky = HwSparky::GetInstance(uavoManager);
    Q_ASSERT(hwSparky);
    if (!hwSparky)
        return false;

    HwSparky::DataFields settings = hwSparky->getData();

    switch(type) {
    case INPUT_TYPE_PPM:
        settings.RcvrPort = HwSparky::RCVRPORT_PPM;
        break;
    case INPUT_TYPE_SBUS:
        settings.RcvrPort = HwSparky::RCVRPORT_SBUS;
        break;
    case INPUT_TYPE_DSM2:
        settings.RcvrPort = HwSparky::RCVRPORT_DSM2;
        break;
    case INPUT_TYPE_DSMX10BIT:
        settings.RcvrPort = HwSparky::RCVRPORT_DSMX10BIT;
        break;
    case INPUT_TYPE_DSMX11BIT:
        settings.RcvrPort = HwSparky::RCVRPORT_DSMX11BIT;
        break;
    default:
        return false;
    }

    // Apply these changes
    hwSparky->setData(settings);

    return true;
}

/**
 * @brief SparkyBGC::getInputOnPort fetch the currently selected input type
 * @param port_num the port number to query (must be zero)
 * @return the selected input type
 */
enum Core::IBoardType::InputType SparkyBGC::getInputOnPort(int port_num)
{
    if (port_num != 0)
        return INPUT_TYPE_UNKNOWN;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwSparky *hwSparky = HwSparky::GetInstance(uavoManager);
    Q_ASSERT(hwSparky);
    if (!hwSparky)
        return INPUT_TYPE_UNKNOWN;

    HwSparky::DataFields settings = hwSparky->getData();

    switch(settings.RcvrPort) {
    case HwSparky::RCVRPORT_PPM:
        return INPUT_TYPE_PPM;
    case HwSparky::RCVRPORT_SBUS:
        return INPUT_TYPE_SBUS;
    case HwSparky::RCVRPORT_DSM2:
        return INPUT_TYPE_DSM2;
    case HwSparky::RCVRPORT_DSMX10BIT:
        return INPUT_TYPE_DSMX10BIT;
    case HwSparky::RCVRPORT_DSMX11BIT:
        return INPUT_TYPE_DSMX11BIT;
    default:
        return INPUT_TYPE_UNKNOWN;
    }
}
