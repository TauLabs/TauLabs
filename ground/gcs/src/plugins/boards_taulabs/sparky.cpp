/**
 ******************************************************************************
 * @file       sparky.cpp
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

#include "sparky.h"

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>

#include "hwsparky.h"

/**
 * @brief Sparky::Sparky
 *  This is the Sparky board definition
 */
Sparky::Sparky(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x88;

    // Define the bank of channels that are connected to a given timer
    channelBanks.resize(6);
    channelBanks[0] = QVector<int> () << 1 << 2;      // TIM15
    channelBanks[1] = QVector<int> () << 3;           // TIM1
    channelBanks[2] = QVector<int> () << 4 << 7 << 9; // TIM3
    channelBanks[3] = QVector<int> () << 5;           // TIM16
    channelBanks[4] = QVector<int> () << 6 << 10;     // TIM2
    channelBanks[5] = QVector<int> () << 8;           // TIM17
}

Sparky::~Sparky()
{

}


QString Sparky::shortName()
{
    return QString("Sparky");
}

QString Sparky::boardDescription()
{
    return QString("The Tau Labs project Sparky boards");
}

//! Return which capabilities this board has
bool Sparky::queryCapabilities(BoardCapabilities capability)
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
 * @brief Sparky::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList Sparky::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap Sparky::getBoardPicture()
{
    return QPixmap(":/taulabs/images/sparky.png");
}

QString Sparky::getHwUAVO()
{
    return "HwSparky";
}

//! Determine if this board supports configuring the receiver
bool Sparky::isInputConfigurationSupported()
{
    return true;
}

/**
 * Configure the board to use a receiver input type on a port number
 * @param type the type of receiver to use
 * @param port_num which input port to configure (board specific numbering)
 * @return true if successfully configured or false otherwise
 */
bool Sparky::setInputOnPort(enum InputType type, int port_num)
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
    case INPUT_TYPE_DSM:
        settings.RcvrPort = HwSparky::RCVRPORT_DSM;
        break;
    case INPUT_TYPE_HOTTSUMD:
        settings.RcvrPort = HwSparky::RCVRPORT_HOTTSUMD;
        break;
    default:
        return false;
    }

    // Apply these changes
    hwSparky->setData(settings);

    return true;
}

/**
 * @brief Sparky::getInputOnPort fetch the currently selected input type
 * @param port_num the port number to query (must be zero)
 * @return the selected input type
 */
enum Core::IBoardType::InputType Sparky::getInputOnPort(int port_num)
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
    case HwSparky::RCVRPORT_DSM:
        return INPUT_TYPE_DSM;
    case HwSparky::RCVRPORT_HOTTSUMD:
        return INPUT_TYPE_HOTTSUMD;
    default:
        return INPUT_TYPE_UNKNOWN;
    }
}

int Sparky::queryMaxGyroRate()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwSparky *hwSparky = HwSparky::GetInstance(uavoManager);
    Q_ASSERT(hwSparky);
    if (!hwSparky)
        return 0;

    HwSparky::DataFields settings = hwSparky->getData();

    switch(settings.GyroRange) {
    case HwSparky::GYRORANGE_250:
        return 250;
    case HwSparky::GYRORANGE_500:
        return 500;
    case HwSparky::GYRORANGE_1000:
        return 1000;
    case HwSparky::GYRORANGE_2000:
        return 2000;
    default:
        return 500;
    }
}
