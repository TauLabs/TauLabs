/**
 ******************************************************************************
 *
 * @file       revomini.cpp
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

#include "revomini.h"

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>

#include "hwrevomini.h"

/**
 * @brief RevoMini::RevoMini
 *  This is the Revo Mini (3D) board definition
 */
RevoMini::RevoMini(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x09;

    // Define the bank of channels that are connected to a given timer
    channelBanks.resize(6);
    channelBanks[0] = QVector<int> () << 1 << 2;
    channelBanks[1] = QVector<int> () << 3;
    channelBanks[2] = QVector<int> () << 4;
    channelBanks[3] = QVector<int> () << 5 << 6;
}

RevoMini::~RevoMini()
{

}


QString RevoMini::shortName()
{
    return QString("RevoMini");
}

QString RevoMini::boardDescription()
{
    return QString("The OpenPilot project Revolution Mini boards");
}

//! Return which capabilities this board has
bool RevoMini::queryCapabilities(BoardCapabilities capability)
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
 * @brief RevoMini::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList RevoMini::getSupportedProtocols()
{

    return QStringList("uavtalk");
}


QPixmap RevoMini::getBoardPicture()
{
    return QPixmap();
}

QString RevoMini::getHwUAVO()
{
    return "HwRevoMini";
}

//! Determine if this board supports configuring the receiver
bool RevoMini::isInputConfigurationSupported()
{
    return true;
}

/**
 * Configure the board to use a receiver input type on a port number
 * @param type the type of receiver to use
 * @param port_num which input port to configure (board specific numbering)
 * @return true if successfully configured or false otherwise
 */
bool RevoMini::setInputOnPort(enum InputType type, int port_num)
{
    if (port_num != 0)
        return false;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwRevoMini *hwRevoMini = HwRevoMini::GetInstance(uavoManager);
    Q_ASSERT(hwRevoMini);
    if (!hwRevoMini)
        return false;

    HwRevoMini::DataFields settings = hwRevoMini->getData();

    // Default to serial telemetry on the serial port
    settings.MainPort = HwRevoMini::MAINPORT_TELEMETRY;

    switch(type) {
    case INPUT_TYPE_PWM:
        settings.RcvrPort = HwRevoMini::RCVRPORT_PWM;
        break;
    case INPUT_TYPE_PPM:
        settings.RcvrPort = HwRevoMini::RCVRPORT_PPM;
        break;
    case INPUT_TYPE_SBUS:
        settings.FlexiPort = HwRevoMini::FLEXIPORT_TELEMETRY;
        settings.MainPort = HwRevoMini::MAINPORT_SBUS;
        break;
    case INPUT_TYPE_DSM2:
        settings.FlexiPort = HwRevoMini::FLEXIPORT_DSM2;
        break;
    case INPUT_TYPE_DSMX10BIT:
        settings.FlexiPort = HwRevoMini::FLEXIPORT_DSMX10BIT;
        break;
    case INPUT_TYPE_DSMX11BIT:
        settings.FlexiPort = HwRevoMini::FLEXIPORT_DSMX11BIT;
        break;
    default:
        return false;
    }

    // Apply these changes
    hwRevoMini->setData(settings);

    return true;
}

/**
 * @brief RevoMini::getInputOnPort fetch the currently selected input type
 * @param port_num the port number to query (must be zero)
 * @return the selected input type
 */
enum Core::IBoardType::InputType RevoMini::getInputOnPort(int port_num)
{
    if (port_num != 0)
        return INPUT_TYPE_UNKNOWN;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwRevoMini *hwRevoMini = HwRevoMini::GetInstance(uavoManager);
    Q_ASSERT(hwRevoMini);
    if (!hwRevoMini)
        return INPUT_TYPE_UNKNOWN;

    HwRevoMini::DataFields settings = hwRevoMini->getData();

    switch(settings.FlexiPort) {
    case HwRevoMini::FLEXIPORT_DSM2:
        return INPUT_TYPE_DSM2;
    case HwRevoMini::FLEXIPORT_DSMX10BIT:
        return INPUT_TYPE_DSMX10BIT;
    case HwRevoMini::FLEXIPORT_DSMX11BIT:
        return INPUT_TYPE_DSMX11BIT;
    default:
        break;
    }

    switch(settings.MainPort) {
    case HwRevoMini::MAINPORT_SBUS:
        return INPUT_TYPE_SBUS;
    default:
        break;
    }

    switch(settings.RcvrPort) {
    case HwRevoMini::RCVRPORT_PPM:
        return INPUT_TYPE_PPM;
    case HwRevoMini::RCVRPORT_PWM:
        return INPUT_TYPE_PWM;
    default:
        break;
    }

    return INPUT_TYPE_UNKNOWN;
}

int RevoMini::queryMaxGyroRate()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwRevoMini *hwRevoMini = HwRevoMini::GetInstance(uavoManager);
    Q_ASSERT(hwRevoMini);
    if (!hwRevoMini)
        return 0;

    HwRevoMini::DataFields settings = hwRevoMini->getData();

    switch(settings.GyroRange) {
    case HwRevoMini::GYRORANGE_250:
        return 250;
    case HwRevoMini::GYRORANGE_500:
        return 500;
    case HwRevoMini::GYRORANGE_1000:
        return 1000;
    case HwRevoMini::GYRORANGE_2000:
        return 2000;
    default:
        return 500;
    }
}
