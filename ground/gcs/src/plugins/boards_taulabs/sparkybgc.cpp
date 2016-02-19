/**
 ******************************************************************************
 * @file       sparkybgc.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
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
#include "sparkybgcconfiguration.h"

#include "hwsparkybgc.h"

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
    case BOARD_CAPABILITIES_OSD:
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
    return QPixmap(":/taulabs/images/sparkybgc.png");
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
    HwSparkyBGC *hwSparkyBGC = HwSparkyBGC::GetInstance(uavoManager);
    Q_ASSERT(hwSparkyBGC);
    if (!hwSparkyBGC)
        return false;

    HwSparkyBGC::DataFields settings = hwSparkyBGC->getData();

    switch(type) {
    case INPUT_TYPE_PPM:
        settings.RcvrPort = HwSparkyBGC::RCVRPORT_PPM;
        break;
    case INPUT_TYPE_SBUS:
        settings.RcvrPort = HwSparkyBGC::RCVRPORT_SBUS;
        break;
    case INPUT_TYPE_DSM:
        settings.RcvrPort = HwSparkyBGC::RCVRPORT_DSM;
        break;
    default:
        return false;
    }

    // Apply these changes
    hwSparkyBGC->setData(settings);

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
    HwSparkyBGC *hwSparkyBGC = HwSparkyBGC::GetInstance(uavoManager);
    Q_ASSERT(hwSparkyBGC);
    if (!hwSparkyBGC)
        return INPUT_TYPE_UNKNOWN;

    HwSparkyBGC::DataFields settings = hwSparkyBGC->getData();

    switch(settings.RcvrPort) {
    case HwSparkyBGC::RCVRPORT_PPM:
        return INPUT_TYPE_PPM;
    case HwSparkyBGC::RCVRPORT_SBUS:
        return INPUT_TYPE_SBUS;
    case HwSparkyBGC::RCVRPORT_DSM:
        return INPUT_TYPE_DSM;
    default:
        return INPUT_TYPE_UNKNOWN;
    }
}

int SparkyBGC::queryMaxGyroRate()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwSparkyBGC *hwSparkyBGC = HwSparkyBGC::GetInstance(uavoManager);
    Q_ASSERT(hwSparkyBGC);
    if (!hwSparkyBGC)
        return 0;

    HwSparkyBGC::DataFields settings = hwSparkyBGC->getData();

    switch(settings.GyroRange) {
    case HwSparkyBGC::GYRORANGE_250:
        return 250;
    case HwSparkyBGC::GYRORANGE_500:
        return 500;
    case HwSparkyBGC::GYRORANGE_1000:
        return 1000;
    case HwSparkyBGC::GYRORANGE_2000:
        return 2000;
    default:
        return 500;
    }
}

/**
 * @brief SparkyBGC::getBoardConfiguration create the custom configuration
 * dialog for SparkyBGC.
 * @param parent
 * @param connected
 * @return
 */
QWidget * SparkyBGC::getBoardConfiguration(QWidget *parent, bool connected)
{
    Q_UNUSED(connected);
    return new SparkyBgcConfiguration(parent);

}


QStringList SparkyBGC::getAdcNames()
{
    // TODO: Confirm with @peabody124 if this is correct
    return QStringList() << "VBAT" << "Current" << "RSSI";
}
