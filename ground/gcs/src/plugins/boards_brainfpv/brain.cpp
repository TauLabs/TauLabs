/**
 ******************************************************************************
 * @file       Brain.cpp
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

#include "brain.h"

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>

#include "brainconfiguration.h"
#include "hwbrain.h"

/**
 * @brief Brain::Brain
 *  This is the Brain board definition
 */
Brain::Brain(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x8A;

    // Define the bank of channels that are connected to a given timer
    channelBanks.resize(3);
    channelBanks[0] = QVector<int> () << 1 << 4;
    channelBanks[1] = QVector<int> () << 5 << 8;
    channelBanks[2] = QVector<int> () << 9 << 10;
}

Brain::~Brain()
{

}


QString Brain::shortName()
{
    return QString("Brain");
}

QString Brain::boardDescription()
{
    return QString("Brain - FPV Flight Controller");
}

//! Return which capabilities this board has
bool Brain::queryCapabilities(BoardCapabilities capability)
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
        return true;
    }
    return false;
}


/**
 * @brief Brain::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList Brain::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap Brain::getBoardPicture()
{
    return QPixmap(":/brainfpv/images/brain.png");
}

QString Brain::getHwUAVO()
{
    return "HwBrain";
}

//! Determine if this board supports configuring the receiver
bool Brain::isInputConfigurationSupported()
{
    return true;
}

/**
 * Configure the board to use a receiver input type on a port number
 * @param type the type of receiver to use
 * @param port_num which input port to configure (board specific numbering)
 * @return true if successfully configured or false otherwise
 */
bool Brain::setInputOnPort(enum InputType type, int port_num)
{
    if (port_num != 0)
        return false;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwBrain *hwBrain = HwBrain::GetInstance(uavoManager);
    Q_ASSERT(hwBrain);
    if (!hwBrain)
        return false;

    HwBrain::DataFields settings = hwBrain->getData();

    switch(type) {
    case INPUT_TYPE_PPM:
        settings.RxPort = HwBrain::RXPORT_PPM;
        break;
    case INPUT_TYPE_PWM:
        settings.RxPort = HwBrain::RXPORT_PWM;
        break;
    default:
        return false;
    }

    // Apply these changes
    hwBrain->setData(settings);

    return true;
}

/**
 * @brief Brain::getInputOnPort fetch the currently selected input type
 * @param port_num the port number to query (must be zero)
 * @return the selected input type
 */
enum Core::IBoardType::InputType Brain::getInputOnPort(int port_num)
{
    if (port_num != 0)
        return INPUT_TYPE_UNKNOWN;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwBrain *hwBrain = HwBrain::GetInstance(uavoManager);
    Q_ASSERT(hwBrain);
    if (!hwBrain)
        return INPUT_TYPE_UNKNOWN;

    HwBrain::DataFields settings = hwBrain->getData();

    switch(settings.RxPort) {
    case HwBrain::RXPORT_PPM:
        return INPUT_TYPE_PPM;
    case HwBrain::RXPORT_PWM:
        return INPUT_TYPE_PWM;
    default:
        return INPUT_TYPE_UNKNOWN;
    }
}

int Brain::queryMaxGyroRate()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwBrain *hwBrain = HwBrain::GetInstance(uavoManager);
    Q_ASSERT(hwBrain);
    if (!hwBrain)
        return 0;

    HwBrain::DataFields settings = hwBrain->getData();

    switch(settings.GyroRange) {
    case HwBrain::GYRORANGE_250:
        return 250;
    case HwBrain::GYRORANGE_500:
        return 500;
    case HwBrain::GYRORANGE_1000:
        return 1000;
    case HwBrain::GYRORANGE_2000:
        return 2000;
    default:
        return 500;
    }
}

QWidget * Brain::getBoardConfiguration(QWidget *parent, bool connected)
{
    Q_UNUSED(connected);
    return new BrainConfiguration(parent);
}