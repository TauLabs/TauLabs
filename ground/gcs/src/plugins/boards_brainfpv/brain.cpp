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
    board.productID = 0x4242;

    setUSBInfo(board);

    boardType = 0x8A;

    // Define the bank of channels that are connected to a given timer
    channelBanks.resize(3);
    channelBanks[0] = QVector<int> () << 1 << 2 << 3 << 4; // TIM5 main outputs
    channelBanks[1] = QVector<int> () << 5 << 6 << 7 << 8; // TIM8 on receiverport
    channelBanks[2] = QVector<int> () << 9 << 10; // TIM12 on receiverport
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
    case INPUT_TYPE_SBUS:
        settings.MainPort = HwBrain::MAINPORT_SBUS;
        break;
    case INPUT_TYPE_DSM:
        settings.MainPort = HwBrain::MAINPORT_DSM;
        break;
    case INPUT_TYPE_HOTTSUMD:
        settings.MainPort = HwBrain::MAINPORT_HOTTSUMD;
        break;
    case INPUT_TYPE_HOTTSUMH:
        settings.MainPort = HwBrain::MAINPORT_HOTTSUMH;
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

    switch(settings.MainPort) {
        case HwBrain::MAINPORT_SBUS:
            return INPUT_TYPE_SBUS;
        case HwBrain::MAINPORT_DSM:
            return INPUT_TYPE_DSM;
        case HwBrain::MAINPORT_HOTTSUMD:
            return INPUT_TYPE_HOTTSUMD;
        case HwBrain::MAINPORT_HOTTSUMH:
            return INPUT_TYPE_HOTTSUMH;
    }

    switch(settings.FlxPort) {
        case HwBrain::FLXPORT_DSM:
            return INPUT_TYPE_DSM;
        case HwBrain::FLXPORT_HOTTSUMD:
            return INPUT_TYPE_HOTTSUMD;
        case HwBrain::FLXPORT_HOTTSUMH:
            return INPUT_TYPE_HOTTSUMH;
    }

    switch(settings.RxPort) {
        case HwBrain::RXPORT_PPM:
        case HwBrain::RXPORT_PPMPWM:
        case HwBrain::RXPORT_PPMOUTPUTS:
        case HwBrain::RXPORT_PPMUART:
        case HwBrain::RXPORT_PPMUARTOUTPUTS:
        case HwBrain::RXPORT_PPMFRSKY:
            return INPUT_TYPE_PPM;
        case HwBrain::RXPORT_PWM:
            return INPUT_TYPE_PWM;
    }

    return INPUT_TYPE_UNKNOWN;
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

    switch(settings.GyroFullScale) {
    case HwBrain::GYROFULLSCALE_250:
        return 250;
    case HwBrain::GYROFULLSCALE_500:
        return 500;
    case HwBrain::GYROFULLSCALE_1000:
        return 1000;
    case HwBrain::GYROFULLSCALE_2000:
        return 2000;
    default:
        return 2000;
    }
}

QWidget * Brain::getBoardConfiguration(QWidget *parent, bool connected)
{
    Q_UNUSED(connected);
    return new BrainConfiguration(parent);
}
