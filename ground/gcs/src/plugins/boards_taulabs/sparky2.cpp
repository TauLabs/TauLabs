/**
 ******************************************************************************
 * @file       sparky2.cpp
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

#include "sparky2.h"

#include <uavobjectmanager.h>
#include <extensionsystem/pluginmanager.h>

#include "rfm22bstatus.h"


/**
 * @brief Sparky2::Sparky2
 *  This is the Sparky board definition
 */
Sparky2::Sparky2(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415b;

    setUSBInfo(board);

    boardType = 0x92;

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
    channelBanks.resize(6);
    channelBanks[0] = QVector<int> () << 1 << 2;
    channelBanks[1] = QVector<int> () << 3 << 4;
    channelBanks[2] = QVector<int> () << 5 << 6;
    channelBanks[3] = QVector<int> () << 7 << 8;
    channelBanks[4] = QVector<int> () << 9 << 10;
    channelBanks[5] = QVector<int> ();

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    uavoUtilManager = pm->getObject<UAVObjectUtilManager>();
}

Sparky2::~Sparky2()
{

}


QString Sparky2::shortName()
{
    return QString("Sparky2");
}

QString Sparky2::boardDescription()
{
    return QString("The Tau Labs project Sparky2 boards");
}

//! Return which capabilities this board has
bool Sparky2::queryCapabilities(BoardCapabilities capability)
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
 * @brief Sparky2::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 * @return
 */
QStringList Sparky2::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap Sparky2::getBoardPicture()
{
    return QPixmap(":/taulabs/images/sparky2.png");
}

QString Sparky2::getHwUAVO()
{
    return "HwSparky2";
}

//! Get the settings object
HwSparky2 * Sparky2::getSettings()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();

    HwSparky2 *hwSparky2 = HwSparky2::GetInstance(uavoManager);
    Q_ASSERT(hwSparky2);

    return hwSparky2;
}

//! Determine if this board supports configuring the receiver
bool Sparky2::isInputConfigurationSupported()
{
    return true;
}

/**
 * Configure the board to use a receiver input type on a port number
 * @param type the type of receiver to use
 * @param port_num which input port to configure (board specific numbering)
 * @return true if successfully configured or false otherwise
 */
bool Sparky2::setInputOnPort(enum InputType type, int port_num)
{
    if (port_num != 0)
        return false;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwSparky2 *hwSparky2 = HwSparky2::GetInstance(uavoManager);
    Q_ASSERT(hwSparky2);
    if (!hwSparky2)
        return false;

    HwSparky2::DataFields settings = hwSparky2->getData();

    switch(type) {
    case INPUT_TYPE_PPM:
        settings.RcvrPort = HwSparky2::RCVRPORT_PPM;
        break;
    case INPUT_TYPE_SBUS:
        settings.RcvrPort = HwSparky2::RCVRPORT_SBUS;
        break;
    case INPUT_TYPE_DSM:
        settings.RcvrPort = HwSparky2::RCVRPORT_DSM;
        break;
    default:
        return false;
    }

    // Apply these changes
    hwSparky2->setData(settings);

    return true;
}

/**
 * @brief Sparky2::getInputOnPort fetch the currently selected input type
 * @param port_num the port number to query (must be zero)
 * @return the selected input type
 */
enum Core::IBoardType::InputType Sparky2::getInputOnPort(int port_num)
{
    if (port_num != 0)
        return INPUT_TYPE_UNKNOWN;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwSparky2 *hwSparky2 = HwSparky2::GetInstance(uavoManager);
    Q_ASSERT(hwSparky2);
    if (!hwSparky2)
        return INPUT_TYPE_UNKNOWN;

    HwSparky2::DataFields settings = hwSparky2->getData();

    switch(settings.RcvrPort) {
    case HwSparky2::RCVRPORT_PPM:
        return INPUT_TYPE_PPM;
    case HwSparky2::RCVRPORT_SBUS:
        return INPUT_TYPE_SBUS;
    case HwSparky2::RCVRPORT_DSM:
        return INPUT_TYPE_DSM;
    default:
        return INPUT_TYPE_UNKNOWN;
    }
}

int Sparky2::queryMaxGyroRate()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    HwSparky2 *hwSparky2 = HwSparky2::GetInstance(uavoManager);
    Q_ASSERT(hwSparky2);
    if (!hwSparky2)
        return 0;

    HwSparky2::DataFields settings = hwSparky2->getData();

    switch(settings.GyroRange) {
    case HwSparky2::GYRORANGE_250:
        return 250;
    case HwSparky2::GYRORANGE_500:
        return 500;
    case HwSparky2::GYRORANGE_1000:
        return 1000;
    case HwSparky2::GYRORANGE_2000:
        return 2000;
    default:
        return 500;
    }
}


/**
 * Get the RFM22b device ID this modem
 * @return RFM22B device ID or 0 if not supported
 */
quint32 Sparky2::getRfmID()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();

    // Flight controllers are instance 1
    RFM22BStatus *rfm22bStatus = RFM22BStatus::GetInstance(uavoManager,1);
    Q_ASSERT(rfm22bStatus);
    RFM22BStatus::DataFields rfm22b = rfm22bStatus->getData();

    return rfm22b.DeviceID;
}

/**
 * Set the coordinator ID. If set to zero this device will
 * be a coordinator.
 * @return true if successful or false if not
 */
bool Sparky2::bindRadio(quint32 id, quint32 baud_rate, float rf_power,
                        Core::IBoardType::LinkMode linkMode, quint8 min, quint8 max)
{
    HwSparky2::DataFields settings = getSettings()->getData();

    settings.CoordID = id;

    switch(baud_rate) {
    case 9600:
        settings.MaxRfSpeed = HwSparky2::MAXRFSPEED_9600;
        break;
    case 19200:
        settings.MaxRfSpeed = HwSparky2::MAXRFSPEED_19200;
        break;
    case 32000:
        settings.MaxRfSpeed = HwSparky2::MAXRFSPEED_32000;
        break;
    case 64000:
        settings.MaxRfSpeed = HwSparky2::MAXRFSPEED_64000;
        break;
    case 100000:
        settings.MaxRfSpeed = HwSparky2::MAXRFSPEED_100000;
        break;
    case 192000:
        settings.MaxRfSpeed = HwSparky2::MAXRFSPEED_192000;
        break;
    }

    // Round to an integer to use a switch statement
    quint32 rf_power_100 = rf_power * 100;
    switch(rf_power_100) {
    case 0:
        settings.MaxRfPower = HwSparky2::MAXRFPOWER_0;
        break;
    case 125:
        settings.MaxRfPower = HwSparky2::MAXRFPOWER_125;
        break;
    case 160:
        settings.MaxRfPower = HwSparky2::MAXRFPOWER_16;
        break;
    case 316:
        settings.MaxRfPower = HwSparky2::MAXRFPOWER_316;
        break;
    case 630:
        settings.MaxRfPower = HwSparky2::MAXRFPOWER_63;
        break;
    case 1260:
        settings.MaxRfPower = HwSparky2::MAXRFPOWER_126;
        break;
    case 2500:
        settings.MaxRfPower = HwSparky2::MAXRFPOWER_25;
        break;
    case 5000:
        settings.MaxRfPower = HwSparky2::MAXRFPOWER_50;
        break;
    case 10000:
        settings.MaxRfPower = HwSparky2::MAXRFPOWER_100;
        break;
    }

    switch(linkMode) {
    case Core::IBoardType::LINK_TELEM:
        settings.Radio = HwSparky2::RADIO_TELEM;
        break;
    case Core::IBoardType::LINK_TELEM_PPM:
        settings.Radio = HwSparky2::RADIO_TELEMPPM;
        break;
    case Core::IBoardType::LINK_PPM:
        settings.Radio = HwSparky2::RADIO_PPM;
        break;
    }

    settings.MinChannel = min;
    settings.MaxChannel = max;

    getSettings()->setData(settings);
    uavoUtilManager->saveObjectToFlash(getSettings());

    return true;
}
