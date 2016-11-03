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
#include <extensionsystem/pluginmanager.h>

#include "rfm22bstatus.h"

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
    channelBanks[1] = QVector<int> () << 3;
    channelBanks[2] = QVector<int> () << 4 << 5;
    channelBanks[3] = QVector<int> () << 6;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    uavoUtilManager = pm->getObject<UAVObjectUtilManager>();
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
    case BOARD_CAPABILITIES_OSD:
        return false;
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

//! Get the settings object
HwFreedom * Freedom::getSettings()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();

    HwFreedom *hwFreedom = HwFreedom::GetInstance(uavoManager);
    Q_ASSERT(hwFreedom);

    return hwFreedom;
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

/**
 * Get the RFM22b device ID this modem
 * @return RFM22B device ID or 0 if not supported
 */
quint32 Freedom::getRfmID()
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
bool Freedom::bindRadio(quint32 id, quint32 baud_rate, float rf_power,
                        Core::IBoardType::LinkMode linkMode, quint8 min, quint8 max)
{
    HwFreedom::DataFields settings = getSettings()->getData();

    settings.CoordID = id;

    switch(baud_rate) {
    case 9600:
        settings.MaxRfSpeed = HwFreedom::MAXRFSPEED_9600;
        break;
    case 19200:
        settings.MaxRfSpeed = HwFreedom::MAXRFSPEED_19200;
        break;
    case 32000:
        settings.MaxRfSpeed = HwFreedom::MAXRFSPEED_32000;
        break;
    case 64000:
        settings.MaxRfSpeed = HwFreedom::MAXRFSPEED_64000;
        break;
    case 100000:
        settings.MaxRfSpeed = HwFreedom::MAXRFSPEED_100000;
        break;
    case 192000:
        settings.MaxRfSpeed = HwFreedom::MAXRFSPEED_192000;
        break;
    }

    // Round to an integer to use a switch statement
    quint32 rf_power_100 = rf_power * 100;
    switch(rf_power_100) {
    case 0:
        settings.MaxRfPower = HwFreedom::MAXRFPOWER_0;
        break;
    case 125:
        settings.MaxRfPower = HwFreedom::MAXRFPOWER_125;
        break;
    case 160:
        settings.MaxRfPower = HwFreedom::MAXRFPOWER_16;
        break;
    case 316:
        settings.MaxRfPower = HwFreedom::MAXRFPOWER_316;
        break;
    case 630:
        settings.MaxRfPower = HwFreedom::MAXRFPOWER_63;
        break;
    case 1260:
        settings.MaxRfPower = HwFreedom::MAXRFPOWER_126;
        break;
    case 2500:
        settings.MaxRfPower = HwFreedom::MAXRFPOWER_25;
        break;
    case 5000:
        settings.MaxRfPower = HwFreedom::MAXRFPOWER_50;
        break;
    case 10000:
        settings.MaxRfPower = HwFreedom::MAXRFPOWER_100;
        break;
    }

    switch(linkMode) {
    case Core::IBoardType::LINK_TELEM:
        settings.Radio = HwFreedom::RADIO_TELEM;
        break;
    case Core::IBoardType::LINK_TELEM_PPM:
        settings.Radio = HwFreedom::RADIO_TELEMPPM;
        break;
    case Core::IBoardType::LINK_PPM:
        settings.Radio = HwFreedom::RADIO_PPM;
        break;
    }

    settings.MinChannel = min;
    settings.MaxChannel = max;

    getSettings()->setData(settings);
    uavoUtilManager->saveObjectToFlash(getSettings());

    return true;
}

