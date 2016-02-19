/**
 ******************************************************************************
 * @file       taulink.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
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

#include "taulink.h"

#include <uavobjectmanager.h>
#include <extensionsystem/pluginmanager.h>

#include "rfm22bstatus.h"

/**
 * @brief TauLink::TauLink
 *  This is the PipXtreme radio modem definition
 */
TauLink::TauLink(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415c;

    setUSBInfo(board);

    boardType = 0x03;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    uavoUtilManager = pm->getObject<UAVObjectUtilManager>();
}

TauLink::~TauLink()
{

}


QString TauLink::shortName()
{
    return QString("PipXtreme");
}

QString TauLink::boardDescription()
{
    return QString("The TauLink radio modem");
}

//! Return which capabilities this board has
bool TauLink::queryCapabilities(BoardCapabilities capability)
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
        return true;
    case BOARD_CAPABILITIES_OSD:
        return false;
    }
    return false;
}

/**
 * @brief TauLink::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 *  TODO: for the TauLink, depending on its configuration, it might offer several protocols (uavtalk and raw)
 * @return
 */
QStringList TauLink::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap TauLink::getBoardPicture()
{
    return QPixmap(":/images/taulink.png");
}

QString TauLink::getHwUAVO()
{
    return "HwTauLink";
}

//! Get the settings object
HwTauLink * TauLink::getSettings()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();

    HwTauLink *wwTauLink = HwTauLink::GetInstance(uavoManager);
    Q_ASSERT(wwTauLink);

    return wwTauLink;
}
/**
 * Get the RFM22b device ID this modem
 * @return RFM22B device ID or 0 if not supported
 */
quint32 TauLink::getRfmID()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();

    // Modem has instance 0
    RFM22BStatus *rfm22bStatus = RFM22BStatus::GetInstance(uavoManager,0);
    Q_ASSERT(rfm22bStatus);
    RFM22BStatus::DataFields rfm22b = rfm22bStatus->getData();

    return rfm22b.DeviceID;
}

/**
 * Set the coordinator ID. If set to zero this device will
 * be a coordinator.
 * @return true if successful or false if not
 */
bool TauLink::bindRadio(quint32 id, quint32 baud_rate, float rf_power, Core::IBoardType::LinkMode linkMode, quint8 min, quint8 max)
{
    HwTauLink::DataFields settings = getSettings()->getData();

    settings.CoordID = id;

    switch(baud_rate) {
    case 9600:
        settings.MaxRfSpeed = HwTauLink::MAXRFSPEED_9600;
        break;
    case 19200:
        settings.MaxRfSpeed = HwTauLink::MAXRFSPEED_19200;
        break;
    case 32000:
        settings.MaxRfSpeed = HwTauLink::MAXRFSPEED_32000;
        break;
    case 64000:
        settings.MaxRfSpeed = HwTauLink::MAXRFSPEED_64000;
        break;
    case 100000:
        settings.MaxRfSpeed = HwTauLink::MAXRFSPEED_100000;
        break;
    case 192000:
        settings.MaxRfSpeed = HwTauLink::MAXRFSPEED_192000;
        break;
    }

    // Round to an integer to use a switch statement
    quint32 rf_power_100 = rf_power * 100;
    switch(rf_power_100) {
    case 0:
        settings.MaxRfPower = HwTauLink::MAXRFPOWER_0;
        break;
    case 125:
        settings.MaxRfPower = HwTauLink::MAXRFPOWER_125;
        break;
    case 160:
        settings.MaxRfPower = HwTauLink::MAXRFPOWER_16;
        break;
    case 316:
        settings.MaxRfPower = HwTauLink::MAXRFPOWER_316;
        break;
    case 630:
        settings.MaxRfPower = HwTauLink::MAXRFPOWER_63;
        break;
    case 1260:
        settings.MaxRfPower = HwTauLink::MAXRFPOWER_126;
        break;
    case 2500:
        settings.MaxRfPower = HwTauLink::MAXRFPOWER_25;
        break;
    case 5000:
        settings.MaxRfPower = HwTauLink::MAXRFPOWER_50;
        break;
    case 10000:
        settings.MaxRfPower = HwTauLink::MAXRFPOWER_100;
        break;
    }

    switch(linkMode) {
    case Core::IBoardType::LINK_TELEM:
        settings.Radio = HwTauLink::RADIO_TELEM;
        break;
    case Core::IBoardType::LINK_TELEM_PPM:
        settings.Radio = HwTauLink::RADIO_TELEMPPM;
        break;
    case Core::IBoardType::LINK_PPM:
        settings.Radio = HwTauLink::RADIO_PPM;
        break;
    }

    settings.MinChannel = min;
    settings.MaxChannel = max;

    getSettings()->setData(settings);
    uavoUtilManager->saveObjectToFlash( getSettings());

    return true;
}
