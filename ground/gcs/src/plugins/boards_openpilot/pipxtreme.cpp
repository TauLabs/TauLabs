/**
 ******************************************************************************
 *
 * @file       pipxtreme.cpp
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

#include "pipxtreme.h"

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>

#include "hwtaulink.h"
#include "rfm22bstatus.h"

/**
 * @brief PipXtreme::PipXtreme
 *  This is the PipXtreme radio modem definition
 */
PipXtreme::PipXtreme(void)
{
    // Initialize our USB Structure definition here:
    USBInfo board;
    board.vendorID = 0x20A0;
    board.productID = 0x415c;

    setUSBInfo(board);

    boardType = 0x03;
}

PipXtreme::~PipXtreme()
{

}


QString PipXtreme::shortName()
{
    return QString("PipXtreme");
}

QString PipXtreme::boardDescription()
{
    return QString("The OpenPilot project PipXtreme RF radio modem");
}

//! Return which capabilities this board has
bool PipXtreme::queryCapabilities(BoardCapabilities capability)
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
    }
    return false;
}

/**
 * @brief PipXtreme::getSupportedProtocols
 *  TODO: this is just a stub, we'll need to extend this a lot with multi protocol support
 *  TODO: for the PipXtreme, depending on its configuration, it might offer several protocols (uavtalk and raw)
 * @return
 */
QStringList PipXtreme::getSupportedProtocols()
{

    return QStringList("uavtalk");
}

QPixmap PipXtreme::getBoardPicture()
{
    return QPixmap(":/images/pipx.png");
}

QString PipXtreme::getHwUAVO()
{
    return "PipXtreme";
}

/**
 * Get the RFM22b device ID this modem
 * @return RFM22B device ID or 0 if not supported
 */
quint32 PipXtreme::getRfmID()
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
bool PipXtreme::setCoordID(quint32 id, quint32 baud_rate, float rf_power)
{
    Q_UNUSED(baud_rate);
    Q_UNUSED(rf_power);

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();

    HwTauLink *hwTauLink = HwTauLink::GetInstance(uavoManager);
    Q_ASSERT(hwTauLink);
    HwTauLink::DataFields settings = hwTauLink->getData();

    if (id == 0) {
        // set as coordinator
        settings.Radio = HwTauLink::RADIO_TELEMCOORD;
        settings.CoordID = 0;
    } else {
        settings.Radio = HwTauLink::RADIO_TELEM;
        settings.CoordID = id;
    }

    hwTauLink->setData(settings);

    // save the settings
    UAVObjectUtilManager* uavoUtilManager = pm->getObject<UAVObjectUtilManager>();
    uavoUtilManager->saveObjectToFlash(hwTauLink);

    return true;
}
