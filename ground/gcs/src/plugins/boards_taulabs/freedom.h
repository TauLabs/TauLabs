/**
 ******************************************************************************
 * @file       freedom.h
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
#ifndef FREEDOM_H
#define FREEDOM_H

#include "hwfreedom.h"
#include <coreplugin/iboardtype.h>
#include <uavobjectutil/uavobjectutilmanager.h>

class IBoardType;

class Freedom : public Core::IBoardType
{
public:
    Freedom();
    virtual ~Freedom();

    virtual QString shortName();
    virtual QString boardDescription();
    virtual bool queryCapabilities(BoardCapabilities capability);
    virtual QStringList getSupportedProtocols();
    virtual QPixmap getBoardPicture();
    virtual QString getHwUAVO();
    virtual int queryMaxGyroRate();

    HwFreedom * getSettings();

    /**
     * Get the RFM22b device ID this modem
     * @return RFM22B device ID or 0 if not supported
     */
    virtual quint32 getRfmID();

    /**
     * Set the coordinator ID. If set to zero this device will
     * be a coordinator.
     * @return true if successful or false if not
     */
    virtual bool setCoordID(quint32 id, quint32 baud_rate = 0, float rf_power = -1);

    //! Set the radio link mode
    virtual bool setLinkMode(Core::IBoardType::LinkMode /* linkMode */);

    //! Set the minimum and maximum channel index
    virtual bool setMinMaxChannel(quint8 /* min */, quint8 /* max */);

private:
    UAVObjectUtilManager* uavoUtilManager;
};


#endif // FREEDOM_H
