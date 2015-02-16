/**
 ******************************************************************************
 * @file       sparky2.h
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
#ifndef SPARKY2_H
#define SPARKY2_H

#include "hwsparky2.h"
#include <coreplugin/iboardtype.h>
#include <uavobjectutil/uavobjectutilmanager.h>

class IBoardType;

class Sparky2 : public Core::IBoardType
{
public:
    Sparky2();
    virtual ~Sparky2();

    virtual QString shortName();
    virtual QString boardDescription();
    virtual bool queryCapabilities(BoardCapabilities capability);
    virtual QStringList getSupportedProtocols();
    virtual QPixmap getBoardPicture();
    virtual QString getHwUAVO();
    HwSparky2 * getSettings();

    //! Determine if this board supports configuring the receiver
    virtual bool isInputConfigurationSupported();

    /**
     * Configure the board to use an receiver input type on a port number
     * @param type the type of receiver to use
     * @param port_num which input port to configure (board specific numbering)
     */
    virtual bool setInputOnPort(enum InputType type, int port_num = 0);

    /**
     * @brief getInputOnPort get the current input type
     * @param port_num which input port to query (board specific numbering)
     * @return the currently selected input type
     */
    virtual enum InputType getInputOnPort(int port_num = 0);

    /**
     * @brief getConnectionDiagram get the connection diagram for this board
     * @return a string with the name of the resource for this board diagram
     */
    virtual QString getConnectionDiagram() { return ":/taulabs/images/sparky-connection-diagram.svg"; }

    virtual int queryMaxGyroRate();

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
    virtual bool bindRadio(quint32 id, quint32 baud_rate, float rf_power,
                           Core::IBoardType::LinkMode linkMode, quint8 min, quint8 max);

private:
    UAVObjectUtilManager* uavoUtilManager;
};


#endif // SPARKY2_H
