/**
 ******************************************************************************
 *
 * @file       iboardtype.h
 * @author     The PhoenixPilot Team, Copyright (C) 2012.
 *             Parts by Nokia Corporation (qt-info@nokia.com) Copyright (C) 2009.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup CorePlugin Core Plugin
 * @{
 * @brief The Core GCS plugin
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
#ifndef IBOARDTYPE_H
#define IBOARDTYPE_H

#include <QObject>
#include <QtCore/QStringList>

#include "core_global.h"

namespace Core {

/**
*   An IBoardType object defines an autopilot or more generally a hardware device,
*   that is supported by the GCS. It provides basic information to the GCS to detect
*   and use this board type.
*
*   Note: at present (2012), the GCS only supports autopilots, and assumes they will
*         talk with UAVTalk. Further down the line, GCS will be able to support additional
*         protocols, as well as other device types (beacons, other).
*/
class CORE_EXPORT IBoardType : public QObject
{
    Q_OBJECT
public:

    /**
     * Short description of the board / friendly name
     */
    virtual QString shortName() = 0;

    /**
     * Long description of the board
     */
    virtual QString boardDescription() = 0;

    /**
     * Get supported protocol(s) for this board
     *
     * TODO: extend GCS to support multiple protocol types.
     */
    virtual QStringList getSupportedProtocols() = 0;

    /**
     * @brief The USBInfo struct
     * TODO: finalize what we will put there, not everything
     *       is relevant.
     */
    struct USBInfo {
        QString serialNumber;
        QString manufacturer;
        QString product;
        int UsagePage;
        int Usage;
        int vendorID;
        int productID;
        int bcdDevice;
    };

    /**
     * Get USB descriptors to detect the board
     */
    USBInfo getUSBInfo() { return boardUSBInfo; }

    /**
     * Does this board support the bootloader and DFU protocol ?
     */
    bool isDFUSupported() { return dfuSupport; }

signals:

private:
    USBInfo boardUSBInfo;
    bool dfuSupport;
};

} //namespace Core


#endif // IBOARDTYPE_H
