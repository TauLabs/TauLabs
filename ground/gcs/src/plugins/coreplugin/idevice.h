/**
 ******************************************************************************
 *
 * @file       idevice.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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
#ifndef IDEVICE_H
#define IDEVICE_H

#include <QObject>
#include "core_global.h"


namespace Core {

/**
*   The IDevice interface class is used by the connection manager and plugins that
* provide connections to contain plugin-specific info for a given device: for a USB
* device, that will be PID/VID/Serial number, for serial devices, that will be the
* port name, etc.
*
*  As such, this interface is minimal - only short and display name - , it is up to
*  the plugins to define additional methods/properties they need and use dynamic
*  casting when getting device objects back from the connection manager.
*
*/
class CORE_EXPORT IDevice : public QObject
{
    Q_OBJECT
public:


    QString getName() const { return name; }
    void setName(QString theName) { name = theName; }
    QString getDisplayName() const { return displayName; }
    void setDisplayName( QString dn ) { displayName = dn; }

    /*
    bool operator==(const IDevice *idv) const {
        return name == idv->getName()  && displayName == idv->getDisplayName();
    }
    */

    bool equals(const IDevice *idv) const {
        return name == idv->getName()  && displayName == idv->getDisplayName();
    }

private:
    QString name;
    QString displayName;


};

}

#endif // IDEVICE_H
