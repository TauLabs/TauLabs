/**
 ******************************************************************************
 * @file       devicedescriptorstruct.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @see        The GNU Public License (GPL) Version 3
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVObjectUtilPlugin UAVObjectUtil Plugin
 * @{
 * @brief      The UAVUObjectUtil GCS plugin
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

#ifndef DEVICEDESCRIPTORSTRUCT_H
#define DEVICEDESCRIPTORSTRUCT_H

#include "uavobjectutil_global.h"
#include <QString>
#include <QPixmap>

class UAVOBJECTUTIL_EXPORT deviceDescriptorStruct
{
public:
    QString gitHash;
    QString gitDate;
    QString gitTag;
    QByteArray fwHash;
    QByteArray uavoHash;
    int boardType;
    int boardRevision;
    static QString idToBoardName(quint16 id);
    static QPixmap idToBoardPicture(quint16 id);

    deviceDescriptorStruct();
};

#endif // DEVICEDESCRIPTORSTRUCT_H
