/**
 ******************************************************************************
 * @file       devicedescriptorstruct.cpp
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

#include "devicedescriptorstruct.h"
#include "uavobjectutilmanager.h"
#include <QList>
#include <QString>
#include <QDebug>

deviceDescriptorStruct::deviceDescriptorStruct()
{

}

//! Get the name for a board via the plugin system
QString deviceDescriptorStruct::idToBoardName(quint16 id)
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    if (pm == NULL)
        return "Unknown";

    QList <Core::IBoardType *> boards = pm->getObjects<Core::IBoardType>();
    foreach (Core::IBoardType *board, boards) {
        if (board->getBoardType() == (id >> 8))
            return board->shortName();
    }

    return "Unknown";
}

QPixmap deviceDescriptorStruct::idToBoardPicture(quint16 id)
{
    qDebug() << "Getting board picture";
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    if (pm == NULL)
        return QPixmap();

    QList <Core::IBoardType *> boards = pm->getObjects<Core::IBoardType>();
    foreach (Core::IBoardType *board, boards) {
        if (board->getBoardType() == (id >> 8)) {
            qDebug() << "Found board. " << board->getBoardPicture().isNull();
            return board->getBoardPicture();
        }
    }

    qDebug() << "Not found";
    return QPixmap();
}
