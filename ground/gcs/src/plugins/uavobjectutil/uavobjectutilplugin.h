/**
 ******************************************************************************
 *
 * @file       uavobjectutilplugin.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @see        The GNU Public License (GPL) Version 3
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVObjectUtilPlugin UAVObjectUtil Plugin
 * @{
 * @brief The UAVUObjectUtil GCS plugin
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
#ifndef UAVOBJECTUTILPLUGIN_H
#define UAVOBJECTUTILPLUGIN_H

#include "uavobjectutil_global.h"
#include <extensionsystem/iplugin.h>
#include <QtPlugin>
#include "uavobjectutilmanager.h"

class UAVOBJECTUTIL_EXPORT UAVObjectUtilPlugin: public ExtensionSystem::IPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "TauLabs.plugins.UAVObjectUtil" FILE "UAVObjectUtil.json")

public:
	UAVObjectUtilPlugin();
	~UAVObjectUtilPlugin();

    void extensionsInitialized();
    bool initialize(const QStringList & arguments, QString * errorString);
    void shutdown();
};

#endif
