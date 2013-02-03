/**
 ******************************************************************************
 *
 * @file       sysalarmsmessagingplugin.h
 * @author     PhoenixPilot, http://github.com/PhoenixPilot Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup System Alarms Messaging Plugin
 * @{
 * @brief Map fields from the @ref SystemAlarms object to the global messaging
 * system
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

#ifndef SYSALARMSMESSAGINGPLUGIN_H
#define SYSALARMSMESSAGINGPLUGIN_H


#include <extensionsystem/iplugin.h>
#include "uavobject.h"
#include "uavtalk/telemetrymanager.h"
#include <coreplugin/globalmessaging.h>

using namespace Core;

/**
 * @brief The SysAlarmsMessagingPlugin class keeps a mapping from the field
 * names in the @ref SystemAlarms object to @ref GlobalMessage messages and
 * whenever the alarms change updates the messaging system
 */
class SysAlarmsMessagingPlugin
    : public ExtensionSystem::IPlugin
{
    Q_OBJECT

public:
    SysAlarmsMessagingPlugin();
    ~SysAlarmsMessagingPlugin();

    //! Create the message for all of the alarm fields
    virtual bool initialize(const QStringList &arguments, QString *error_message);
    virtual void extensionsInitialized();
private:
    QMap<QString,GlobalMessage*> errorMessages;
    QMap<QString,GlobalMessage*> warningMessages;
private slots:
    void updateAlarms(UAVObject*);
    void onAutopilotDisconnect();

};


#endif
