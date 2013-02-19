/**
 ******************************************************************************
 *
 * @file       sysalarmsmessagingplugin.cpp
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

#include "sysalarmsmessagingplugin.h"

#include <extensionsystem/pluginmanager.h>
#include <coreplugin/icore.h>
#include "systemalarms.h"
#include <QtCore/QtPlugin>
#include <QtGui/QMainWindow>

#include <QDebug>

SysAlarmsMessagingPlugin::SysAlarmsMessagingPlugin()
{
}

SysAlarmsMessagingPlugin::~SysAlarmsMessagingPlugin()
{
}

void SysAlarmsMessagingPlugin::extensionsInitialized()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    SystemAlarms* obj = dynamic_cast<SystemAlarms*>(objManager->getObject(QString("SystemAlarms")));
    connect(obj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(updateAlarms(UAVObject*)));

}

/**
 * @brief SysAlarmsMessagingPlugin::initialize Create a warning and error message for each
 * field in the SystemAlarms object
 * @return True
 */
bool SysAlarmsMessagingPlugin::initialize(const QStringList &arguments, QString *errorString)
{
    Q_UNUSED(arguments);
    Q_UNUSED(errorString);
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    SystemAlarms* obj = SystemAlarms::GetInstance(objManager);

    foreach (UAVObjectField *field, obj->getFields()) {
        for (uint i = 0; i < field->getNumElements(); ++i) {
            QString element = field->getElementNames()[i];
            GlobalMessage * msg=Core::ICore::instance()->globalMessaging()->addErrorMessage(element,"");
            msg->setActive(false);
            errorMessages.insert(element,msg);
            msg=Core::ICore::instance()->globalMessaging()->addWarningMessage(element,"");
            msg->setActive(false);
            warningMessages.insert(element,msg);
        }
    }
    TelemetryManager* telMngr = pm->getObject<TelemetryManager>();
    connect(telMngr, SIGNAL(disconnected()), this, SLOT(onAutopilotDisconnect()));
    return true;
}

/**
 * @brief SysAlarmsMessagingPlugin::updateAlarms Called whenever the SystemAlarms
 * object is updated to set the alarm messages appropriately.
 * @param[in] systemAlarm Must be the SystemAlarms object
 */
void SysAlarmsMessagingPlugin::updateAlarms(UAVObject* systemAlarm)
{
    Q_ASSERT(systemAlarm->getObjID() == SystemAlarms::OBJID);

    foreach (UAVObjectField *field, systemAlarm->getFields()) {
        for (uint i = 0; i < field->getNumElements(); ++i) {
            QString element = field->getElementNames()[i];
            QString value = field->getValue(i).toString();
            {
                if(value==field->getOptions().at(SystemAlarms::ALARM_ERROR))
                {
                    errorMessages.value(element)->setActive(true);
                    errorMessages.value(element)->setDescription(element+" module is in error state");
                    warningMessages.value(element)->setActive(false);
                }
                else if(value==field->getOptions().at(SystemAlarms::ALARM_CRITICAL))
                {
                    errorMessages.value(element)->setActive(true);
                    errorMessages.value(element)->setDescription(element+" module is in CRITICAL state");
                    warningMessages.value(element)->setActive(false);
                }
                else if(value==field->getOptions().at(SystemAlarms::ALARM_WARNING))
                {
                    warningMessages.value(element)->setActive(true);
                    warningMessages.value(element)->setDescription(element+" module is in warning state");
                    errorMessages.value(element)->setActive(false);
                }
                else if(value==field->getOptions().at(SystemAlarms::ALARM_UNINITIALISED))
                {
                    warningMessages.value(element)->setActive(false);
                    errorMessages.value(element)->setActive(false);
                }
                else if(value==field->getOptions().at(SystemAlarms::ALARM_OK))
                {
                    warningMessages.value(element)->setActive(false);
                    errorMessages.value(element)->setActive(false);
                }
            }
        }
    }
}

void SysAlarmsMessagingPlugin::onAutopilotDisconnect()
{
    foreach(GlobalMessage * msg,errorMessages.values())
        msg->setActive(false);
    foreach(GlobalMessage * msg,warningMessages.values())
        msg->setActive(false);
}

Q_EXPORT_PLUGIN(SysAlarmsMessagingPlugin)
