/**
 ******************************************************************************
 *
 * @file       uavtalkplugin.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVTalkPlugin UAVTalk Plugin
 * @{
 * @brief The UAVTalk protocol plugin
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
#include "uavtalkrelayplugin.h"

#include <coreplugin/icore.h>
#include <coreplugin/connectionmanager.h>
#include <QtPlugin>
#include "uavtalkrelayoptionspage.h"
#include "uavobjectmanager.h"
static const QString VERSION = "1.0.0";

UavTalkRelayPlugin::UavTalkRelayPlugin():m_IpAddress(""),m_Port(2000)
{


}

UavTalkRelayPlugin::~UavTalkRelayPlugin()
{

}
/**
  * Called once all the plugins which depend on us have been loaded
  */
void UavTalkRelayPlugin::extensionsInitialized()
{
}

/**
  * Called at startup, before any plugin which depends on us is initialized
  */
bool UavTalkRelayPlugin::initialize(const QStringList & arguments, QString * errorString)
{
    // Done
    Q_UNUSED(arguments);
    Q_UNUSED(errorString);
    // Get UAVObjectManager instance
    ExtensionSystem::PluginManager* pm = ExtensionSystem::PluginManager::instance();
    mop = new UavTalkRelayOptionsPage(this);
    addAutoReleasedObject(mop);
    UAVObjectManager * objMngr = pm->getObject<UAVObjectManager>();

    relay=new UavTalkRelay(objMngr,IpAdress(),Port(),UseTCP());


    return true;
}

void UavTalkRelayPlugin::shutdown()
{

}

void UavTalkRelayPlugin::readConfig(QSettings *qSettings, UAVConfigInfo *configInfo)
{
    Q_UNUSED(configInfo)
    qSettings->beginGroup(QLatin1String("UavTalkRelay"));

    qSettings->beginReadArray("Current");
    qSettings->setArrayIndex(0);
    m_IpAddress = (qSettings->value(QLatin1String("IpAddress"), tr("")).toString());
    m_Port = (qSettings->value(QLatin1String("Port"), tr("")).toInt());
    m_UseTCP = (qSettings->value(QLatin1String("UseTCP"), tr("")).toInt());
    qSettings->endArray();
    qSettings->endGroup();
}

void UavTalkRelayPlugin::saveConfig(QSettings *qSettings, UAVConfigInfo *configInfo)
{
    Q_UNUSED(configInfo)
    qSettings->beginGroup(QLatin1String("UavTalkRelay"));

    qSettings->beginWriteArray("Current");
    qSettings->setArrayIndex(0);
    qSettings->setValue(QLatin1String("IpAddress"), m_IpAddress);
    qSettings->setValue(QLatin1String("Port"), m_Port);
    qSettings->setValue(QLatin1String("UseTCP"), m_UseTCP);
    qSettings->endArray();
    qSettings->endGroup();
}

void UavTalkRelayPlugin::setPort(int value)
{
}

void UavTalkRelayPlugin::onDeviceConnect(QIODevice *dev)
{
}

void UavTalkRelayPlugin::onDeviceDisconnect()
{
}
void UavTalkRelayPlugin::setUseTCP(bool value)
{
}

void UavTalkRelayPlugin::setIpAdress(QString value)
{
}


Q_EXPORT_PLUGIN(UavTalkRelayPlugin)




