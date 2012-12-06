/**
 ******************************************************************************
 * @file       uavtalkrelayplugin.c
 * @author     The PhoenixPilot Team, http://github.com/PhoenixPilot
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVTalk relay plugin
 * @{
 *
 * @brief Relays UAVTalk data trough UDP to another GCS
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

#include "uavobjectmanager.h"
static const QString VERSION = "1.0.0";

UavTalkRelayPlugin::UavTalkRelayPlugin():m_IpAddress(""),m_Port(2000),m_DefaultRule(UavTalkRelay::ReadWrite),relay(0)
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
    Q_UNUSED(arguments);
    Q_UNUSED(errorString);
    Core::ICore::instance()->readSettings(this);
    ExtensionSystem::PluginManager* pm = ExtensionSystem::PluginManager::instance();
    mop = new UavTalkRelayOptionsPage(this);
    addAutoReleasedObject(mop);
    UAVObjectManager * objMngr = pm->getObject<UAVObjectManager>();
    relay=new UavTalkRelay(objMngr,m_IpAddress,m_Port,rules,m_DefaultRule);
    addAutoReleasedObject(relay);
    return true;
}

void UavTalkRelayPlugin::shutdown()
{

}

void UavTalkRelayPlugin::readConfig(QSettings *qSettings, UAVConfigInfo *configInfo)
{
    Q_UNUSED(configInfo)
    qSettings->beginGroup(QLatin1String("General Settings"));
    m_IpAddress = (qSettings->value(QLatin1String("ListeningInterface"),m_IpAddress).toString());
    m_Port = (qSettings->value(QLatin1String("ListeningPort"), m_Port).toInt());
    m_DefaultRule=(UavTalkRelay::accessType)qSettings->value(QLatin1String("Defaul Rule"),m_DefaultRule).toInt();
    qSettings->endGroup();
    int size=qSettings->beginReadArray(QLatin1String("Rule"));
    for(int i=0;i < size;++i)
    {
        qSettings->setArrayIndex(i);
        QString host=qSettings->value(QLatin1String("Host"),tr("")).toString();
        quint32 uavo=qSettings->value(QLatin1String("UAVO"),tr("")).toInt();
        UavTalkRelay::accessType aType=(UavTalkRelay::accessType)qSettings->value(QLatin1String("Access Type"),tr("")).toInt();
        if(rules.keys().contains(host))
        {
           QHash <quint32,UavTalkRelay::accessType> temp=rules.value(host);
           temp.insert(uavo,aType);
        }
        else
        {
            QHash <quint32,UavTalkRelay::accessType> temp;
            temp.insert(uavo,aType);
            rules.insert(host,temp);
        }
    }
    qSettings->endArray();
}
void UavTalkRelayPlugin::updateSettings()
{
    Core::ICore::instance()->saveSettings(this);
}

void UavTalkRelayPlugin::saveConfig(QSettings *qSettings, UAVConfigInfo *configInfo)
{
    Q_UNUSED(configInfo)
    qSettings->remove("Rule");
    qSettings->beginGroup(QLatin1String("General Settings"));
    qSettings->setValue(QLatin1String("ListeningInterface"), m_IpAddress);
    qSettings->setValue(QLatin1String("ListeningPort"), m_Port);
    qSettings->setValue(QLatin1String("Default Rule"),m_DefaultRule);
    qSettings->endGroup();
    qSettings->beginWriteArray(QLatin1String("Rule"));
    int index=0;
    foreach(QString host,rules.keys())
    {
        qSettings->setArrayIndex(index);
        ++index;
        foreach(quint32 uavo,rules.value(host).keys())
        {
            qSettings->setValue(QLatin1String("Host"),host);
            qSettings->setValue(QLatin1String("UAVO"),uavo);
            qSettings->setValue(QLatin1String("Access Type"),rules.value(host).value(uavo));
        }
    }
    qSettings->endArray();
    Q_ASSERT(relay);
    relay->setIpAdress(m_IpAddress);
    relay->setPort(m_Port);
    relay->setRules(rules);
    relay->restartServer();
}

void UavTalkRelayPlugin::onDeviceConnect(QIODevice *dev)
{
}

void UavTalkRelayPlugin::onDeviceDisconnect()
{
}

Q_EXPORT_PLUGIN(UavTalkRelayPlugin)
