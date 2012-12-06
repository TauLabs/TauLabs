/**
 ******************************************************************************
 * @file       uavtalkrelayplugin.h
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
#ifndef UAVTALKRELAYPLUGIN_H
#define UAVTALKRELAYPLUGIN_H

#include "uavtalkrelay.h"

#include <extensionsystem/iplugin.h>
#include <extensionsystem/pluginmanager.h>
#include <coreplugin/iconfigurableplugin.h>
#include "uavobjectmanager.h"
#include "uavtalkrelay_global.h"
#include <QHash>
#include "uavtalkrelayoptionspage.h"
class UAVTALKRELAY_EXPORT UavTalkRelayPlugin: public Core::IConfigurablePlugin
{
    Q_OBJECT
    friend class UavTalkRelayOptionsPage;
public:

    UavTalkRelayPlugin();
    ~UavTalkRelayPlugin();

    void extensionsInitialized();
    bool initialize(const QStringList & arguments, QString * errorString);
    void shutdown();
    void readConfig( QSettings* qSettings, Core::UAVConfigInfo *configInfo);
    void saveConfig( QSettings* qSettings, Core::UAVConfigInfo *configInfo);
protected slots:
    void onDeviceConnect(QIODevice *dev);
    void onDeviceDisconnect();
    void updateSettings();
private:
    QHash<QString,QHash<quint32,UavTalkRelay::accessType> > rules;
    ExtensionSystem::PluginManager* plMngr;
    UavTalkRelay * relay;
    UavTalkRelayOptionsPage * mop;
    QString m_IpAddress;
    int m_Port;
    UavTalkRelay::accessType m_DefaultRule;
};

#endif // UAVTALKPRELAYLUGIN_H
