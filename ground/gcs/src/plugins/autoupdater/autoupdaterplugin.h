/**
 ******************************************************************************
 * @file       autoupdaterplugin.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup AutoUpdater plugin
 * @{
 *
 * @brief Auto updates the GCS from GitHub releases
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
#ifndef AUTOUPDATERPLUGIN_H
#define AUTOUPDATERPLUGIN_H

#include "autoupdater.h"

#include <extensionsystem/iplugin.h>
#include <extensionsystem/pluginmanager.h>
#include <coreplugin/iconfigurableplugin.h>
#include "uavobjectmanager.h"
#include "autoupdater_global.h"
#include <QHash>
#include "autoupdateroptionspage.h"

class AUTOUPDATER_EXPORT AutoUpdaterPlugin: public Core::IConfigurablePlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "TauLabs.plugins.AutoUpdater" FILE "autoupdater.json")
    friend class AutoUpdaterOptionsPage;
public:

    AutoUpdaterPlugin();
    ~AutoUpdaterPlugin();

    void extensionsInitialized();
    bool initialize(const QStringList & arguments, QString * errorString);
    void shutdown();
    void readConfig( QSettings* qSettings, Core::UAVConfigInfo *configInfo);
    void saveConfig( QSettings* qSettings, Core::UAVConfigInfo *configInfo);
    bool getUsePreRelease() const;
    void setUsePreRelease(bool value);

    int getRefreshInterval() const;
    void setRefreshInterval(int value);

    QString getGitHubAPIUrl() const;
    void setGitHubAPIUrl(const QString &value);

    QString getGitHubUsername() const;
    void setGitHubUsername(const QString &value);

    QString getGitHubPassword() const;
    void setGitHubPassword(const QString &value);

protected slots:
    void updateSettings();
private:
    ExtensionSystem::PluginManager* plMngr;
    AutoUpdater * updater;
    AutoUpdaterOptionsPage * options;
    bool usePreRelease;
    int refreshInterval;
    QString gitHubAPIUrl;
    QString gitHubUsername;
    QString gitHubPassword;
};

#endif // AUTOUPDATERPLUGIN_H
