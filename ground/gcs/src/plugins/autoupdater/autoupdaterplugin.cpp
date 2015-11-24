/**
 ******************************************************************************
 * @file       autoupdaterplugin.c
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

#include "autoupdaterplugin.h"

#include <coreplugin/icore.h>
#include <coreplugin/connectionmanager.h>
#include <QtPlugin>
#include "uavobjectmanager.h"

AutoUpdaterPlugin::AutoUpdaterPlugin() : updater(NULL), usePreRelease(false), refreshInterval(600), gitHubAPIUrl("https://api.github.com/repos/taulabs/taulabs/")
{
}

AutoUpdaterPlugin::~AutoUpdaterPlugin()
{

}
/**
  * Called once all the plugins which depend on us have been loaded
  */
void AutoUpdaterPlugin::extensionsInitialized()
{

}

/**
  * Called at startup, before any plugin which depends on us is initialized
  */
bool AutoUpdaterPlugin::initialize(const QStringList & arguments, QString * errorString)
{
    Q_UNUSED(arguments);
    Q_UNUSED(errorString);
    Core::ICore::instance()->readSettings(this);
    ExtensionSystem::PluginManager* pm = ExtensionSystem::PluginManager::instance();
    options = new AutoUpdaterOptionsPage(this);
    addAutoReleasedObject(options);
    UAVObjectManager * objMngr = pm->getObject<UAVObjectManager>();
    updater = new AutoUpdater(Core::ICore::instance()->mainWindow(), objMngr, refreshInterval, usePreRelease, gitHubAPIUrl, gitHubUsername, gitHubAPIUrl);
    addAutoReleasedObject(updater);
    return true;
}

void AutoUpdaterPlugin::shutdown()
{

}

void AutoUpdaterPlugin::readConfig(QSettings *qSettings, UAVConfigInfo *configInfo)
{
    Q_UNUSED(configInfo)
    qSettings->beginGroup(QLatin1String("General Settings"));
    refreshInterval = (qSettings->value(QLatin1String("AutoUpdateInterval"), refreshInterval).toInt());
    usePreRelease = (qSettings->value(QLatin1String("AutoUpdateUsePreRelease"), usePreRelease).toBool());
    gitHubAPIUrl = (qSettings->value(QLatin1String("AutoUpdateGitHubURL"), gitHubAPIUrl).toString());
    gitHubUsername = (qSettings->value(QLatin1String("AutoUpdateGitHubUsername"), gitHubUsername).toString());
    gitHubPassword = (qSettings->value(QLatin1String("AutoUpdateGitHubPassword"), gitHubPassword).toString());
    qSettings->endGroup();
}
void AutoUpdaterPlugin::updateSettings()
{
    Core::ICore::instance()->saveSettings(this);
}
QString AutoUpdaterPlugin::getGitHubPassword() const
{
    return gitHubPassword;
}

void AutoUpdaterPlugin::setGitHubPassword(const QString &value)
{
    gitHubPassword = value;
}

QString AutoUpdaterPlugin::getGitHubUsername() const
{
    return gitHubUsername;
}

void AutoUpdaterPlugin::setGitHubUsername(const QString &value)
{
    gitHubUsername = value;
}

QString AutoUpdaterPlugin::getGitHubAPIUrl() const
{
    return gitHubAPIUrl;
}

void AutoUpdaterPlugin::setGitHubAPIUrl(const QString &value)
{
    gitHubAPIUrl = value;
}

int AutoUpdaterPlugin::getRefreshInterval() const
{
    return refreshInterval;
}

void AutoUpdaterPlugin::setRefreshInterval(int value)
{
    refreshInterval = value;
}

bool AutoUpdaterPlugin::getUsePreRelease() const
{
    return usePreRelease;
}

void AutoUpdaterPlugin::setUsePreRelease(bool value)
{
    usePreRelease = value;
}

void AutoUpdaterPlugin::saveConfig(QSettings *qSettings, UAVConfigInfo *configInfo)
{
    Q_UNUSED(configInfo)
    qSettings->beginGroup(QLatin1String("General Settings"));
    qSettings->setValue(QLatin1String("AutoUpdateInterval"), refreshInterval);
    qSettings->setValue(QLatin1String("AutoUpdateUsePreRelease"), usePreRelease);
    qSettings->setValue(QLatin1String("AutoUpdateGitHubURL"), gitHubAPIUrl);
    qSettings->setValue(QLatin1String("AutoUpdateGitHubUsername"), gitHubUsername);
    qSettings->setValue(QLatin1String("AutoUpdateGitHubPassword"), gitHubPassword);
    qSettings->endGroup();
    Q_ASSERT(updater);
}
