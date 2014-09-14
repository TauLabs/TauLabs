/**
 ******************************************************************************
 * @file       sparkybgcconfiguration.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_TauLabsPlugin Tau Labs boards support Plugin
 * @{
 * @brief Plugin to support boards by the Tau Labs project
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

#include <QWidget>
#include <QPushButton>
#include <QDesktopServices>
#include <QUrl>
#include <extensionsystem/pluginmanager.h>
#include <coreplugin/generalsettings.h>

#include "sparkybgcconfiguration.h"
#include "ui_sparkybgcconfiguration.h"

SparkyBgcConfiguration::SparkyBgcConfiguration(QWidget *parent) : ConfigTaskWidget(parent),
    ui(new Ui::SparkyBgcConfiguration)
{
    ui->setupUi(this);


    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings * settings=pm->getObject<Core::Internal::GeneralSettings>();
    if(!settings->useExpertMode())
        ui->saveSettings->setVisible(false);

    addApplySaveButtons(ui->saveSettings, ui->applySettings);

    // Load UAVObjects to widget relations from UI file
    // using objrelation dynamic property
    autoLoadWidgets();

    connect(ui->helpButton, SIGNAL(clicked()), this, SLOT(openHelp()));
    enableControls(true);
    populateWidgets();
    refreshWidgetsValues();
    forceConnectedState();
}

SparkyBgcConfiguration::~SparkyBgcConfiguration()
{
    delete ui;
}

void SparkyBgcConfiguration::refreshValues()
{
}

void SparkyBgcConfiguration::widgetsContentsChanged()
{
    ConfigTaskWidget::widgetsContentsChanged();
    enableControls(true);
}

void SparkyBgcConfiguration::openHelp()
{
    QDesktopServices::openUrl( QUrl("http://wiki.taulabs.org/OnlineHelp:-Hardware-Settings", QUrl::StrictMode) );
}
