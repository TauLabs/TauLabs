/**
 ******************************************************************************
 *
 * @file       navwizardplugin.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup NavWizard Setup Wizard
 * @{
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
#include "navwizardplugin.h"

#include <QDebug>
#include <QtPlugin>
#include <QStringList>
#include <extensionsystem/pluginmanager.h>

#include <coreplugin/coreconstants.h>
#include <coreplugin/actionmanager/actionmanager.h>
#include <coreplugin/icore.h>
#include <QKeySequence>
#include <coreplugin/modemanager.h>

NavWizardPlugin::NavWizardPlugin() : wizardRunning(false)
{}

NavWizardPlugin::~NavWizardPlugin()
{}

bool NavWizardPlugin::initialize(const QStringList & args, QString *errMsg)
{
    Q_UNUSED(args);
    Q_UNUSED(errMsg);

    // Add Menu entry
    Core::ActionManager *am   = Core::ICore::instance()->actionManager();
    Core::ActionContainer *ac = am->actionContainer(Core::Constants::M_TOOLS);

    // Add entry points for navigation setup wizard
    Core::Command *cmd = am->registerAction(new QAction(this),
                                        "NavWizardPlugin.ShowNavigationWizard",
                                        QList<int>() <<
                                        Core::Constants::C_GLOBAL_ID);
    cmd->action()->setText(tr("Navigation Setup Wizard"));

    Core::ModeManager::instance()->addAction(cmd, 1);

    ac->menu()->addSeparator();
    ac->appendGroup("Wizard");
    ac->addAction(cmd, "Wizard");
    ac->addAction(cmd, "Navigation Wizard");

    connect(cmd->action(), SIGNAL(triggered(bool)), this, SLOT(showNavigationWizard()));

    return true;
}

void NavWizardPlugin::extensionsInitialized()
{}

void NavWizardPlugin::shutdown()
{}


void NavWizardPlugin::showNavigationWizard()
{
    if (!wizardRunning) {
        wizardRunning = true;
        NavigationWizard *m_wiz = new NavigationWizard();
        connect(m_wiz, SIGNAL(finished(int)), this, SLOT(wizardTerminated()));
        m_wiz->setAttribute(Qt::WA_DeleteOnClose, true);
        m_wiz->setWindowFlags(m_wiz->windowFlags() | Qt::WindowStaysOnTopHint);
        m_wiz->show();
    }
}

void NavWizardPlugin::wizardTerminated()
{
    wizardRunning = false;
    disconnect(this, SLOT(wizardTerminated()));
}

Q_EXPORT_PLUGIN(NavWizardPlugin)
