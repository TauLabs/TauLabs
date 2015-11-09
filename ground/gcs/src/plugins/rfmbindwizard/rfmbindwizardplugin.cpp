/**
 ******************************************************************************
 *
 * @file       rfmbindwizardplugin.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup RfmBindWizard RFM22b Binding Wizard
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
#include "rfmbindwizardplugin.h"

#include <QDebug>
#include <QtPlugin>
#include <QStringList>
#include <extensionsystem/pluginmanager.h>

#include <coreplugin/coreconstants.h>
#include <coreplugin/actionmanager/actionmanager.h>
#include <coreplugin/icore.h>
#include <QKeySequence>
#include <coreplugin/modemanager.h>

RfmBindWizardPlugin::RfmBindWizardPlugin() : bindWizardRunning(false)
{}

RfmBindWizardPlugin::~RfmBindWizardPlugin()
{}

bool RfmBindWizardPlugin::initialize(const QStringList & args, QString *errMsg)
{
    Q_UNUSED(args);
    Q_UNUSED(errMsg);

    // Add Menu entry
    Core::ActionManager *am   = Core::ICore::instance()->actionManager();
    Core::ActionContainer *ac = am->actionContainer(Core::Constants::M_TOOLS);

    Core::Command *cmd = am->registerAction(new QAction(this),
                                            "RfmBindWizardPlugin.ShowBindWizard",
                                            QList<int>() <<
                                            Core::Constants::C_GLOBAL_ID);
    cmd->action()->setText(tr("Rfm Bind Wizard"));

    Core::ModeManager::instance()->addAction(cmd, 1);

    ac->menu()->addSeparator();
    ac->appendGroup("Bind Wizard");
    ac->addAction(cmd, "Bind Wizard");

    connect(cmd->action(), SIGNAL(triggered(bool)), this, SLOT(showBindWizard()));
    return true;
}

void RfmBindWizardPlugin::extensionsInitialized()
{}

void RfmBindWizardPlugin::shutdown()
{}

void RfmBindWizardPlugin::showBindWizard()
{
    if (!bindWizardRunning) {
        bindWizardRunning = true;
        RfmBindWizard *m_wiz = new RfmBindWizard();
        connect(m_wiz, SIGNAL(finished(int)), this, SLOT(bindWizardTerminated()));
        m_wiz->setAttribute(Qt::WA_DeleteOnClose, true);
        m_wiz->setWindowFlags(m_wiz->windowFlags() | Qt::WindowStaysOnTopHint);
        m_wiz->show();
    }
}

void RfmBindWizardPlugin::bindWizardTerminated()
{
    bindWizardRunning = false;
    disconnect(this, SLOT(bindWizardTerminated()));
}

/**
 * @}
 * @}
 */
