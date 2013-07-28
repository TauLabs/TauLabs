/**
 ******************************************************************************
 * @file       configmodulewidget.cpp
 * @brief      Configure the optional modules
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
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

#include "configmodulewidget.h"

#include <extensionsystem/pluginmanager.h>
#include <coreplugin/generalsettings.h>


ConfigModuleWidget::ConfigModuleWidget(QWidget *parent) : ConfigTaskWidget(parent)
{
    ui = new Ui::Modules();
    ui->setupUi(this);

    // Create the general modules page
    ModuleSettingsForm *optionalModuleSettings =
            new ModuleSettingsForm(this, ui->saveButton, ui->applyButton, ui->reloadButton);
    QString modulesTabText = ui->tabWidget->tabText(ui->tabWidget->indexOf(ui->general));
    ui->tabWidget->removeTab(ui->tabWidget->indexOf(ui->general));
    ui->tabWidget->addTab(optionalModuleSettings, modulesTabText);
}

ConfigModuleWidget::~ConfigModuleWidget()
{

}

void ConfigModuleWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
}

void ConfigModuleWidget::enableControls(bool enable)
{
    Q_UNUSED(enable);
}
/**
 * @}
 * @}
 */
