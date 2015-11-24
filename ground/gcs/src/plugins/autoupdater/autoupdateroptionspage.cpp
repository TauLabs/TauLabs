/**
 ******************************************************************************
 * @file       autoupdateroptionspage.c
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
#include "autoupdateroptionspage.h"
#include <QLabel>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QRadioButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include "ui_autoupdateroptionspage.h"

AutoUpdaterOptionsPage::AutoUpdaterOptionsPage(QObject *parent) :
    IOptionsPage(parent)
{
    m_config = qobject_cast<AutoUpdaterPlugin *>(parent);
    connect(this, SIGNAL(settingsUpdated()), m_config, SLOT(updateSettings()));
}
AutoUpdaterOptionsPage::~AutoUpdaterOptionsPage()
{
}
QWidget *AutoUpdaterOptionsPage::createPage(QWidget *parent)
{
    m_page = new Ui::AutoUpdaterOptionsPage();
    QWidget *w = new QWidget(parent);
    m_page->setupUi(w);
    m_page->gitHubAPIUrl->setText(m_config->getGitHubAPIUrl());
    m_page->usePreReleases->setChecked(m_config->getUsePreRelease());
    m_page->interval->setValue(m_config->getRefreshInterval());
    m_page->gitHubUsername->setText(m_config->getGitHubUsername());
    m_page->gitHubPassword->setText(m_config->getGitHubPassword());
    return w;
}

void AutoUpdaterOptionsPage::apply()
{
    m_config->setGitHubAPIUrl(m_page->gitHubAPIUrl->text());
    m_config->setRefreshInterval(m_page->interval->value());
    m_config->setUsePreRelease(m_page->usePreReleases->isChecked());
    m_config->setGitHubUsername(m_page->gitHubUsername->text());
    m_config->setGitHubPassword(m_page->gitHubPassword->text());
    emit settingsUpdated();
}

void AutoUpdaterOptionsPage::finish()
{
    delete m_page;
}
