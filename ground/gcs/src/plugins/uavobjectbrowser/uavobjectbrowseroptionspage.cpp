/**
 ******************************************************************************
 *
 * @file       uavobjectbrowseroptionspage.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVObjectBrowserPlugin UAVObject Browser Plugin
 * @{
 * @brief The UAVObject Browser gadget plugin
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

#include "uavobjectbrowseroptionspage.h"
#include "uavobjectbrowserconfiguration.h"
#include <QLabel>
#include <QSpinBox>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QColorDialog>

#include "ui_uavobjectbrowseroptionspage.h"


UAVObjectBrowserOptionsPage::UAVObjectBrowserOptionsPage(UAVObjectBrowserConfiguration *config, QObject *parent) :
    IOptionsPage(parent),
    m_config(config)
{
}

QWidget *UAVObjectBrowserOptionsPage::createPage(QWidget *parent)
{
    m_page = new Ui::UAVObjectBrowserOptionsPage();
    QWidget *w = new QWidget(parent);
    m_page->setupUi(w);

    m_page->recentlyUpdatedButton->setColor(m_config->recentlyUpdatedColor());
    m_page->manuallyChangedButton->setColor(m_config->manuallyChangedColor());
    m_page->notPresentOnHwButton->setColor(m_config->notPresentOnHwColor());
    m_page->recentlyUpdatedTimeoutSpinBox->setValue(m_config->recentlyUpdatedTimeout());
    m_page->highlightBox->setChecked(m_config->onlyHighlightChangedValues());

    return w;

}

void UAVObjectBrowserOptionsPage::apply()
{
    m_config->setRecentlyUpdatedColor(m_page->recentlyUpdatedButton->color());
    m_config->setManuallyChangedColor(m_page->manuallyChangedButton->color());
    m_config->setNotPresentOnHwColor(m_page->notPresentOnHwButton->color());
    m_config->setRecentlyUpdatedTimeout(m_page->recentlyUpdatedTimeoutSpinBox->value());
    m_config->setOnlyHighlightChangedValues(m_page->highlightBox->isChecked());
}

void UAVObjectBrowserOptionsPage::finish()
{
    delete m_page;
}
