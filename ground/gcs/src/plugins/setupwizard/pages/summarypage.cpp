/**
 ******************************************************************************
 *
 * @file       summarypage.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup SetupWizard Setup Wizard
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

#include "summarypage.h"
#include "ui_summarypage.h"
#include "setupwizard.h"
#include "connectiondiagram.h"

SummaryPage::SummaryPage(SetupWizard *wizard, QWidget *parent) :
    AbstractWizardPage(wizard, parent),
    ui(new Ui::SummaryPage)
{
    ui->setupUi(this);
    connect(ui->illustrationButton, SIGNAL(clicked()), this, SLOT(showDiagram()));
}

SummaryPage::~SummaryPage()
{
    delete ui;
}

bool SummaryPage::validatePage()
{
    return true;
}

void SummaryPage::initializePage()
{
    ui->configurationSummary->setText(getWizard()->getSummaryText());
}

void SummaryPage::showDiagram()
{
    ConnectionDiagram diagram(this, getWizard());

    diagram.exec();
}
