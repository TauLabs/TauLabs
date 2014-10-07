/**
 ******************************************************************************
 *
 * @file       rfmbindwizard.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup RfmBindWizard Rfm Bind Wizard
 * @{
 * @brief A wizard to help bind the modem and flight controller
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

#include "rfmbindwizard.h"
#include "pages/tlstartpage.h"
#include "pages/tlendpage.h"
#include "pages/coordinatorpage.h"
#include "pages/coordinatedpage.h"

#include "uploader/uploadergadgetfactory.h"

using namespace uploader;

RfmBindWizard::RfmBindWizard(QWidget *parent) : QWizard(parent),
    m_ppm(false), m_maxBps(64000), m_connectionManager(0)
{
    setWindowTitle(tr("RFM22b Binding Wizard"));
    setOption(QWizard::IndependentPages, false);
    setWizardStyle(QWizard::ModernStyle);
    setMinimumSize(600, 500);
    resize(600, 500);
    createPages();
}

int RfmBindWizard::nextId() const
{
    switch (currentId()) {
    case PAGE_START:
        return PAGE_COORDINATOR;
    case PAGE_COORDINATOR:
        return PAGE_COORDINATED;
    case PAGE_COORDINATED:
        return PAGE_END;
    default:
        return -1;
    }
}

void RfmBindWizard::createPages()
{
    setPage(PAGE_START, new TLStartPage(this));
    setPage(PAGE_COORDINATOR, new CoordinatorPage(this));
    setPage(PAGE_COORDINATED, new CoordinatedPage(this));
    setPage(PAGE_END, new TLEndPage(this));

    setStartId(PAGE_START);

    connect(button(QWizard::CustomButton1), SIGNAL(clicked()), this, SLOT(customBackClicked()));
    setButtonText(QWizard::CustomButton1, buttonText(QWizard::BackButton));
    QList<QWizard::WizardButton> button_layout;
    button_layout << QWizard::Stretch << QWizard::CustomButton1 << QWizard::NextButton << QWizard::CancelButton << QWizard::FinishButton;
    setButtonLayout(button_layout);
    connect(this, SIGNAL(currentIdChanged(int)), this, SLOT(pageChanged(int)));
}

void RfmBindWizard::customBackClicked()
{
    back();
}

void RfmBindWizard::pageChanged(int currId)
{
    button(QWizard::CustomButton1)->setVisible(currId != PAGE_START);
    button(QWizard::CancelButton)->setVisible(currId != PAGE_END);
}

