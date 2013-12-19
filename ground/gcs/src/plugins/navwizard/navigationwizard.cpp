/**
 ******************************************************************************
 * @file       navigationwizard.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup NavWizard Navigation Wizard
 * @{
 * @brief A Wizard to make the initial setup easy for everyone.
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

#include "navigationwizard.h"
#include "pages/startpage.h"
#include "pages/endpage.h"
#include "pages/failsafepage.h"
#include "pages/rebootpage.h"
#include "pages/savepage.h"
#include "extensionsystem/pluginmanager.h"
#include "vehicleconfigurationhelper.h"
#include "uploader/uploadergadgetfactory.h"

NavigationWizard::NavigationWizard(QWidget *parent) : QWizard(parent),
    m_restartNeeded(false), m_connectionManager(0)
{
    setWindowTitle(tr("Tau Labs Navigation Setup Wizard"));
    setOption(QWizard::IndependentPages, false);
    setWizardStyle(QWizard::ModernStyle);
    setMinimumSize(600, 500);
    resize(600, 500);
    createPages();
}

int NavigationWizard::nextId() const
{
    switch (currentId()) {
    case PAGE_START:
        return PAGE_FAILSAFE;
    case PAGE_FAILSAFE:
        return PAGE_SAVE;
    case PAGE_SAVE:
        return PAGE_REBOOT;
    case PAGE_REBOOT:
        return PAGE_END;
    default:
        return -1;
    }
}

void NavigationWizard::createPages()
{
    setPage(PAGE_START, new StartPage(this));
    setPage(PAGE_FAILSAFE, new FailsafePage(this));
    setPage(PAGE_REBOOT, new RebootPage(this));
    setPage(PAGE_SAVE, new SavePage(this));
    setPage(PAGE_END, new EndPage(this));

    setStartId(PAGE_START);

    connect(button(QWizard::CustomButton1), SIGNAL(clicked()), this, SLOT(back()));
    setButtonText(QWizard::CustomButton1, buttonText(QWizard::BackButton));
    QList<QWizard::WizardButton> button_layout;
    button_layout << QWizard::Stretch << QWizard::CustomButton1 << QWizard::NextButton << QWizard::CancelButton << QWizard::FinishButton;
    setButtonLayout(button_layout);
    connect(this, SIGNAL(currentIdChanged(int)), this, SLOT(pageChanged(int)));
}

void NavigationWizard::pageChanged(int currId)
{
    button(QWizard::CustomButton1)->setVisible(currId != PAGE_START);
    button(QWizard::CancelButton)->setVisible(currId != PAGE_END);
}

QString NavigationWizard::getSummaryText()
{
    return QString("");
}

bool NavigationWizard::isCalibrationPerformed() const
{
    return false;
}

/**
 * @}
 * @}
 */
