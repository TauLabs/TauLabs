/**
 ******************************************************************************
 *
 * @file       coordinatedpage.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup RfmBindWizard Setup Wizard
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

#include "coordinatedpage.h"
#include "ui_coordinatedpage.h"
#include "rfmbindwizard.h"

#include <extensionsystem/pluginmanager.h>
#include <uavobjectutil/uavobjectutilmanager.h>

#include <QTimer>
#include <coreplugin/iboardtype.h>

CoordinatedPage::CoordinatedPage(RfmBindWizard *wizard, QWidget *parent) :
    RadioProbePage(wizard, parent), ui(new Ui::CoordinatedPage),
    m_coordinatorConfigured(false)
{
    ui->setupUi(this);
    ui->setCoordinator->setEnabled(false);

    connect(ui->setCoordinator, SIGNAL(clicked()), this, SLOT(bindCoordinator()));
    connect(this, SIGNAL(probeChanged(bool)), this, SLOT(updateProbe(bool)));
}

CoordinatedPage::~CoordinatedPage()
{
    delete ui;
}

void CoordinatedPage::initializePage()
{
    emit completeChanged();
}

bool CoordinatedPage::isComplete() const
{
    return m_coordinatorConfigured;
}

bool CoordinatedPage::validatePage()
{
    disconnect(this);
    return true;
}

//! Update the UI when the probe status changes
void CoordinatedPage::updateProbe(bool probed)
{
    Core::IBoardType *board = getBoardType();
    if (probed && board) {
        ui->boardTypeLabel->setText(board->shortName());

        // Get the ID for this board and make it a coordinator
        quint32 rfmId = board->getRfmID();

        if (rfmId == getWizard()->getCoordID()) {
            // Don't allow binding the same board
            ui->setCoordinator->setEnabled(false);
            return;
        }

        // Do not allow performing this for multiple boards
        if (!m_coordinatorConfigured)
            ui->setCoordinator->setEnabled(true);
    } else {
        ui->boardTypeLabel->setText("Unknown");
        ui->setCoordinator->setEnabled(false);
    }
}

bool CoordinatedPage::bindCoordinator()
{
    Core::IBoardType *board = getBoardType();
    if (board == NULL)
        return false;

    // Get the ID for this board and make it a coordinator
    quint32 rfmId = board->getRfmID();

    // This is the same board we selected as the coordinator
    if (rfmId == getWizard()->getCoordID())
        return false;

    board->bindRadio(getWizard()->getCoordID(), getWizard()->getMaxBps(), getWizard()->getMaxRfPower(),
                     getWizard()->getLinkMode(),getWizard()->getMinChannel(), getWizard()->getMaxChannel());

    m_coordinatorConfigured = true;
    ui->setCoordinator->setEnabled(false);

    emit completeChanged();
    stopProbing();

    return true;
}

