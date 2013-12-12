/**
 ******************************************************************************
 *
 * @file       biascalibrationpage.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

#include <QMessageBox>
#include <QDebug>
#include "biascalibrationpage.h"
#include "ui_biascalibrationpage.h"
#include "setupwizard.h"

BiasCalibrationPage::BiasCalibrationPage(SetupWizard *wizard, QWidget *parent) :
    AbstractWizardPage(wizard, parent),
    ui(new Ui::BiasCalibrationPage), m_calibrationUtil(0)
{
    ui->setupUi(this);
    connect(ui->levelButton, SIGNAL(clicked()), this, SLOT(performCalibration()));
}

BiasCalibrationPage::~BiasCalibrationPage()
{
    if (m_calibrationUtil) {
        delete m_calibrationUtil;
    }
    delete ui;
}

bool BiasCalibrationPage::validatePage()
{
    return true;
}

bool BiasCalibrationPage::isComplete() const
{
    return ui->levelButton->isEnabled();
}

void BiasCalibrationPage::enableButtons(bool enable)
{
    ui->levelButton->setEnabled(enable);
    getWizard()->button(QWizard::NextButton)->setEnabled(enable);
    getWizard()->button(QWizard::CancelButton)->setEnabled(enable);
    getWizard()->button(QWizard::BackButton)->setEnabled(enable);
    getWizard()->button(QWizard::CustomButton1)->setEnabled(enable);
    QApplication::processEvents();
}

void BiasCalibrationPage::performCalibration()
{
    if (!getWizard()->getConnectionManager()->isConnected()) {
        QMessageBox msgBox;
        msgBox.setText(tr("A flight controller must be connected to your computer to perform bias "
                          "calculations.\nPlease connect your flight controller to your computer and try again."));
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.setDefaultButton(QMessageBox::Ok);
        msgBox.exec();
        return;
    }

    // Get the calibration helper object
    if (!m_calibrationUtil) {
        ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
        m_calibrationUtil = pm->getObject<Calibration>();
        Q_ASSERT(m_calibrationUtil);
        if (!m_calibrationUtil)
            return;
    }

    enableButtons(false);
    ui->progressLabel->setText(QString(tr("Retrieving data...")));

    connect(m_calibrationUtil, SIGNAL(levelingProgressChanged(int)), this, SLOT(calibrationProgress(int)));
    connect(m_calibrationUtil, SIGNAL(calibrationCompleted()), this, SLOT(calibrationDone()));
    QTimer::singleShot(20000, this, SLOT(calibrationDone()));

    m_calibrationUtil->doStartBiasAndLeveling();
}

void BiasCalibrationPage::calibrationProgress(int current)
{
    const int total = 100;
    if (ui->levellinProgressBar->maximum() != (int)total) {
        ui->levellinProgressBar->setMaximum((int)total);
    }
    if (ui->levellinProgressBar->value() != (int)current) {
        ui->levellinProgressBar->setValue((int)current);
    }
}

void BiasCalibrationPage::calibrationDone()
{
    disconnect(this, SLOT(calibrationTimeout()));
    stopCalibration();
    emit completeChanged();
}

void BiasCalibrationPage::calibrationTimeout()
{
    stopCalibration();

    QMessageBox msgBox;
    msgBox.setText(tr("Calibration timed out"));
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
}

void BiasCalibrationPage::stopCalibration()
{
    if (m_calibrationUtil) {
        disconnect(m_calibrationUtil, SIGNAL(levelingProgressChanged(int)), this, SLOT(calibrationProgress(int)));
        disconnect(m_calibrationUtil, SIGNAL(calibrationCompleted()), this, SLOT(calibrationDone()));
        ui->progressLabel->setText(QString(tr("<font color='green'>Done!</font>")));
        enableButtons(true);
    }
}
