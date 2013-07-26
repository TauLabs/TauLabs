/**
 ******************************************************************************
 *
 * @file       failsafepage.cpp
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

#include "flightstatus.h"

#include "failsafepage.h"
#include "ui_failsafepage.h"

FailsafePage::FailsafePage(QWizard *wizard, QWidget *parent) :
    AbstractWizardPage(wizard, parent),
    ui(new Ui::FailsafePage), failsafe_disengaged(false), failsafe_reengaged(false)
{
    ui->setupUi(this);
    setFont(QFont("Ubuntu", 2));

    // Disable next button until failsafe verified
    getQWizard()->button(QWizard::NextButton)->setEnabled(false);

    // Monitor for updates from flight status
    FlightStatus * flightStatus = FlightStatus::GetInstance(getObjectManager());
    Q_ASSERT(flightStatus);
    connect(flightStatus, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(flightStatusUpdated(UAVObject*)));
}

FailsafePage::~FailsafePage()
{
    delete ui;
}

//! Toggle the failsafe enabled signal when necessary
void FailsafePage::flightStatusUpdated(UAVObject *obj)
{
    FlightStatus *flightStatusObj = dynamic_cast<FlightStatus *>(obj);
    Q_ASSERT(flightStatusObj);
    if (!flightStatusObj)
        return;

    FlightStatus::DataFields flightStatus = flightStatusObj->getData();
    bool failsafe = flightStatus.ControlSource == FlightStatus::CONTROLSOURCE_FAILSAFE;
    ui->cbFailsafe->setChecked(failsafe);

    // First require failsafe to be disengaged (transmitter found)
    failsafe_disengaged |= !failsafe;

    // Then require it be reengaged by turning off the transmitter
    if (failsafe_disengaged && failsafe)
        failsafe_reengaged = true;

    // If both steps have been performed then allow going forward
    if (failsafe_disengaged && failsafe_reengaged)
        getQWizard()->button(QWizard::NextButton)->setEnabled(true);
}
