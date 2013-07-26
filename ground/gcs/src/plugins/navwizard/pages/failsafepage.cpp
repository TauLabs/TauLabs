/**
 ******************************************************************************
 *
 * @file       failsafepage.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup NavWizard Navigation Wizard
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
    ui(new Ui::FailsafePage), failsafe_test_state(FAILSAFE_START)
{
    ui->setupUi(this);
    setFont(QFont("Ubuntu", 2));

    // Monitor for updates from flight status
    FlightStatus * flightStatus = FlightStatus::GetInstance(getObjectManager());
    Q_ASSERT(flightStatus);
    connect(flightStatus, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(flightStatusUpdated(UAVObject*)));

    // Refresh UI and get initial state
    flightStatusUpdated(flightStatus);
}

FailsafePage::~FailsafePage()
{
    delete ui;
}

//! Check failsafe has been cycled
bool FailsafePage::isComplete() const
{
    return failsafe_test_state == FAILSAFE_ENGAGED2;
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

    if (failsafe)
        ui->label->setText(tr("Failsafe enabled"));
    else
        ui->label->setText(tr("Failsafe disabled"));

    // Track that the failsafe is engaged a few times
    switch(failsafe_test_state) {
    case FAILSAFE_START:
        if (failsafe)
            failsafe_test_state = FAILSAFE_ENGAGED1;
        break;
    case FAILSAFE_ENGAGED1:
        if (!failsafe)
            failsafe_test_state = FAILSAFE_DISEGNAGED;
        break;
    case FAILSAFE_DISEGNAGED:
        if (failsafe) {
            failsafe_test_state = FAILSAFE_ENGAGED2;
            emit completeChanged();
        }
        break;
    case FAILSAFE_ENGAGED2:
        // nothing to do. terminal state.
        break;
    }

}
