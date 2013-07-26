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
#ifndef FAILSAFEPAGE_H
#define FAILSAFEPAGE_H

#include "abstractwizardpage.h"

namespace Ui {
class FailsafePage;
}

class FailsafePage : public AbstractWizardPage {
    Q_OBJECT

public:
    explicit FailsafePage(QWizard *wizard, QWidget *parent = 0);
    ~FailsafePage();
    bool isComplete() const;

private slots:
    void flightStatusUpdated(UAVObject *);

private:
    Ui::FailsafePage *ui;

    // Check failsafe engaged twice
    enum {
        FAILSAFE_START,
        FAILSAFE_ENGAGED1,
        FAILSAFE_DISEGNAGED,
        FAILSAFE_ENGAGED2
    } failsafe_test_state;
};

#endif // FAILSAFEPAGE_H
