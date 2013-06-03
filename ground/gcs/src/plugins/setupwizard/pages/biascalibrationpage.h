/**
 ******************************************************************************
 *
 * @file       biascalibrationpage.h
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

#ifndef BIASCALIBRATIONPAGE_H
#define BIASCALIBRATIONPAGE_H

#include "abstractwizardpage.h"
#include "calibration.h"

namespace Ui {
class BiasCalibrationPage;
}

class BiasCalibrationPage : public AbstractWizardPage {
    Q_OBJECT

public:
    explicit BiasCalibrationPage(SetupWizard *wizard, QWidget *parent = 0);
    ~BiasCalibrationPage();
    bool validatePage();
    bool isComplete() const;

private slots:
    void performCalibration();
    void calibrationProgress(int current);
    void calibrationDone();
    void calibrationTimeout();

private:
    static const int BIAS_CYCLES = 200;
    static const int BIAS_RATE   = 50;

    Ui::BiasCalibrationPage *ui;
    Calibration *m_calibrationUtil;

    void stopCalibration();
    void enableButtons(bool enable);
};

#endif // BIASCALIBRATIONPAGE_H
