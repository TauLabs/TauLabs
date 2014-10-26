/**
 ******************************************************************************
 * @file       convertmwrate.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief The Configuration Gadget used to update settings in the firmware
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
#include "convertmwrate.h"
#include "ui_convertmwrate.h"

ConvertMWRate::ConvertMWRate(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ConvertMWRate)
{
    ui->setupUi(this);
}

ConvertMWRate::~ConvertMWRate()
{
    delete ui;
}

double ConvertMWRate::getRollKp() const
{
    return ui->rollKp->value() * 10.0 * 4 / 80.0 / 500.0;
}

double ConvertMWRate::getRollKi() const
{
    return ui->rollKi->value() * 1000.0 * 4 / 125.0 / 64.0 / (ui->loopTime->value() * 1e-6) / 500.0;
}

double ConvertMWRate::getRollKd() const
{
    return ui->rollKd->value() * ui->loopTime->value() * 1e-6 * 4 * 3 / 32.0 / 500.0;
}

double ConvertMWRate::getPitchKp() const
{
    return ui->pitchKp->value() * 10.0 * 4 / 80.0 / 500.0;
}

double ConvertMWRate::getPitchKi() const
{
    return ui->pitchKi->value() * 1000.0 * 4 / 125.0 / 64.0 / (ui->loopTime->value() * 1e-6) / 500.0;
}

double ConvertMWRate::getPitchKd() const
{
    return ui->pitchKd->value() * ui->loopTime->value() * 1e-6 * 4 * 3 / 32.0 / 500.0;
}

double ConvertMWRate::getYawKp() const
{
    return ui->yawKp->value() * 10.0 * 4 / 80.0 / 500.0;
}

double ConvertMWRate::getYawKi() const
{
    return ui->yawKi->value() * 1000.0 * 4 / 125.0 / 64.0 / (ui->loopTime->value() * 1e-6) / 500.0;
}

double ConvertMWRate::getYawKd() const
{
    return ui->yawKd->value() * ui->loopTime->value() * 1e-6 * 4 * 3 / 32.0 / 500.0;
}

/**
 * @}
 * @}
 */
