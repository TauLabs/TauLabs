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

#ifndef CONVERTMWRATE_H
#define CONVERTMWRATE_H

#include <QDialog>

namespace Ui {
class ConvertMWRate;
}

class ConvertMWRate : public QDialog
{
    Q_OBJECT
    
public:
    explicit ConvertMWRate(QWidget *parent = 0);
    ~ConvertMWRate();

    double getRollKp() const;
    double getRollKi() const;
    double getRollKd() const;
    double getPitchKp() const;
    double getPitchKi() const;
    double getPitchKd() const;
    double getYawKp() const;
    double getYawKi() const;
    double getYawKd() const;

private:
    Ui::ConvertMWRate *ui;
};

#endif // CONVERTMWRATE_H

/**
 * @}
 * @}
 */
