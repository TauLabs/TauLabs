/**
 ******************************************************************************
 * @file       hwfieldselector.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup CorePlugin Core Plugin
 * @{
 * @brief A selector widget for hardware configuration field options
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
#ifndef HWFIELDSELECTOR_H
#define HWFIELDSELECTOR_H

#include <QWidget>
#include <hwfieldselector.h>
#include <ui_hwfieldselector.h>
#include <uavobjectfield.h>

namespace Ui {
    class HwFieldSelector;
}

class HwFieldSelector : public QWidget
{
    Q_OBJECT
public:
    explicit HwFieldSelector(QWidget *parent = 0);
    void setUavoField(UAVObjectField *field);

    QComboBox *getCombo();
signals:
    
public slots:

private:
    Ui::HwFieldSelector *ui;
};

#endif // HWFIELDSELECTOR_H
