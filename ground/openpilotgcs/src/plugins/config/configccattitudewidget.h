/**
 ******************************************************************************
 *
 * @file       configccattitudewidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Configure the properties of the attitude module in CopterControl
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
#ifndef CCATTITUDEWIDGET_H
#define CCATTITUDEWIDGET_H

#include "ui_ccattitude.h"
#include "../uavobjectwidgetutils/configtaskwidget.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include "calibration.h"
#include <QtGui/QWidget>
#include <QTimer>

class Ui_Widget;

class ConfigCCAttitudeWidget : public ConfigTaskWidget
{
    Q_OBJECT

public:
    explicit ConfigCCAttitudeWidget(QWidget *parent = 0);
    ~ConfigCCAttitudeWidget();

    virtual void updateObjectsFromWidgets();

private slots:
    void openHelp();

    //! Display the plane in various positions
    void displayPlane(int i);

private:
    Ui_ccattitude *ui;

    QGraphicsSvgItem *paperplane;

    Calibration calibration;
protected slots:
    virtual void enableControls(bool enable);

protected:
    void showEvent(QShowEvent *event);
    void resizeEvent(QResizeEvent *event);

    void computeScaleBias();

};

#endif // CCATTITUDEWIDGET_H
