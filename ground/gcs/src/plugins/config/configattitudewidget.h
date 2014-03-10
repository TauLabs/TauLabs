/**
 ******************************************************************************
 *
 * @file       configattitudetwidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Telemetry configuration panel
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
#ifndef CONFIGATTITUDEWIDGET_H
#define CONFIGATTITUDEWIDGET_H

#include "ui_attitude.h"
#include "calibration.h"

#include "configtaskwidget.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include "calibration.h"
#include <QWidget>
#include <QSvgRenderer>
#include <QGraphicsSvgItem>
#include <QList>
#include <QTimer>
#include <QMutex>

class Ui_Widget;

class ConfigAttitudeWidget: public ConfigTaskWidget
{
    Q_OBJECT

public:
    ConfigAttitudeWidget(QWidget *parent = 0);
    ~ConfigAttitudeWidget();
    
protected:
    void showEvent(QShowEvent *event);
    void resizeEvent(QResizeEvent *event);

    Calibration calibration;

private:
    Ui_AttitudeWidget *m_ui;
    QGraphicsSvgItem *paperplane;

    QMutex sensorsUpdateLock;
    int phaseCounter;
    const static double maxVarValue;
    const static int calibrationDelay = 10;

    float initialMagCorrectionRate;

    QMap<QString, UAVObject::Metadata> originalMetaData;

    bool board_has_accelerometer;
    bool board_has_magnetometer;

private slots:
    //! Overriden method from the configTaskWidget to update UI
    virtual void refreshWidgetsValues(UAVObject * obj=NULL);

    //! Display the plane in various positions
    void displayPlane(int i);

    // Slots for measuring the sensor noise
    void do_SetDirty();
    void configureSixPoint();

};

#endif // CONFIGATTITUDEWIDGET_H
