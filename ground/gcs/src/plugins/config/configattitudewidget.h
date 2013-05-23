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
#include <QtGui/QWidget>
#include <QtSvg/QSvgRenderer>
#include <QtSvg/QGraphicsSvgItem>
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
    void drawVariancesGraph();

    Ui_AttitudeWidget *m_ui;
    QGraphicsSvgItem *paperplane;
    QGraphicsSvgItem *sensorsBargraph;
    QGraphicsSvgItem *accel_x;
    QGraphicsSvgItem *accel_y;
    QGraphicsSvgItem *accel_z;
    QGraphicsSvgItem *gyro_x;
    QGraphicsSvgItem *gyro_y;
    QGraphicsSvgItem *gyro_z;
    QGraphicsSvgItem *mag_x;
    QGraphicsSvgItem *mag_y;
    QGraphicsSvgItem *mag_z;
    QGraphicsSvgItem *baro;
    QMutex sensorsUpdateLock;
    double maxBarHeight;
    int phaseCounter;
    const static double maxVarValue;
    const static int calibrationDelay = 10;

    QList<double> gyro_accum_x;
    QList<double> gyro_accum_y;
    QList<double> gyro_accum_z;
    QList<double> accel_accum_x;
    QList<double> accel_accum_y;
    QList<double> accel_accum_z;
    QList<double> mag_accum_x;
    QList<double> mag_accum_y;
    QList<double> mag_accum_z;
    QList<double> baro_accum;

    double accel_data_x[6], accel_data_y[6], accel_data_z[6];
    double mag_data_x[6], mag_data_y[6], mag_data_z[6];

    UAVObject::Metadata initialAccelsMdata;
    UAVObject::Metadata initialGyrosMdata;
    UAVObject::Metadata initialMagMdata;
    UAVObject::Metadata initialBaroMdata;
    float initialMagCorrectionRate;

    static const int NOISE_SAMPLES = 100;

    QMap<QString, UAVObject::Metadata> originalMetaData;

    bool board_has_accelerometer;
    bool board_has_magnetometer;

private slots:
    //! Overriden method from the configTaskWidget to update UI
    virtual void refreshWidgetsValues(UAVObject * obj=NULL);

    //! Display the plane in various positions
    void displayPlane(int i);

    // Slots for measuring the sensor noise
    void doStartNoiseMeasurement();
    void doGetNoiseSample(UAVObject *);
    void do_SetDirty();
    void configureSixPoint();

};

#endif // CONFIGATTITUDEWIDGET_H
