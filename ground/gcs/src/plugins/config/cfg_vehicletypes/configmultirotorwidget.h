/**
 ******************************************************************************
 *
 * @file       configairframetwidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Airframe configuration panel
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
#ifndef CONFIGMULTIROTORWIDGET_H
#define CONFIGMULTIROTORWIDGET_H

#include "ui_airframe.h"
#include "../uavobjectwidgetutils/configtaskwidget.h"
#include "cfg_vehicletypes/vehicleconfig.h"

#include "multirotorairframesettings.h"

#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include "uavtalk/telemetrymanager.h"
#include <QWidget>
#include <QList>
#include <QItemDelegate>

class Ui_Widget;

class ConfigMultiRotorWidget: public VehicleConfig
{
    Q_OBJECT

public:
    ConfigMultiRotorWidget(Ui_AircraftWidget *aircraft = 0, QWidget *parent = 0);
    ~ConfigMultiRotorWidget();

    friend class ConfigVehicleTypeWidget;

private:
    Ui_AircraftWidget *m_aircraft;

    QWidget *uiowner;
    QGraphicsSvgItem *quad;

    MultirotorAirframeSettings *multirotorAirframeSettings;
    MultirotorAirframeSettings::MultirotorTypeOptions multirotorSelector;

    bool setupTri();
    bool setupQuad();
    bool setupHexa();
    bool setupOcto();
    bool setupMultiRotorMixer(double mixerFactors[8][3]);
    void assignOutputNames(QStringList motorList);
    void setupMultirotorMotor(int channel, double roll, double pitch, double yaw);
    bool throwConfigError();

    float motorDirectionCoefficient;

    uint8_t numMotors;
    static const QString CHANNELBOXNAME;
    static const uint8_t MAX_SUPPORTED_MULTIROTOR;

private slots:
    void setupUI();
    void updateConfigObjectsFromWidgets();
    void drawAirframe();
    void updateOutputLabels();

protected:

signals:
    void configurationChanged();

};


#endif // CONFIGMULTIROTORWIDGET_H
