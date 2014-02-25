/**
 ******************************************************************************
 * @file       configmodulewidget.h
 * @brief      Configure the optional modules
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
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
#ifndef CONFIGMODULEWIDGET_H
#define CONFIGMODULEWIDGET_H

#include "ui_modules.h"

#include "uavobjectwidgetutils/configtaskwidget.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"

namespace Ui {
    class Modules;
}

class ConfigModuleWidget: public ConfigTaskWidget
{
    Q_OBJECT

public:
        ConfigModuleWidget(QWidget *parent = 0);
        ~ConfigModuleWidget();

private slots:
    void updateAirspeedUAVO(UAVObject *);
    void updateAirspeedGroupbox(UAVObject *);
    void toggleVibrationTest();

    void recheckTabs();
    void objectUpdated(UAVObject * obj, bool success);

private:
    QVariant getVariantFromWidget(QWidget * widget, double scale);
    bool setWidgetFromVariant(QWidget *widget, QVariant value, double scale);

    /* To activate the appropriate tabs */
    void enableBatteryTab(bool enabled);
    void enableAirspeedTab(bool enabled);
    void enableVibrationTab(bool enabled);
    void enableHoTTTelemetryTab(bool enabled);

    static QString trueString;
    static QString falseString;

    Ui::Modules *ui;

protected:
    void resizeEvent(QResizeEvent *event);
    virtual void enableControls(bool enable);
};

#endif // CONFIGMODULEWIDGET_H
