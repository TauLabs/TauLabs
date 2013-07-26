/**
 ******************************************************************************
 *
 * @file       modulesettingsform.h
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
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

#ifndef MODULESETTINGSFORM_H
#define MODULESETTINGSFORM_H

#include <QWidget>
#include "uavobjectwidgetutils/configtaskwidget.h"
#include "ui_modules.h"

#include "taskinfo.h"

namespace Ui {
    class ModuleSettingsWidget;
}

class ModuleSettingsForm : public ConfigTaskWidget
{
    Q_OBJECT

    enum messageFlag {AIRSPEED = 0x01,
                      BATTERY = 0x02,
                      COMBRIDGE = 0x04,
                      OVEROSYNC = 0x08,
                      VIBRATION = 0x10};

public:
    explicit ModuleSettingsForm(Ui::Modules *ui, QWidget *parent = 0);
    ~ModuleSettingsForm();
    friend class ConfigInputWidget;
private slots:
    void updateAirspeedUAVO(UAVObject *);
    void updateAirspeedGroupbox(UAVObject *);
    void updatePitotType(int comboboxValue);
    void toggleVibrationTest();

    void toggleAirspeedModule(bool toggleState);
    void toggleBatteryModule(bool toggleState);
    void toggleComBridgeModule(bool toggleState);
    void toggleOveroSyncModule(bool toggleState);
    void toggleVibrationAnalysisModule(bool toggleState);

private:
    QVariant getVariantFromWidget(QWidget * widget, double scale);
    bool setWidgetFromVariant(QWidget *widget, QVariant value, double scale);
    void toggleErrorMessage();

    static QString trueString;
    static QString falseString;

    TaskInfo *taskInfo;

    Ui::ModuleSettingsWidget *moduleSettingsWidget;
    Ui::Modules *modulesTab;

    uint32_t rebootMessage_flag;
};

#endif // MODULESETTINGSFORM_H
