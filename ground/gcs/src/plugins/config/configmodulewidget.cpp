/**
 ******************************************************************************
 * @file       configmodulewidget.cpp
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

#include "configmodulewidget.h"

#include <extensionsystem/pluginmanager.h>
#include <coreplugin/generalsettings.h>


#include "airspeedsettings.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "modulesettings.h"
#include "vibrationanalysissettings.h"

// Define static variables
QString ConfigModuleWidget::trueString("TrueString");
QString ConfigModuleWidget::falseString("FalseString");


ConfigModuleWidget::ConfigModuleWidget(QWidget *parent) : ConfigTaskWidget(parent)
{
    ui = new Ui::Modules();
    ui->setupUi(this);

    connect(this, SIGNAL(autoPilotConnected()), this, SLOT(recheckTabs()));

    // Populate UAVO strings
    AirspeedSettings *airspeedSettings;
    airspeedSettings = AirspeedSettings::GetInstance(getObjectManager());
    QString airspeedSettingsName = airspeedSettings->getName();

    FlightBatterySettings batterySettings;
    QString batterySettingsName = batterySettings.getName();

    FlightBatteryState batteryState;
    QString batteryStateName = batteryState.getName();

    ModuleSettings moduleSettings;
    QString moduleSettingsName = moduleSettings.getName();

    VibrationAnalysisSettings vibrationAnalysisSettings;
    QString vibrationAnalysisSettingsName = vibrationAnalysisSettings.getName();

    // Link the checkboxes
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbAirspeed, ModuleSettings::ADMINSTATE_AIRSPEED);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbBattery, ModuleSettings::ADMINSTATE_BATTERY);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbComBridge, ModuleSettings::ADMINSTATE_COMUSBBRIDGE);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbGPS, ModuleSettings::ADMINSTATE_GPS);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbUavoMavlink, ModuleSettings::ADMINSTATE_UAVOMAVLINKBRIDGE);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbOveroSync, ModuleSettings::ADMINSTATE_OVEROSYNC);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbVibrationAnalysis, ModuleSettings::ADMINSTATE_VIBRATIONANALYSIS);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbVtolFollower, ModuleSettings::ADMINSTATE_VTOLPATHFOLLOWER);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbPathPlanner, ModuleSettings::ADMINSTATE_PATHPLANNER);

    addUAVObjectToWidgetRelation(batterySettingsName, "SensorType", ui->gb_measureVoltage, FlightBatterySettings::SENSORTYPE_BATTERYVOLTAGE);
    addUAVObjectToWidgetRelation(batterySettingsName, "SensorType", ui->gb_measureCurrent, FlightBatterySettings::SENSORTYPE_BATTERYCURRENT);

    // Link the fields
    addUAVObjectToWidgetRelation(airspeedSettingsName, "GPSSamplePeriod_ms", ui->sb_gpsUpdateRate);
    addUAVObjectToWidgetRelation(airspeedSettingsName, "Scale", ui->sb_pitotScale);
    addUAVObjectToWidgetRelation(airspeedSettingsName, "ZeroPoint", ui->sb_pitotZeroPoint);
    addUAVObjectToWidgetRelation(airspeedSettingsName, "AnalogPin", ui->cbAirspeedAnalog);

    addUAVObjectToWidgetRelation(batterySettingsName, "Type", ui->cb_batteryType);
    addUAVObjectToWidgetRelation(batterySettingsName, "NbCells", ui->sb_numBatteryCells);
    addUAVObjectToWidgetRelation(batterySettingsName, "Capacity", ui->sb_batteryCapacity);
    addUAVObjectToWidgetRelation(batterySettingsName, "VoltagePin", ui->cbVoltagePin);
    addUAVObjectToWidgetRelation(batterySettingsName, "CurrentPin", ui->cbCurrentPin);
    addUAVObjectToWidgetRelation(batterySettingsName, "VoltageThresholds", ui->sb_lowVoltageAlarm, FlightBatterySettings::VOLTAGETHRESHOLDS_ALARM);
    addUAVObjectToWidgetRelation(batterySettingsName, "VoltageThresholds", ui->sb_lowVoltageWarning, FlightBatterySettings::VOLTAGETHRESHOLDS_WARNING);
    addUAVObjectToWidgetRelation(batterySettingsName, "SensorCalibrationFactor", ui->sb_voltageFactor, FlightBatterySettings::SENSORCALIBRATIONFACTOR_VOLTAGE);
    addUAVObjectToWidgetRelation(batterySettingsName, "SensorCalibrationFactor", ui->sb_currentFactor, FlightBatterySettings::SENSORCALIBRATIONFACTOR_CURRENT);
    addUAVObjectToWidgetRelation(batterySettingsName, "SensorCalibrationOffset", ui->sb_voltageOffSet, FlightBatterySettings::SENSORCALIBRATIONOFFSET_VOLTAGE);
    addUAVObjectToWidgetRelation(batterySettingsName, "SensorCalibrationOffset", ui->sb_currentOffSet, FlightBatterySettings::SENSORCALIBRATIONOFFSET_CURRENT);

    addUAVObjectToWidgetRelation(batteryStateName, "Voltage", ui->le_liveVoltageReading);
    addUAVObjectToWidgetRelation(batteryStateName, "Current", ui->le_liveCurrentReading);

    addUAVObjectToWidgetRelation(vibrationAnalysisSettingsName, "SampleRate", ui->sb_sampleRate);
    addUAVObjectToWidgetRelation(vibrationAnalysisSettingsName, "FFTWindowSize", ui->cb_windowSize);

    addHelpButton(ui->inputHelp,"https://github.com/TauLabs/TauLabs/wiki/OnlineHelp:-Modules");

    // Connect any remaining widgets
    connect(airspeedSettings, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(updateAirspeedUAVO(UAVObject *)));
    connect(ui->pb_startVibrationTest, SIGNAL(clicked()), this, SLOT(toggleVibrationTest()));

    // Set text properties for checkboxes. The second argument is the UAVO field that corresponds
    // to the checkbox's true (respectively, false) state.
    ui->cbAirspeed->setProperty(trueString.toAscii(), "Enabled");
    ui->cbAirspeed->setProperty(falseString.toAscii(), "Disabled");

    ui->cbBattery->setProperty(trueString.toAscii(), "Enabled");
    ui->cbBattery->setProperty(falseString.toAscii(), "Disabled");

    ui->cbComBridge->setProperty(trueString.toAscii(), "Enabled");
    ui->cbComBridge->setProperty(falseString.toAscii(), "Disabled");

    ui->cbGPS->setProperty(trueString.toAscii(), "Enabled");
    ui->cbGPS->setProperty(falseString.toAscii(), "Disabled");

    ui->cbUavoMavlink->setProperty(trueString.toAscii(), "Enabled");
    ui->cbUavoMavlink->setProperty(falseString.toAscii(), "Disabled");

    ui->cbOveroSync->setProperty(trueString.toAscii(), "Enabled");
    ui->cbOveroSync->setProperty(falseString.toAscii(), "Disabled");

    ui->cbVibrationAnalysis->setProperty(trueString.toAscii(), "Enabled");
    ui->cbVibrationAnalysis->setProperty(falseString.toAscii(), "Disabled");

    ui->cbVtolFollower->setProperty(trueString.toAscii(), "Enabled");
    ui->cbVtolFollower->setProperty(falseString.toAscii(), "Disabled");

    ui->cbPathPlanner->setProperty(trueString.toAscii(), "Enabled");
    ui->cbPathPlanner->setProperty(falseString.toAscii(), "Disabled");

    ui->gb_measureVoltage->setProperty(trueString.toAscii(), "Enabled");
    ui->gb_measureVoltage->setProperty(falseString.toAscii(), "Disabled");

    ui->gb_measureCurrent->setProperty(trueString.toAscii(), "Enabled");
    ui->gb_measureCurrent->setProperty(falseString.toAscii(), "Disabled");

    enableBatteryTab(false);
    enableAirspeedTab(false);
    enableVibrationTab(false);

    // Load UAVObjects to widget relations from UI file
    // using objrelation dynamic property
    autoLoadWidgets();

    // Refresh widget contents
    refreshWidgetsValues();

    // Prevent mouse wheel from changing values
    disableMouseWheelEvents();
}

ConfigModuleWidget::~ConfigModuleWidget()
{
    delete ui;
}

void ConfigModuleWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
}

void ConfigModuleWidget::enableControls(bool enable)
{
    Q_UNUSED(enable);
}

//! Query optional objects to determine which tabs can be configured
void ConfigModuleWidget::recheckTabs()
{
    UAVObject * obj;

    obj = getObjectManager()->getObject(AirspeedSettings::NAME);
    connect(obj, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(objectUpdated(UAVObject*,bool)), Qt::UniqueConnection);
    obj->requestUpdate();

    obj = getObjectManager()->getObject(FlightBatterySettings::NAME);
    connect(obj, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(objectUpdated(UAVObject*,bool)), Qt::UniqueConnection);
    obj->requestUpdate();

    obj = getObjectManager()->getObject(VibrationAnalysisSettings::NAME);
    connect(obj, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(objectUpdated(UAVObject*,bool)), Qt::UniqueConnection);
    obj->requestUpdate();
}

//! Enable appropriate tab when objects are updated
void ConfigModuleWidget::objectUpdated(UAVObject * obj, bool success)
{
    if (!obj)
        return;

    QString objName = obj->getName();
    if (objName.compare(AirspeedSettings::NAME) == 0)
        enableAirspeedTab(success);
    else if (objName.compare(FlightBatterySettings::NAME) == 0)
        enableBatteryTab(success);
    else if (objName.compare(VibrationAnalysisSettings::NAME) == 0)
        enableVibrationTab(success);
}

/**
 * @brief ModuleSettingsForm::getWidgetFromVariant Reimplements getWidgetFromVariant. This version supports "FalseString".
 * @param widget pointer to the widget from where to get the value
 * @param scale scale to be used on the assignement
 * @return returns the value of the widget times the scale
 */
QVariant ConfigModuleWidget::getVariantFromWidget(QWidget * widget, double scale)
{
    if(QGroupBox * groupBox=qobject_cast<QGroupBox *>(widget)) {
        QString ret;
        if (groupBox->property("TrueString").isValid() && groupBox->property("FalseString").isValid()) {
            if(groupBox->isChecked())
                ret = groupBox->property("TrueString").toString();
            else
                ret = groupBox->property("FalseString").toString();
        } else {
            if(groupBox->isChecked())
                ret = "TRUE";
            else
                ret = "FALSE";
        }

        return ret;
    } else if(QCheckBox * checkBox=qobject_cast<QCheckBox *>(widget)) {
        QString ret;
        if (checkBox->property("TrueString").isValid() && checkBox->property("FalseString").isValid()) {
            if (checkBox->isChecked())
                ret = checkBox->property("TrueString").toString();
            else
                ret = checkBox->property("FalseString").toString();
        }
        else {
            if(checkBox->isChecked())
                ret = "TRUE";
            else
                ret = "FALSE";
        }

        return ret;
    } else {
        return ConfigTaskWidget::getVariantFromWidget(widget, scale);
    }
}


/**
 * @brief ModuleSettingsForm::setWidgetFromVariant Reimplements setWidgetFromVariant. This version supports "FalseString".
 * @param widget pointer for the widget to set
 * @param scale scale to be used on the assignement
 * @param value value to be used on the assignement
 * @return returns true if the assignement was successfull
 */
bool ConfigModuleWidget::setWidgetFromVariant(QWidget *widget, QVariant value, double scale)
{
    if(QGroupBox * groupBox=qobject_cast<QGroupBox *>(widget)) {
        bool bvalue;
        if (groupBox->property("TrueString").isValid() && groupBox->property("FalseString").isValid()) {
            bvalue = value.toString()==groupBox->property("TrueString").toString();
        }
        else{
            bvalue = value.toString()=="TRUE";
        }
        groupBox->setChecked(bvalue);
        return true;
    } else if(QCheckBox * checkBox=qobject_cast<QCheckBox *>(widget)) {
        bool bvalue;
        if (checkBox->property("TrueString").isValid() && checkBox->property("FalseString").isValid()) {
            bvalue = value.toString()==checkBox->property("TrueString").toString();
        }
        else {
            bvalue = value.toString()=="TRUE";
        }
        checkBox->setChecked(bvalue);
        return true;
    } else {
        return ConfigTaskWidget::setWidgetFromVariant(widget, value, scale);
    }
}


void ConfigModuleWidget::toggleVibrationTest()
{
    VibrationAnalysisSettings *vibrationAnalysisSettings;
    vibrationAnalysisSettings = VibrationAnalysisSettings::GetInstance(getObjectManager());
    VibrationAnalysisSettings::DataFields vibrationAnalysisSettingsData;
    vibrationAnalysisSettingsData = vibrationAnalysisSettings->getData();

    // Toggle state
    if (vibrationAnalysisSettingsData.TestingStatus == VibrationAnalysisSettings::TESTINGSTATUS_ON)
        vibrationAnalysisSettingsData.TestingStatus = VibrationAnalysisSettings::TESTINGSTATUS_OFF;
    else
        vibrationAnalysisSettingsData.TestingStatus = VibrationAnalysisSettings::TESTINGSTATUS_ON;

    // Send data
    vibrationAnalysisSettings->setData(vibrationAnalysisSettingsData);
    vibrationAnalysisSettings->updated();
}

void ConfigModuleWidget::updateAirspeedUAVO(UAVObject *obj)
{
    Q_UNUSED(obj);
}

/**
 * @brief ConfigModuleWidget::updateAirspeedGroupbox Updates groupbox when airspeed UAVO changes
 * @param obj
 */
void ConfigModuleWidget::updateAirspeedGroupbox(UAVObject *obj)
{
    Q_UNUSED(obj);

    AirspeedSettings *airspeedSettings;
    airspeedSettings = AirspeedSettings::GetInstance(getObjectManager());
    AirspeedSettings::DataFields airspeedSettingsData;
    airspeedSettingsData = airspeedSettings->getData();

    ui->gb_airspeedGPS->setChecked(false);
    ui->gb_airspeedPitot->setChecked(false);

    switch (airspeedSettingsData.AirspeedSensorType) {
    case AirspeedSettings::AIRSPEEDSENSORTYPE_GPSONLY:
        ui->gb_airspeedGPS->setChecked(true);
        break;
    case AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV5004:
        ui->cb_pitotType->setCurrentIndex(AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV5004);
        ui->gb_airspeedPitot->setChecked(true);
    case AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV7002:
        ui->cb_pitotType->setCurrentIndex(AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV7002);
        ui->gb_airspeedPitot->setChecked(true);
    case AirspeedSettings::AIRSPEEDSENSORTYPE_EAGLETREEAIRSPEEDV3:
        ui->cb_pitotType->setCurrentIndex(AirspeedSettings::AIRSPEEDSENSORTYPE_EAGLETREEAIRSPEEDV3);
        ui->gb_airspeedPitot->setChecked(true);
        break;
    }
}

//! Enable or disable the battery tab
void ConfigModuleWidget::enableBatteryTab(bool enabled)
{
    int idx = ui->moduleTab->indexOf(ui->tabBattery);
    ui->moduleTab->setTabEnabled(idx,enabled);
}

//! Enable or disable the airspeed tab
void ConfigModuleWidget::enableAirspeedTab(bool enabled)
{
    int idx = ui->moduleTab->indexOf(ui->tabAirspeed);
    ui->moduleTab->setTabEnabled(idx,enabled);
}

//! Enable or disable the vibration tab
void ConfigModuleWidget::enableVibrationTab(bool enabled)
{
    int idx = ui->moduleTab->indexOf(ui->tabVibration);
    ui->moduleTab->setTabEnabled(idx,enabled);
}

/**
 * @}
 * @}
 */
