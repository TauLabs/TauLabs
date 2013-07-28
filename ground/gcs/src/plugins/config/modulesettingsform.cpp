/**
 ******************************************************************************
 *
 * @file       modulesettingsform.cpp
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

#include "modulesettingsform.h"
#include "ui_modulesettingsform.h"

#include "airspeedsettings.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "modulesettings.h"
#include "vibrationanalysissettings.h"

// Define static variables
QString ModuleSettingsForm::trueString("TrueString");
QString ModuleSettingsForm::falseString("FalseString");

/**
 * @brief ModuleSettingsForm::ModuleSettingsForm Constructor
 * @param parent
 */
ModuleSettingsForm::ModuleSettingsForm(QWidget *parent, QPushButton *saveButton, QPushButton *applyButton, QPushButton *reloadButton) :
    ConfigTaskWidget(parent),
    moduleSettingsWidget(new Ui::ModuleSettingsWidget)
{
    moduleSettingsWidget->setupUi(this);

    // Connect the apply/save/reload buttons.
    if (applyButton != NULL || saveButton != NULL)
        addApplySaveButtons(applyButton, saveButton);

    if (reloadButton != NULL)
        addReloadButton(reloadButton, 0);

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
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", moduleSettingsWidget->gb_Airspeed, ModuleSettings::ADMINSTATE_AIRSPEED);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", moduleSettingsWidget->gb_Battery, ModuleSettings::ADMINSTATE_BATTERY);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", moduleSettingsWidget->gb_ComBridge, ModuleSettings::ADMINSTATE_COMUSBBRIDGE);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", moduleSettingsWidget->gb_GPS, ModuleSettings::ADMINSTATE_GPS);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", moduleSettingsWidget->gb_OveroSync, ModuleSettings::ADMINSTATE_OVEROSYNC);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", moduleSettingsWidget->gb_VibrationAnalysis, ModuleSettings::ADMINSTATE_VIBRATIONANALYSIS);

    addUAVObjectToWidgetRelation(batterySettingsName, "SensorType", moduleSettingsWidget->gb_measureVoltage, FlightBatterySettings::SENSORTYPE_BATTERYVOLTAGE);
    addUAVObjectToWidgetRelation(batterySettingsName, "SensorType", moduleSettingsWidget->gb_measureCurrent, FlightBatterySettings::SENSORTYPE_BATTERYCURRENT);

    // Link the fields
    addUAVObjectToWidgetRelation(airspeedSettingsName, "GPSSamplePeriod_ms", moduleSettingsWidget->sb_gpsUpdateRate);
    addUAVObjectToWidgetRelation(airspeedSettingsName, "Scale", moduleSettingsWidget->sb_pitotScale);
    addUAVObjectToWidgetRelation(airspeedSettingsName, "ZeroPoint", moduleSettingsWidget->sb_pitotZeroPoint);

    addUAVObjectToWidgetRelation(batterySettingsName, "Type", moduleSettingsWidget->cb_batteryType);
    addUAVObjectToWidgetRelation(batterySettingsName, "NbCells", moduleSettingsWidget->sb_numBatteryCells);
    addUAVObjectToWidgetRelation(batterySettingsName, "Capacity", moduleSettingsWidget->sb_batteryCapacity);
    addUAVObjectToWidgetRelation(batterySettingsName, "VoltageThresholds", moduleSettingsWidget->sb_lowVoltageAlarm, FlightBatterySettings::VOLTAGETHRESHOLDS_ALARM);
    addUAVObjectToWidgetRelation(batterySettingsName, "VoltageThresholds", moduleSettingsWidget->sb_lowVoltageWarning, FlightBatterySettings::VOLTAGETHRESHOLDS_WARNING);
    addUAVObjectToWidgetRelation(batterySettingsName, "SensorCalibrations", moduleSettingsWidget->sb_voltageCalibration, FlightBatterySettings::SENSORCALIBRATIONS_VOLTAGEFACTOR);
    addUAVObjectToWidgetRelation(batterySettingsName, "SensorCalibrations", moduleSettingsWidget->sb_currentCalibration, FlightBatterySettings::SENSORCALIBRATIONS_CURRENTFACTOR);

    addUAVObjectToWidgetRelation(batteryStateName, "Voltage", moduleSettingsWidget->le_liveVoltageReading);
    addUAVObjectToWidgetRelation(batteryStateName, "Current", moduleSettingsWidget->le_liveCurrentReading);

    addUAVObjectToWidgetRelation(moduleSettingsName, "ComUsbBridgeSpeed", moduleSettingsWidget->cb_combridgeSpeed);

    addUAVObjectToWidgetRelation(moduleSettingsName, "GPSSpeed", moduleSettingsWidget->cb_gpsSpeed);

    addUAVObjectToWidgetRelation(vibrationAnalysisSettingsName, "SampleRate", moduleSettingsWidget->sb_sampleRate);
    addUAVObjectToWidgetRelation(vibrationAnalysisSettingsName, "FFTWindowSize", moduleSettingsWidget->cb_windowSize);

    // Connect any remaining widgets
    connect(airspeedSettings, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(updateAirspeedUAVO(UAVObject *)));
    connect(moduleSettingsWidget->cb_pitotType, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePitotType(int)));
    connect(moduleSettingsWidget->pb_startVibrationTest, SIGNAL(clicked()), this, SLOT(toggleVibrationTest()));

    // Set text properties for checkboxes. The second argument is the UAVO field that corresponds
    // to the checkbox's true (respectively, false) state.
    moduleSettingsWidget->gb_Airspeed->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_Airspeed->setProperty(falseString.toAscii(), "Disabled");

    moduleSettingsWidget->gb_Battery->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_Battery->setProperty(falseString.toAscii(), "Disabled");

    moduleSettingsWidget->gb_ComBridge->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_ComBridge->setProperty(falseString.toAscii(), "Disabled");

    moduleSettingsWidget->gb_GPS->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_GPS->setProperty(falseString.toAscii(), "Disabled");

    moduleSettingsWidget->gb_OveroSync->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_OveroSync->setProperty(falseString.toAscii(), "Disabled");

    moduleSettingsWidget->gb_VibrationAnalysis->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_VibrationAnalysis->setProperty(falseString.toAscii(), "Disabled");

    moduleSettingsWidget->gb_measureVoltage->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_measureVoltage->setProperty(falseString.toAscii(), "Disabled");

    moduleSettingsWidget->gb_measureCurrent->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_measureCurrent->setProperty(falseString.toAscii(), "Disabled");

    // Refresh widget contents
    refreshWidgetsValues();

    // Prevent mouse wheel from changing values
    disableMouseWheelEvents();
}


ModuleSettingsForm::~ModuleSettingsForm()
{
    delete moduleSettingsWidget;
}


/**
 * @brief ModuleSettingsForm::getWidgetFromVariant Reimplements getWidgetFromVariant. This version supports "FalseString".
 * @param widget pointer to the widget from where to get the value
 * @param scale scale to be used on the assignement
 * @return returns the value of the widget times the scale
 */
QVariant ModuleSettingsForm::getVariantFromWidget(QWidget * widget, double scale)
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
bool ModuleSettingsForm::setWidgetFromVariant(QWidget *widget, QVariant value, double scale)
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


void ModuleSettingsForm::toggleVibrationTest()
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

void ModuleSettingsForm::updatePitotType(int comboboxValue)
{

//    AirspeedSettings *airspeedSettings;
//    airspeedSettings = AirspeedSettings::GetInstance(getObjectManager());
//    AirspeedSettings::DataFields airspeedSettingsData;
//    airspeedSettingsData = airspeedSettings->getData();

//    switch (comboboxValue) {
//    case AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV5004:
//        moduleSettingsWidget->cb_pitotType->setCurrentIndex(AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV5004);
//        moduleSettingsWidget->gb_airspeedPitot->setChecked(true);
//    case AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV7002:
//        moduleSettingsWidget->cb_pitotType->setCurrentIndex(AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV7002);
//        moduleSettingsWidget->gb_airspeedPitot->setChecked(true);
//    case AirspeedSettings::AIRSPEEDSENSORTYPE_EAGLETREEAIRSPEEDV3:
//        moduleSettingsWidget->cb_pitotType->setCurrentIndex(AirspeedSettings::AIRSPEEDSENSORTYPE_EAGLETREEAIRSPEEDV3);
//        moduleSettingsWidget->gb_airspeedPitot->setChecked(true);
//        break;
//    }

}


void ModuleSettingsForm::updateAirspeedUAVO(UAVObject *obj)
{
    Q_UNUSED(obj);
}

/**
 * @brief ModuleSettingsForm::updateAirspeedGroupbox Updates groupbox when airspeed UAVO changes
 * @param obj
 */
void ModuleSettingsForm::updateAirspeedGroupbox(UAVObject *obj)
{
    Q_UNUSED(obj);

    AirspeedSettings *airspeedSettings;
    airspeedSettings = AirspeedSettings::GetInstance(getObjectManager());
    AirspeedSettings::DataFields airspeedSettingsData;
    airspeedSettingsData = airspeedSettings->getData();

    moduleSettingsWidget->gb_airspeedGPS->setChecked(false);
    moduleSettingsWidget->gb_airspeedPitot->setChecked(false);

    switch (airspeedSettingsData.AirspeedSensorType) {
    case AirspeedSettings::AIRSPEEDSENSORTYPE_GPSONLY:
        moduleSettingsWidget->gb_airspeedGPS->setChecked(true);
        break;
    case AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV5004:
        moduleSettingsWidget->cb_pitotType->setCurrentIndex(AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV5004);
        moduleSettingsWidget->gb_airspeedPitot->setChecked(true);
    case AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV7002:
        moduleSettingsWidget->cb_pitotType->setCurrentIndex(AirspeedSettings::AIRSPEEDSENSORTYPE_DIYDRONESMPXV7002);
        moduleSettingsWidget->gb_airspeedPitot->setChecked(true);
    case AirspeedSettings::AIRSPEEDSENSORTYPE_EAGLETREEAIRSPEEDV3:
        moduleSettingsWidget->cb_pitotType->setCurrentIndex(AirspeedSettings::AIRSPEEDSENSORTYPE_EAGLETREEAIRSPEEDV3);
        moduleSettingsWidget->gb_airspeedPitot->setChecked(true);
        break;
    }
}
