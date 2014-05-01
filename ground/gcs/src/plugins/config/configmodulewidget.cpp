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
#include "hottsettings.h"
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

    HoTTSettings hoTTSettings;
    QString hoTTSettingsName = hoTTSettings.getName();

    // Link the checkboxes
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbAirspeed, ModuleSettings::ADMINSTATE_AIRSPEED);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbAltitudeHold, ModuleSettings::ADMINSTATE_ALTITUDEHOLD);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbBattery, ModuleSettings::ADMINSTATE_BATTERY);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbComBridge, ModuleSettings::ADMINSTATE_COMUSBBRIDGE);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbGPS, ModuleSettings::ADMINSTATE_GPS);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbUavoMavlink, ModuleSettings::ADMINSTATE_UAVOMAVLINKBRIDGE);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbOveroSync, ModuleSettings::ADMINSTATE_OVEROSYNC);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbVibrationAnalysis, ModuleSettings::ADMINSTATE_VIBRATIONANALYSIS);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbVtolFollower, ModuleSettings::ADMINSTATE_VTOLPATHFOLLOWER);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbPathPlanner, ModuleSettings::ADMINSTATE_PATHPLANNER);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbUAVOHottBridge, ModuleSettings::ADMINSTATE_UAVOHOTTBRIDGE);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbUAVOLighttelemetryBridge, ModuleSettings::ADMINSTATE_UAVOLIGHTTELEMETRYBRIDGE);
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", ui->cbUAVOFrskyBridge, ModuleSettings::ADMINSTATE_UAVOFRSKYSENSORHUBBRIDGE);

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

    //HoTT Sensor
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Sensor", ui->cb_GAM, HoTTSettings::SENSOR_GAM);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Sensor", ui->cb_EAM, HoTTSettings::SENSOR_EAM);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Sensor", ui->cb_Vario, HoTTSettings::SENSOR_VARIO);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Sensor", ui->cb_GPS, HoTTSettings::SENSOR_GPS);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Sensor", ui->cb_ESC, HoTTSettings::SENSOR_ESC);

    ui->cb_GAM->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_GAM->setProperty(falseString.toLatin1(), "Disabled");

    ui->cb_EAM->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_EAM->setProperty(falseString.toLatin1(), "Disabled");

    ui->cb_Vario->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_Vario->setProperty(falseString.toLatin1(), "Disabled");

    ui->cb_GPS->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_GPS->setProperty(falseString.toLatin1(), "Disabled");

    ui->cb_ESC->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_ESC->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings POWERVOLTAGE
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MINPOWERVOLTAGE, HoTTSettings::LIMIT_MINPOWERVOLTAGE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXPOWERVOLTAGE, HoTTSettings::LIMIT_MAXPOWERVOLTAGE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MINPOWERVOLTAGE, HoTTSettings::WARNING_MINPOWERVOLTAGE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXPOWERVOLTAGE, HoTTSettings::WARNING_MAXPOWERVOLTAGE);
    ui->cb_MINPOWERVOLTAGE->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MINPOWERVOLTAGE->setProperty(falseString.toLatin1(), "Disabled");
    ui->cb_MAXPOWERVOLTAGE->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXPOWERVOLTAGE->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings CURRENT
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXCURRENT, HoTTSettings::LIMIT_MAXCURRENT);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXCURRENT, HoTTSettings::WARNING_MAXCURRENT);
    ui->cb_MAXCURRENT->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXCURRENT->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings USEDCAPACITY
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXUSEDCAPACITY, HoTTSettings::LIMIT_MAXUSEDCAPACITY);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXUSEDCAPACITY, HoTTSettings::WARNING_MAXUSEDCAPACITY);
    ui->cb_MAXUSEDCAPACITY->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXUSEDCAPACITY->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings CELLVOLTAGE
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MINCELLVOLTAGE, HoTTSettings::LIMIT_MINCELLVOLTAGE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MINCELLVOLTAGE, HoTTSettings::WARNING_MINCELLVOLTAGE);
    ui->cb_MINCELLVOLTAGE->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MINCELLVOLTAGE->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings SENSOR1VOLTAGE
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MINSENSOR1VOLTAGE, HoTTSettings::LIMIT_MINSENSOR1VOLTAGE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXSENSOR1VOLTAGE, HoTTSettings::LIMIT_MAXSENSOR1VOLTAGE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MINSENSOR1VOLTAGE, HoTTSettings::WARNING_MINSENSOR1VOLTAGE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXSENSOR1VOLTAGE, HoTTSettings::WARNING_MAXSENSOR1VOLTAGE);
    ui->cb_MINSENSOR1VOLTAGE->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MINSENSOR1VOLTAGE->setProperty(falseString.toLatin1(), "Disabled");
    ui->cb_MAXSENSOR1VOLTAGE->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXSENSOR1VOLTAGE->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings SENSOR2VOLTAGE
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MINSENSOR2VOLTAGE, HoTTSettings::LIMIT_MINSENSOR2VOLTAGE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXSENSOR2VOLTAGE, HoTTSettings::LIMIT_MAXSENSOR2VOLTAGE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MINSENSOR2VOLTAGE, HoTTSettings::WARNING_MINSENSOR2VOLTAGE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXSENSOR2VOLTAGE, HoTTSettings::WARNING_MAXSENSOR2VOLTAGE);
    ui->cb_MINSENSOR2VOLTAGE->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MINSENSOR2VOLTAGE->setProperty(falseString.toLatin1(), "Disabled");
    ui->cb_MAXSENSOR2VOLTAGE->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXSENSOR2VOLTAGE->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings SENSOR1TEMP
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MINSENSOR1TEMP, HoTTSettings::LIMIT_MINSENSOR1TEMP);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXSENSOR1TEMP, HoTTSettings::LIMIT_MAXSENSOR1TEMP);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MINSENSOR1TEMP, HoTTSettings::WARNING_MINSENSOR1TEMP);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXSENSOR1TEMP, HoTTSettings::WARNING_MAXSENSOR1TEMP);
    ui->cb_MINSENSOR1TEMP->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MINSENSOR1TEMP->setProperty(falseString.toLatin1(), "Disabled");
    ui->cb_MAXSENSOR1TEMP->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXSENSOR1TEMP->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings SENSOR2TEMP
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MINSENSOR2TEMP, HoTTSettings::LIMIT_MINSENSOR2TEMP);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXSENSOR2TEMP, HoTTSettings::LIMIT_MAXSENSOR2TEMP);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MINSENSOR2TEMP, HoTTSettings::WARNING_MINSENSOR2TEMP);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXSENSOR2TEMP, HoTTSettings::WARNING_MAXSENSOR2TEMP);
    ui->cb_MINSENSOR2TEMP->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MINSENSOR2TEMP->setProperty(falseString.toLatin1(), "Disabled");
    ui->cb_MAXSENSOR2TEMP->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXSENSOR2TEMP->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings FUEL
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MINFUEL, HoTTSettings::LIMIT_MINFUEL);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MINFUEL, HoTTSettings::WARNING_MINFUEL);
    ui->cb_MINFUEL->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MINFUEL->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings SENSOR1TEMP
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MINRPM, HoTTSettings::LIMIT_MINRPM);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXRPM, HoTTSettings::LIMIT_MAXRPM);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MINRPM, HoTTSettings::WARNING_MINRPM);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXRPM, HoTTSettings::WARNING_MAXRPM);
    ui->cb_MINRPM->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MINRPM->setProperty(falseString.toLatin1(), "Disabled");
    ui->cb_MAXRPM->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXRPM->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings SERVOTEMP
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXSERVOTEMP, HoTTSettings::LIMIT_MAXSERVOTEMP);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXSERVOTEMP, HoTTSettings::WARNING_MAXSERVOTEMP);
    ui->cb_MAXSERVOTEMP->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXSERVOTEMP->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings SERVODIFFERENCE
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXSERVODIFFERENCE, HoTTSettings::LIMIT_MAXSERVODIFFERENCE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXSERVODIFFERENCE, HoTTSettings::WARNING_MAXSERVODIFFERENCE);
    ui->cb_MAXSERVODIFFERENCE->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXSERVODIFFERENCE->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings SPEED
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MINSPEED, HoTTSettings::LIMIT_MINSPEED);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXSPEED, HoTTSettings::LIMIT_MAXSPEED);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MINSPEED, HoTTSettings::WARNING_MINSPEED);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXSPEED, HoTTSettings::WARNING_MAXSPEED);
    ui->cb_MINSPEED->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MINSPEED->setProperty(falseString.toLatin1(), "Disabled");
    ui->cb_MAXSPEED->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXSPEED->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings SPEED
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MINHEIGHT, HoTTSettings::LIMIT_MINHEIGHT);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXHEIGHT, HoTTSettings::LIMIT_MAXHEIGHT);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MINHEIGHT, HoTTSettings::WARNING_MINHEIGHT);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXHEIGHT, HoTTSettings::WARNING_MAXHEIGHT);
    ui->cb_MINHEIGHT->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MINHEIGHT->setProperty(falseString.toLatin1(), "Disabled");
    ui->cb_MAXHEIGHT->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXHEIGHT->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings SERVOTEMP
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_MAXDISTANCE, HoTTSettings::LIMIT_MAXDISTANCE);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_MAXDISTANCE, HoTTSettings::WARNING_MAXDISTANCE);
    ui->cb_MAXDISTANCE->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_MAXDISTANCE->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings NEGDIFFERENCE1
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_NEGDIFFERENCE1, HoTTSettings::LIMIT_NEGDIFFERENCE1);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_NEGDIFFERENCE1, HoTTSettings::WARNING_NEGDIFFERENCE1);
    ui->cb_NEGDIFFERENCE1->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_NEGDIFFERENCE1->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings NEGDIFFERENCE2
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_NEGDIFFERENCE2, HoTTSettings::LIMIT_NEGDIFFERENCE2);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_NEGDIFFERENCE2, HoTTSettings::WARNING_NEGDIFFERENCE2);
    ui->cb_NEGDIFFERENCE2->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_NEGDIFFERENCE2->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings POSDIFFERENCE1
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_POSDIFFERENCE1, HoTTSettings::LIMIT_POSDIFFERENCE1);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_POSDIFFERENCE1, HoTTSettings::WARNING_POSDIFFERENCE1);
    ui->cb_POSDIFFERENCE1->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_POSDIFFERENCE1->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings POSDIFFERENCE2
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Limit", ui->sb_POSDIFFERENCE2, HoTTSettings::LIMIT_POSDIFFERENCE2);
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_POSDIFFERENCE2, HoTTSettings::WARNING_POSDIFFERENCE2);
    ui->cb_POSDIFFERENCE2->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_POSDIFFERENCE2->setProperty(falseString.toLatin1(), "Disabled");

    //HoTT Settings ALTITUDEBEEP
    addUAVObjectToWidgetRelation(hoTTSettingsName, "Warning", ui->cb_ALTITUDEBEEP, HoTTSettings::WARNING_ALTITUDEBEEP);
    ui->cb_ALTITUDEBEEP->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_ALTITUDEBEEP->setProperty(falseString.toLatin1(), "Disabled");

    //Help button
    addHelpButton(ui->inputHelp,"http://wiki.taulabs.org/OnlineHelp:-Modules");

    // Connect any remaining widgets
    connect(airspeedSettings, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(updateAirspeedUAVO(UAVObject *)));
    connect(ui->pb_startVibrationTest, SIGNAL(clicked()), this, SLOT(toggleVibrationTest()));

    // Set text properties for checkboxes. The second argument is the UAVO field that corresponds
    // to the checkbox's true (respectively, false) state.
    ui->cbAirspeed->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbAirspeed->setProperty(falseString.toLatin1(), "Disabled");

    ui->cbAltitudeHold->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbAltitudeHold->setProperty(falseString.toLatin1(), "Disabled");

    ui->cbBattery->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbBattery->setProperty(falseString.toLatin1(), "Disabled");

    ui->cbComBridge->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbComBridge->setProperty(falseString.toLatin1(), "Disabled");

    ui->cbGPS->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbGPS->setProperty(falseString.toLatin1(), "Disabled");

    ui->cbUavoMavlink->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbUavoMavlink->setProperty(falseString.toLatin1(), "Disabled");

    ui->cbOveroSync->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbOveroSync->setProperty(falseString.toLatin1(), "Disabled");

    ui->cbVibrationAnalysis->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbVibrationAnalysis->setProperty(falseString.toLatin1(), "Disabled");

    ui->cbVtolFollower->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbVtolFollower->setProperty(falseString.toLatin1(), "Disabled");

    ui->cbPathPlanner->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbPathPlanner->setProperty(falseString.toLatin1(), "Disabled");

    ui->cbUAVOHottBridge->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbUAVOHottBridge->setProperty(falseString.toLatin1(), "Disabled");

    ui->cbUAVOFrskyBridge->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbUAVOFrskyBridge->setProperty(falseString.toLatin1(), "Disabled");
    
    ui->cbUAVOLighttelemetryBridge->setProperty(trueString.toLatin1(), "Enabled");
    ui->cbUAVOLighttelemetryBridge->setProperty(falseString.toLatin1(), "Disabled");	

    ui->gb_measureVoltage->setProperty(trueString.toLatin1(), "Enabled");
    ui->gb_measureVoltage->setProperty(falseString.toLatin1(), "Disabled");

    ui->gb_measureCurrent->setProperty(trueString.toLatin1(), "Enabled");
    ui->gb_measureCurrent->setProperty(falseString.toLatin1(), "Disabled");

    enableBatteryTab(false);
    enableAirspeedTab(false);
    enableVibrationTab(false);
    enableHoTTTelemetryTab(false);

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

    obj = getObjectManager()->getObject(HoTTSettings::NAME);
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
    else if (objName.compare(HoTTSettings::NAME) == 0)
        enableHoTTTelemetryTab(success);
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

//! Enable or disable the HoTT telemetrie tab
void ConfigModuleWidget::enableHoTTTelemetryTab(bool enabled)
{
    int idx = ui->moduleTab->indexOf(ui->tabHoTTTelemetry);
    ui->moduleTab->setTabEnabled(idx,enabled);
}

/**
 * @}
 * @}
 */
