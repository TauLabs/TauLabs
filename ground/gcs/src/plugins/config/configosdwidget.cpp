/**
 ******************************************************************************
 * @file       configosdwidget.cpp
 * @brief      Configure the OSD
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
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

#include "configosdwidget.h"

#include <extensionsystem/pluginmanager.h>
#include <coreplugin/generalsettings.h>

#include "onscreendisplaysettings.h"
#include "onscreendisplaypagesettings.h"

#include "manualcontrolcommand.h"
#include "manualcontrolsettings.h"

#include "ui_osdpage.h"


// Define static variables
QString ConfigOsdWidget::trueString("TrueString");
QString ConfigOsdWidget::falseString("FalseString");


ConfigOsdWidget::ConfigOsdWidget(QWidget *parent) : ConfigTaskWidget(parent)
{
    ui = new Ui::Osd();
    ui->setupUi(this);

    osdSettingsObj = OnScreenDisplaySettings::GetInstance(getObjectManager());
    manualCommandObj = ManualControlCommand::GetInstance(getObjectManager());
    manualSettingsObj = ManualControlSettings::GetInstance(getObjectManager());

    // setup the OSD widgets
    QString osdSettingsName = osdSettingsObj->getName();
    addUAVObjectToWidgetRelation(osdSettingsName, "OSDEnabled", ui->cb_osd_enabled);
    ui->cb_osd_enabled->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_osd_enabled->setProperty(falseString.toLatin1(), "Disabled");

    addUAVObjectToWidgetRelation(osdSettingsName, "PALWhite", ui->sb_white_pal);
    addUAVObjectToWidgetRelation(osdSettingsName, "PALBlack", ui->sb_black_pal);
    addUAVObjectToWidgetRelation(osdSettingsName, "NTSCWhite", ui->sb_white_ntsc);
    addUAVObjectToWidgetRelation(osdSettingsName, "NTSCBlack", ui->sb_black_ntsc);
    addUAVObjectToWidgetRelation(osdSettingsName, "XOffset", ui->sb_x_offset);
    addUAVObjectToWidgetRelation(osdSettingsName, "YOffset", ui->sb_y_offset);
    addUAVObjectToWidgetRelation(osdSettingsName, "Units", ui->cb_units);

    addUAVObjectToWidgetRelation(osdSettingsName, "NumPages", ui->sb_num_pages);
    addUAVObjectToWidgetRelation(osdSettingsName, "PageSwitch", ui->cb_page_switch);

    addUAVObjectToWidgetRelation(osdSettingsName, "PageConfig", ui->osdPagePos1, 0);
    addUAVObjectToWidgetRelation(osdSettingsName, "PageConfig", ui->osdPagePos2, 1);
    addUAVObjectToWidgetRelation(osdSettingsName, "PageConfig", ui->osdPagePos3, 2);
    addUAVObjectToWidgetRelation(osdSettingsName, "PageConfig", ui->osdPagePos4, 3);
    addUAVObjectToWidgetRelation(osdSettingsName, "PageConfig", ui->osdPagePos5, 4);
    addUAVObjectToWidgetRelation(osdSettingsName, "PageConfig", ui->osdPagePos6, 5);

    addUAVObjectToWidgetRelation(osdSettingsName, "DisableMenuWhenArmed", ui->cb_menu_disabled);
    ui->cb_menu_disabled->setProperty(trueString.toLatin1(), "Enabled");
    ui->cb_menu_disabled->setProperty(falseString.toLatin1(), "Disabled");

    connect(ManualControlCommand::GetInstance(getObjectManager()),SIGNAL(objectUpdated(UAVObject*)),this,SLOT(movePageSlider()));
    connect(OnScreenDisplaySettings::GetInstance(getObjectManager()),SIGNAL(objectUpdated(UAVObject*)),this,SLOT(updatePositionSlider()));

    // Load UAVObjects to widget relations from UI file
    // using objrelation dynamic property
    autoLoadWidgets();

    // Refresh widget contents
    refreshWidgetsValues();

    // Prevent mouse wheel from changing values
    disableMouseWheelEvents();

    // Setup OSD pages
    pages[0] = ui->osdPage1;
    pages[1] = ui->osdPage2;
    pages[2] = ui->osdPage3;
    pages[3] = ui->osdPage4;

    ui_pages[0] = new Ui::OsdPage();
    osdPageSettingsObj = OnScreenDisplayPageSettings::GetInstance(getObjectManager());
    setupOsdPage(ui_pages[0], ui->osdPage1, osdPageSettingsObj);

    ui_pages[0]->copyButton1->setText("Copy Page 2");
    ui_pages[0]->copyButton2->setText("Copy Page 3");
    ui_pages[0]->copyButton3->setText("Copy Page 4");

    connect(ui_pages[0]->copyButton1, SIGNAL(released()), this, SLOT(handle_button_0_1()));
    connect(ui_pages[0]->copyButton2, SIGNAL(released()), this, SLOT(handle_button_0_2()));
    connect(ui_pages[0]->copyButton3, SIGNAL(released()), this, SLOT(handle_button_0_3()));

    ui_pages[1] = new Ui::OsdPage();
    osdPageSettings2Obj = OnScreenDisplayPageSettings2::GetInstance(getObjectManager());
    setupOsdPage(ui_pages[1], ui->osdPage2, osdPageSettings2Obj);

    ui_pages[1]->copyButton1->setText("Copy Page 1");
    ui_pages[1]->copyButton2->setText("Copy Page 3");
    ui_pages[1]->copyButton3->setText("Copy Page 4");

    connect(ui_pages[1]->copyButton1, SIGNAL(released()), this, SLOT(handle_button_1_0()));
    connect(ui_pages[1]->copyButton2, SIGNAL(released()), this, SLOT(handle_button_1_2()));
    connect(ui_pages[1]->copyButton3, SIGNAL(released()), this, SLOT(handle_button_1_3()));

    ui_pages[2] = new Ui::OsdPage();
    osdPageSettings3Obj = OnScreenDisplayPageSettings3::GetInstance(getObjectManager());
    setupOsdPage(ui_pages[2], ui->osdPage3, osdPageSettings3Obj);

    ui_pages[2]->copyButton1->setText("Copy Page 1");
    ui_pages[2]->copyButton2->setText("Copy Page 2");
    ui_pages[2]->copyButton3->setText("Copy Page 4");

    connect(ui_pages[2]->copyButton1, SIGNAL(released()), this, SLOT(handle_button_2_0()));
    connect(ui_pages[2]->copyButton2, SIGNAL(released()), this, SLOT(handle_button_2_1()));
    connect(ui_pages[2]->copyButton3, SIGNAL(released()), this, SLOT(handle_button_2_3()));

    ui_pages[3] = new Ui::OsdPage();
    osdPageSettings4Obj = OnScreenDisplayPageSettings4::GetInstance(getObjectManager());
    setupOsdPage(ui_pages[3], ui->osdPage4, osdPageSettings4Obj);

    ui_pages[3]->copyButton1->setText("Copy Page 1");
    ui_pages[3]->copyButton2->setText("Copy Page 2");
    ui_pages[3]->copyButton3->setText("Copy Page 3");

    connect(ui_pages[3]->copyButton1, SIGNAL(released()), this, SLOT(handle_button_3_0()));
    connect(ui_pages[3]->copyButton2, SIGNAL(released()), this, SLOT(handle_button_3_1()));
    connect(ui_pages[3]->copyButton3, SIGNAL(released()), this, SLOT(handle_button_3_2()));
}

ConfigOsdWidget::~ConfigOsdWidget()
{
    delete ui;
    //delete [] ui_pages;
}

/**
 * @brief Converts a raw Switch Channel value into a Switch position index
 * @param channelNumber channel number to convert
 * @param switchPositions total switch positions
 * @return Switch position index converted from the raw value
 */
quint8 ConfigOsdWidget::scaleSwitchChannel(quint8 channelNumber, quint8 switchPositions)
{
    if(channelNumber > (ManualControlSettings::CHANNELMIN_NUMELEM - 1))
            return 0;
    ManualControlSettings::DataFields manualSettingsDataPriv = manualSettingsObj->getData();
    ManualControlCommand::DataFields manualCommandDataPriv = manualCommandObj->getData();

    float valueScaled;
    int chMin = manualSettingsDataPriv.ChannelMin[channelNumber];
    int chMax = manualSettingsDataPriv.ChannelMax[channelNumber];
    int chNeutral = manualSettingsDataPriv.ChannelNeutral[channelNumber];

    int value = manualCommandDataPriv.Channel[channelNumber];
    if ((chMax > chMin && value >= chNeutral) || (chMin > chMax && value <= chNeutral))
    {
        if (chMax != chNeutral)
            valueScaled = (float)(value - chNeutral) / (float)(chMax - chNeutral);
        else
            valueScaled = 0;
    }
    else
    {
        if (chMin != chNeutral)
            valueScaled = (float)(value - chNeutral) / (float)(chNeutral - chMin);
        else
            valueScaled = 0;
    }

    if (valueScaled < -1.0)
        valueScaled = -1.0;
    else
    if (valueScaled >  1.0)
        valueScaled =  1.0;

    // Convert channel value into the switch position in the range [0..N-1]
    // This uses the same optimized computation as flight code to be consistent
    quint8 pos = ((qint16)(valueScaled * 256) + 256) * switchPositions >> 9;
    if (pos >= switchPositions)
        pos = switchPositions - 1;

    return pos;
}

void ConfigOsdWidget::movePageSlider()
{
    OnScreenDisplaySettings::DataFields onScreenDisplaySettingsDataPriv = osdSettingsObj->getData();

    switch(onScreenDisplaySettingsDataPriv.PageSwitch) {
        case OnScreenDisplaySettings::PAGESWITCH_ACCESSORY0:
            ui->osdPageSlider->setValue(scaleSwitchChannel(ManualControlSettings::CHANNELMIN_ACCESSORY0, onScreenDisplaySettingsDataPriv.NumPages));
            break;
        case OnScreenDisplaySettings::PAGESWITCH_ACCESSORY1:
            ui->osdPageSlider->setValue(scaleSwitchChannel(ManualControlSettings::CHANNELMIN_ACCESSORY1, onScreenDisplaySettingsDataPriv.NumPages));
            break;
        case OnScreenDisplaySettings::PAGESWITCH_ACCESSORY2:
            ui->osdPageSlider->setValue(scaleSwitchChannel(ManualControlSettings::CHANNELMIN_ACCESSORY2, onScreenDisplaySettingsDataPriv.NumPages));
            break;
    }
}

void ConfigOsdWidget::updatePositionSlider()
{
    OnScreenDisplaySettings::DataFields onScreenDisplaySettingsDataPriv = osdSettingsObj->getData();

    switch(onScreenDisplaySettingsDataPriv.NumPages) {
    default:
    case 6:
        ui->osdPagePos6->setEnabled(true);
    // pass through
    case 5:
        ui->osdPagePos5->setEnabled(true);
    // pass through
    case 4:
        ui->osdPagePos4->setEnabled(true);
    // pass through
    case 3:
        ui->osdPagePos3->setEnabled(true);
    // pass through
    case 2:
        ui->osdPagePos2->setEnabled(true);
    // pass through
    case 1:
        ui->osdPagePos1->setEnabled(true);
    // pass through
    case 0:
    break;
    }

    switch(onScreenDisplaySettingsDataPriv.NumPages) {
    case 0:
        ui->osdPagePos1->setEnabled(false);
    // pass through
    case 1:
        ui->osdPagePos2->setEnabled(false);
    // pass through
    case 2:
        ui->osdPagePos3->setEnabled(false);
    // pass through
    case 3:
        ui->osdPagePos4->setEnabled(false);
    // pass through
    case 4:
        ui->osdPagePos5->setEnabled(false);
    // pass through
    case 5:
        ui->osdPagePos6->setEnabled(false);
    // pass through
    case 6:
    default:
    break;
    }
}

void ConfigOsdWidget::handle_button_0_1()
{
    copyOsdPage(0, 1);
}

void ConfigOsdWidget::handle_button_0_2()
{
    copyOsdPage(0, 2);
}

void ConfigOsdWidget::handle_button_0_3()
{
    copyOsdPage(0, 3);
}

void ConfigOsdWidget::handle_button_1_0()
{
    copyOsdPage(1, 0);
}

void ConfigOsdWidget::handle_button_1_2()
{
    copyOsdPage(1, 2);
}

void ConfigOsdWidget::handle_button_1_3()
{
    copyOsdPage(1, 3);
}

void ConfigOsdWidget::handle_button_2_0()
{
    copyOsdPage(2, 0);
}

void ConfigOsdWidget::handle_button_2_1()
{
    copyOsdPage(2, 1);
}

void ConfigOsdWidget::handle_button_2_3()
{
    copyOsdPage(2, 3);
}

void ConfigOsdWidget::handle_button_3_0()
{
    copyOsdPage(3, 0);
}

void ConfigOsdWidget::handle_button_3_1()
{
    copyOsdPage(3, 1);
}

void ConfigOsdWidget::handle_button_3_2()
{
    copyOsdPage(3, 2);
}

void ConfigOsdWidget::setupOsdPage(Ui::OsdPage * page, QWidget * page_widget, UAVObject * settings)
{
    page->setupUi(page_widget);
    QString name = settings->getName();
    
    // Alarms
    addUAVObjectToWidgetRelation(name, "Alarm", page->AlarmEnabled);
    page->AlarmEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->AlarmEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "AlarmPosX", page->AlarmX);
    addUAVObjectToWidgetRelation(name, "AlarmPosY", page->AlarmY);
    addUAVObjectToWidgetRelation(name, "AlarmFont", page->AlarmFont);
    addUAVObjectToWidgetRelation(name, "AlarmAlign", page->AlarmAlign);

    // Altitude Scale
    addUAVObjectToWidgetRelation(name, "AltitudeScale", page->AltitudeScaleEnabled);
    page->AltitudeScaleEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->AltitudeScaleEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "AltitudeScalePos", page->AltitudeScaleX);
    addUAVObjectToWidgetRelation(name, "AltitudeScaleAlign", page->AltitudeScaleAlign);
    addUAVObjectToWidgetRelation(name, "AltitudeScaleSource", page->AltitudeScaleSource);

    // Arming Status
    addUAVObjectToWidgetRelation(name, "ArmStatus", page->ArmStatusEnabled);
    page->ArmStatusEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->ArmStatusEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "ArmStatusPosX", page->ArmStatusX);
    addUAVObjectToWidgetRelation(name, "ArmStatusPosY", page->ArmStatusY);
    addUAVObjectToWidgetRelation(name, "ArmStatusFont", page->ArmStatusFont);
    addUAVObjectToWidgetRelation(name, "ArmStatusAlign", page->ArmStatusAlign);

    // Artificial Horizon
    addUAVObjectToWidgetRelation(name, "ArtificialHorizon", page->ArtificialHorizonEnabled);
    page->ArtificialHorizonEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->ArtificialHorizonEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "ArtificialHorizonMaxPitch", page->ArtificialHorizonMaxPitch);

    // Battery Voltage
    addUAVObjectToWidgetRelation(name, "BatteryVolt", page->BatteryVoltEnabled);
    page->BatteryVoltEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->BatteryVoltEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "BatteryVoltPosX", page->BatteryVoltX);
    addUAVObjectToWidgetRelation(name, "BatteryVoltPosY", page->BatteryVoltY);
    addUAVObjectToWidgetRelation(name, "BatteryVoltFont", page->BatteryVoltFont);
    addUAVObjectToWidgetRelation(name, "BatteryVoltAlign", page->BatteryVoltAlign);

    // Battery Current
    addUAVObjectToWidgetRelation(name, "BatteryCurrent", page->BatteryCurrentEnabled);
    page->BatteryCurrentEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->BatteryCurrentEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "BatteryCurrentPosX", page->BatteryCurrentX);
    addUAVObjectToWidgetRelation(name, "BatteryCurrentPosY", page->BatteryCurrentY);
    addUAVObjectToWidgetRelation(name, "BatteryCurrentFont", page->BatteryCurrentFont);
    addUAVObjectToWidgetRelation(name, "BatteryCurrentAlign", page->BatteryCurrentAlign);

    // Battery Consumed
    addUAVObjectToWidgetRelation(name, "BatteryConsumed", page->BatteryConsumedEnabled);
    page->BatteryConsumedEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->BatteryConsumedEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "BatteryConsumedPosX", page->BatteryConsumedX);
    addUAVObjectToWidgetRelation(name, "BatteryConsumedPosY", page->BatteryConsumedY);
    addUAVObjectToWidgetRelation(name, "BatteryConsumedFont", page->BatteryConsumedFont);
    addUAVObjectToWidgetRelation(name, "BatteryConsumedAlign", page->BatteryConsumedAlign);

    // Climb rate
    addUAVObjectToWidgetRelation(name, "ClimbRate", page->ClimbRateEnabled);
    page->ClimbRateEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->ClimbRateEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "ClimbRatePosX", page->ClimbRateX);
    addUAVObjectToWidgetRelation(name, "ClimbRatePosY", page->ClimbRateY);
    addUAVObjectToWidgetRelation(name, "ClimbRateFont", page->ClimbRateFont);
    addUAVObjectToWidgetRelation(name, "ClimbRateAlign", page->ClimbRateAlign);

    // Compass
    addUAVObjectToWidgetRelation(name, "Compass", page->CompassEnabled);
    page->CompassEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->CompassEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "CompassPos", page->CompassY);
    addUAVObjectToWidgetRelation(name, "CompassHomeDir", page->CompassHomeDir);
    page->CompassHomeDir->setProperty(trueString.toLatin1(), "Enabled");
    page->CompassHomeDir->setProperty(falseString.toLatin1(), "Disabled");

    // CPU
    addUAVObjectToWidgetRelation(name, "Cpu", page->CpuEnabled);
    page->CpuEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->CpuEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "CpuPosX", page->CpuX);
    addUAVObjectToWidgetRelation(name, "CpuPosY", page->CpuY);
    addUAVObjectToWidgetRelation(name, "CpuFont", page->CpuFont);
    addUAVObjectToWidgetRelation(name, "CpuAlign", page->CpuAlign);

    // Flight Mode
    addUAVObjectToWidgetRelation(name, "FlightMode", page->FlightModeEnabled);
    page->FlightModeEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->FlightModeEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "FlightModePosX", page->FlightModeX);
    addUAVObjectToWidgetRelation(name, "FlightModePosY", page->FlightModeY);
    addUAVObjectToWidgetRelation(name, "FlightModeFont", page->FlightModeFont);
    addUAVObjectToWidgetRelation(name, "FlightModeAlign", page->FlightModeAlign);

    // G force
    addUAVObjectToWidgetRelation(name, "GForce", page->GForceEnabled);
    page->GForceEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->GForceEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "GForcePosX", page->GForceX);
    addUAVObjectToWidgetRelation(name, "GForcePosY", page->GForceY);
    addUAVObjectToWidgetRelation(name, "GForceFont", page->GForceFont);
    addUAVObjectToWidgetRelation(name, "GForceAlign", page->GForceAlign);

    // GPS Status
    addUAVObjectToWidgetRelation(name, "GpsStatus", page->GpsStatusEnabled);
    page->GpsStatusEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->GpsStatusEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "GpsStatusPosX", page->GpsStatusX);
    addUAVObjectToWidgetRelation(name, "GpsStatusPosY", page->GpsStatusY);
    addUAVObjectToWidgetRelation(name, "GpsStatusFont", page->GpsStatusFont);
    addUAVObjectToWidgetRelation(name, "GpsStatusAlign", page->GpsStatusAlign);

    // GPS Latitude
    addUAVObjectToWidgetRelation(name, "GpsLat", page->GpsLatEnabled);
    page->GpsLatEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->GpsLatEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "GpsLatPosX", page->GpsLatX);
    addUAVObjectToWidgetRelation(name, "GpsLatPosY", page->GpsLatY);
    addUAVObjectToWidgetRelation(name, "GpsLatFont", page->GpsLatFont);
    addUAVObjectToWidgetRelation(name, "GpsLatAlign", page->GpsLatAlign);

    // GPS Longitude
    addUAVObjectToWidgetRelation(name, "GpsLon", page->GpsLonEnabled);
    page->GpsLonEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->GpsLonEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "GpsLonPosX", page->GpsLonX);
    addUAVObjectToWidgetRelation(name, "GpsLonPosY", page->GpsLonY);
    addUAVObjectToWidgetRelation(name, "GpsLonFont", page->GpsLonFont);
    addUAVObjectToWidgetRelation(name, "GpsLonAlign", page->GpsLonAlign);

    // GPS MGRS Location
    addUAVObjectToWidgetRelation(name, "GpsMgrs", page->GpsMgrsEnabled);
    page->GpsMgrsEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->GpsMgrsEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "GpsMgrsPosX", page->GpsMgrsX);
    addUAVObjectToWidgetRelation(name, "GpsMgrsPosY", page->GpsMgrsY);
    addUAVObjectToWidgetRelation(name, "GpsMgrsFont", page->GpsMgrsFont);
    addUAVObjectToWidgetRelation(name, "GpsMgrsAlign", page->GpsMgrsAlign);

    // Home Distance
    addUAVObjectToWidgetRelation(name, "HomeDistance", page->HomeDistanceEnabled);
    page->HomeDistanceEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->HomeDistanceEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "HomeDistancePosX", page->HomeDistanceX);
    addUAVObjectToWidgetRelation(name, "HomeDistancePosY", page->HomeDistanceY);
    addUAVObjectToWidgetRelation(name, "HomeDistanceFont", page->HomeDistanceFont);
    addUAVObjectToWidgetRelation(name, "HomeDistanceAlign", page->HomeDistanceAlign);
    addUAVObjectToWidgetRelation(name, "HomeDistanceShowText", page->HomeDistanceShowText);
    page->HomeDistanceShowText->setProperty(trueString.toLatin1(), "Enabled");
    page->HomeDistanceShowText->setProperty(falseString.toLatin1(), "Disabled");

    // RSSI
    addUAVObjectToWidgetRelation(name, "Rssi", page->RssiEnabled);
    page->RssiEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->RssiEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "RssiPosX", page->RssiX);
    addUAVObjectToWidgetRelation(name, "RssiPosY", page->RssiY);
    addUAVObjectToWidgetRelation(name, "RssiFont", page->RssiFont);
    addUAVObjectToWidgetRelation(name, "RssiAlign", page->RssiAlign);

    // Speed Scale
    addUAVObjectToWidgetRelation(name, "SpeedScale", page->SpeedScaleEnabled);
    page->SpeedScaleEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->SpeedScaleEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "SpeedScalePos", page->SpeedScaleX);
    addUAVObjectToWidgetRelation(name, "SpeedScaleAlign", page->SpeedScaleAlign);
    addUAVObjectToWidgetRelation(name, "SpeedScaleSource", page->SpeedScaleSource);

    // RSSI
    addUAVObjectToWidgetRelation(name, "Time", page->TimeEnabled);
    page->TimeEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->TimeEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "TimePosX", page->TimeX);
    addUAVObjectToWidgetRelation(name, "TimePosY", page->TimeY);
    addUAVObjectToWidgetRelation(name, "TimeFont", page->TimeFont);
    addUAVObjectToWidgetRelation(name, "TimeAlign", page->TimeAlign);
    addUAVObjectToWidgetRelation(name, "RssiShowText", page->RssiShowText);
    page->RssiShowText->setProperty(trueString.toLatin1(), "Enabled");
    page->RssiShowText->setProperty(falseString.toLatin1(), "Disabled");

    // Map
    addUAVObjectToWidgetRelation(name, "Map", page->MapEnabled);
    page->MapEnabled->setProperty(trueString.toLatin1(), "Enabled");
    page->MapEnabled->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "MapCenterMode", page->mapCenterMode);
    addUAVObjectToWidgetRelation(name, "MapShowWp", page->mapShowWp);
    page->mapShowWp->setProperty(trueString.toLatin1(), "Enabled");
    page->mapShowWp->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "MapShowUavHome", page->mapShowUavHome);
    page->mapShowUavHome->setProperty(trueString.toLatin1(), "Enabled");
    page->mapShowUavHome->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "MapShowTablet", page->mapShowTablet);
    page->mapShowTablet->setProperty(trueString.toLatin1(), "Enabled");
    page->mapShowTablet->setProperty(falseString.toLatin1(), "Disabled");
    addUAVObjectToWidgetRelation(name, "MapWidthMeters", page->mapWidthMeters);
    addUAVObjectToWidgetRelation(name, "MapHeightMeters", page->mapHeightMeters);
    addUAVObjectToWidgetRelation(name, "MapWidthPixels", page->mapWidthPixels);
    addUAVObjectToWidgetRelation(name, "MapHeightPixels", page->mapHeightPixels);
}

void ConfigOsdWidget::copyOsdPage(int to, int from)
{
    QList<QCheckBox*> checkboxes = pages[from]->findChildren<QCheckBox*>();

    foreach (QCheckBox *checkbox, checkboxes)
    {
         QCheckBox *cb_to = pages[to]->findChild<QCheckBox *>(checkbox->objectName());
         if (cb_to != NULL) {
             cb_to->setChecked(checkbox->checkState());
         }
    }

    QList<QSpinBox*> spinboxes = pages[from]->findChildren<QSpinBox*>();

    foreach (QSpinBox *spinbox, spinboxes)
    {
         QSpinBox *sb_to = pages[to]->findChild<QSpinBox *>(spinbox->objectName());
         if (sb_to != NULL) {
             sb_to->setValue(spinbox->value());
         }
    }

    QList<QComboBox*> comboboxes = pages[from]->findChildren<QComboBox*>();

    foreach (QComboBox *combobox, comboboxes)
    {
         QComboBox *cmb_to = pages[to]->findChild<QComboBox *>(combobox->objectName());
         if (cmb_to != NULL) {
             cmb_to->setCurrentIndex(combobox->currentIndex());
         }
    }
}

void ConfigOsdWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
}

void ConfigOsdWidget::enableControls(bool enable)
{
    Q_UNUSED(enable);
}

/**
 * @brief ModuleSettingsForm::getWidgetFromVariant Reimplements getWidgetFromVariant. This version supports "FalseString".
 * @param widget pointer to the widget from where to get the value
 * @param scale scale to be used on the assignement
 * @return returns the value of the widget times the scale
 */
QVariant ConfigOsdWidget::getVariantFromWidget(QWidget * widget, double scale)
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
bool ConfigOsdWidget::setWidgetFromVariant(QWidget *widget, QVariant value, double scale)
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
/**
 * @}
 * @}
 */
 
