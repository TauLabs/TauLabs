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
#include "manualcontrolcommand.h"
#include "manualcontrolsettings.h"

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

    connect(ManualControlCommand::GetInstance(getObjectManager()),SIGNAL(objectUpdated(UAVObject*)),this,SLOT(movePageSlider()));
    connect(OnScreenDisplaySettings::GetInstance(getObjectManager()),SIGNAL(objectUpdated(UAVObject*)),this,SLOT(updatePositionSlider()));

    // Load UAVObjects to widget relations from UI file
    // using objrelation dynamic property
    autoLoadWidgets();

    // Refresh widget contents
    refreshWidgetsValues();

    // Prevent mouse wheel from changing values
    disableMouseWheelEvents();
}

ConfigOsdWidget::~ConfigOsdWidget()
{
    delete ui;
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

    qDebug() << "MOVE\n";
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
 
