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

#include "modulesettings.h"

// Define static variables
QString moduleSettingsForm::trueString("TrueString");
QString moduleSettingsForm::falseString("FalseString");

/**
 * @brief moduleSettingsForm::moduleSettingsForm Constructor
 * @param parent
 */
moduleSettingsForm::moduleSettingsForm(QWidget *parent) :
    ConfigTaskWidget(parent),
    moduleSettingsWidget(new Ui::Module_Settings_Widget)
{
    moduleSettingsWidget->setupUi(this);

    ModuleSettings moduleSettings;
    QString moduleSettingsName = moduleSettings.getName();

    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", moduleSettingsWidget->gb_Airspeed, ModuleSettings::ADMINSTATE_AIRSPEED); // TODO: Set the widget based on the UAVO getName() and getField() methods.
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", moduleSettingsWidget->gb_Battery, ModuleSettings::ADMINSTATE_BATTERY); // TODO: Set the widget based on the UAVO getName() and getField() methods.
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", moduleSettingsWidget->gb_VibrationAnalysis, ModuleSettings::ADMINSTATE_VIBRATIONANALYSIS); // TODO: Set the widget based on the UAVO getName() and getField() methods.
    addUAVObjectToWidgetRelation(moduleSettingsName, "AdminState", moduleSettingsWidget->gb_Airspeed, ModuleSettings::ADMINSTATE_AIRSPEED); // TODO: Set the widget based on the UAVO getName() and getField() methods.

    // Set text properties. The second argument is the UAVO field that corresponds to the true state
    moduleSettingsWidget->gb_Airspeed->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_Airspeed->setProperty(falseString.toAscii(), "Disabled");

    moduleSettingsWidget->gb_Battery->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_Battery->setProperty(falseString.toAscii(), "Disabled");

    moduleSettingsWidget->gb_VibrationAnalysis->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_VibrationAnalysis->setProperty(falseString.toAscii(), "Disabled");

    moduleSettingsWidget->gb_GPS->setProperty(trueString.toAscii(), "Enabled");
    moduleSettingsWidget->gb_GPS->setProperty(falseString.toAscii(), "Disabled");

    disableMouseWheelEvents();
}


moduleSettingsForm::~moduleSettingsForm()
{
    delete moduleSettingsWidget;
}

void moduleSettingsForm::setName(QString &name)
{
//    moduleSettingsWidget->channelName->setText(name);
//    QFontMetrics metrics(moduleSettingsWidget->channelName->font());
//    int width=metrics.width(name)+5;
//    foreach(moduleSettingsForm * form,parent()->findChildren<moduleSettingsForm*>())
//    {
//        if(form==this)
//            continue;
//        if(form->moduleSettingsWidget->channelName->minimumSize().width()<width)
//            form->moduleSettingsWidget->channelName->setMinimumSize(width,0);
//        else
//            width=form->moduleSettingsWidget->channelName->minimumSize().width();
//    }
//    moduleSettingsWidget->channelName->setMinimumSize(width,0);
}

/**
  * Update the direction of the slider and boundaries
  */
void moduleSettingsForm::minMaxUpdated()
{
//    bool reverse = moduleSettingsWidget->channelMin->value() > moduleSettingsWidget->channelMax->value();
//    if(reverse) {
//        moduleSettingsWidget->channelNeutral->setMinimum(moduleSettingsWidget->channelMax->value());
//        moduleSettingsWidget->channelNeutral->setMaximum(moduleSettingsWidget->channelMin->value());
//    } else {
//        moduleSettingsWidget->channelNeutral->setMinimum(moduleSettingsWidget->channelMin->value());
//        moduleSettingsWidget->channelNeutral->setMaximum(moduleSettingsWidget->channelMax->value());
//    }
//    moduleSettingsWidget->channelRev->setChecked(reverse);
//    moduleSettingsWidget->channelNeutral->setInvertedAppearance(reverse);
//    moduleSettingsWidget->channelNeutral->setInvertedControls(reverse);
}

void moduleSettingsForm::neutralUpdated(int newval)
{
//    moduleSettingsWidget->neutral->setText(QString::number(newval));
}

/**
  * Update the channel options based on the selected receiver type
  *
  */
void moduleSettingsForm::groupUpdated()
{
//    moduleSettingsWidget->channelNumberDropdown->clear();
//    moduleSettingsWidget->channelNumberDropdown->addItem("Disabled");

//    quint8 count = 0;

//    switch(moduleSettingsWidget->channelGroup->currentIndex()) {
//    case -1: // Nothing selected
//        count = 0;
//        break;
//    case ManualControlSettings::CHANNELGROUPS_PWM:
//        count = 8; // Need to make this 6 for CC
//        break;
//    case ManualControlSettings::CHANNELGROUPS_PPM:
//    case ManualControlSettings::CHANNELGROUPS_DSMMAINPORT:
//    case ManualControlSettings::CHANNELGROUPS_DSMFLEXIPORT:
//        count = 12;
//        break;
//    case ManualControlSettings::CHANNELGROUPS_SBUS:
//        count = 18;
//        break;
//    case ManualControlSettings::CHANNELGROUPS_GCS:
//        count = GCSReceiver::CHANNEL_NUMELEM;
//        break;
//    case ManualControlSettings::CHANNELGROUPS_NONE:
//        count = 0;
//        break;
//    default:
//        Q_ASSERT(0);
//    }

//    for (int i = 0; i < count; i++)
//        moduleSettingsWidget->channelNumberDropdown->addItem(QString(tr("Chan %1").arg(i+1)));

//    moduleSettingsWidget->channelNumber->setMaximum(count);
//    moduleSettingsWidget->channelNumber->setMinimum(0);
}

/**
  * Update the dropdown from the hidden control
  */
void moduleSettingsForm::channelDropdownUpdated(int newval)
{
//    moduleSettingsWidget->channelNumber->setValue(newval);
}

/**
  * Update the hidden control from the dropdown
  */
void moduleSettingsForm::channelNumberUpdated(int newval)
{
//    moduleSettingsWidget->channelNumberDropdown->setCurrentIndex(newval);
}


/**
 * @brief moduleSettingsForm::getWidgetFromVariant Remiplmements getWidgetFromVariant. This version supports "FalseString".
 * @param widget pointer to the widget from where to get the value
 * @param scale scale to be used on the assignement
 * @return returns the value of the widget times the scale
 */
QVariant moduleSettingsForm::getVariantFromWidget(QWidget * widget, double scale)
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
 * @brief moduleSettingsForm::setWidgetFromVariant Remiplmements setWidgetFromVariant. This version supports "FalseString".
 * @param widget pointer for the widget to set
 * @param scale scale to be used on the assignement
 * @param value value to be used on the assignement
 * @return returns true if the assignement was successfull
 */
bool moduleSettingsForm::setWidgetFromVariant(QWidget *widget, QVariant value, double scale)
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
