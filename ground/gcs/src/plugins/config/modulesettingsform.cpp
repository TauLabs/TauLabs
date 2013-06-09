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

moduleSettingsForm::moduleSettingsForm(QWidget *parent) :
    ConfigTaskWidget(parent),
    ui(new Ui::Module_Settings_Widget)
{
    ui->setupUi(this);

//    connect(ui->channelMin,SIGNAL(valueChanged(int)),this,SLOT(minMaxUpdated()));
//    connect(ui->channelMax,SIGNAL(valueChanged(int)),this,SLOT(minMaxUpdated()));
//    connect(ui->channelGroup,SIGNAL(currentIndexChanged(int)),this,SLOT(groupUpdated()));
//    connect(ui->channelNeutral,SIGNAL(valueChanged(int)), this, SLOT(neutralUpdated(int)));

//    // This is awkward but since we want the UI to be a dropdown but the field is not an enum
//    // it breaks the UAUVObject widget relation of the task gadget.  Running the data through
//    // a spin box fixes this
//    connect(ui->channelNumberDropdown,SIGNAL(currentIndexChanged(int)),this,SLOT(channelDropdownUpdated(int)));
//    connect(ui->channelNumber,SIGNAL(valueChanged(int)),this,SLOT(channelNumberUpdated(int)));

    disableMouseWheelEvents();
}


moduleSettingsForm::~moduleSettingsForm()
{
    delete ui;
}

void moduleSettingsForm::setName(QString &name)
{
//    ui->channelName->setText(name);
//    QFontMetrics metrics(ui->channelName->font());
//    int width=metrics.width(name)+5;
//    foreach(moduleSettingsForm * form,parent()->findChildren<moduleSettingsForm*>())
//    {
//        if(form==this)
//            continue;
//        if(form->ui->channelName->minimumSize().width()<width)
//            form->ui->channelName->setMinimumSize(width,0);
//        else
//            width=form->ui->channelName->minimumSize().width();
//    }
//    ui->channelName->setMinimumSize(width,0);
}

/**
  * Update the direction of the slider and boundaries
  */
void moduleSettingsForm::minMaxUpdated()
{
//    bool reverse = ui->channelMin->value() > ui->channelMax->value();
//    if(reverse) {
//        ui->channelNeutral->setMinimum(ui->channelMax->value());
//        ui->channelNeutral->setMaximum(ui->channelMin->value());
//    } else {
//        ui->channelNeutral->setMinimum(ui->channelMin->value());
//        ui->channelNeutral->setMaximum(ui->channelMax->value());
//    }
//    ui->channelRev->setChecked(reverse);
//    ui->channelNeutral->setInvertedAppearance(reverse);
//    ui->channelNeutral->setInvertedControls(reverse);
}

void moduleSettingsForm::neutralUpdated(int newval)
{
//    ui->neutral->setText(QString::number(newval));
}

/**
  * Update the channel options based on the selected receiver type
  *
  * I fully admit this is terrible practice to embed data within UI
  * like this.  Open to suggestions. JC 2011-09-07
  */
void moduleSettingsForm::groupUpdated()
{
//    ui->channelNumberDropdown->clear();
//    ui->channelNumberDropdown->addItem("Disabled");

//    quint8 count = 0;

//    switch(ui->channelGroup->currentIndex()) {
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
//        ui->channelNumberDropdown->addItem(QString(tr("Chan %1").arg(i+1)));

//    ui->channelNumber->setMaximum(count);
//    ui->channelNumber->setMinimum(0);
}

/**
  * Update the dropdown from the hidden control
  */
void moduleSettingsForm::channelDropdownUpdated(int newval)
{
//    ui->channelNumber->setValue(newval);
}

/**
  * Update the hidden control from the dropdown
  */
void moduleSettingsForm::channelNumberUpdated(int newval)
{
//    ui->channelNumberDropdown->setCurrentIndex(newval);
}
