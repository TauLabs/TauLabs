/**
 ******************************************************************************
 *
 * @file       inputchannelform.c
 * @author     OpenPilot, http://openpilot.org, Copyright (C) 2012
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
 * @author     dRonin, http://dronin.org Copyright (C) 2015
 * @see        The GNU Public License (GPL) Version 3
 *
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

#include "inputchannelform.h"
#include "ui_inputchannelform.h"

#include "manualcontrolsettings.h"
#include "gcsreceiver.h"

#include <coreplugin/iboardtype.h>

inputChannelForm::inputChannelForm(QWidget *parent, bool showlegend, bool showSlider, ChannelFunc chanType):
    ConfigTaskWidget(parent),
    ui(new Ui::inputChannelForm),
    m_chanType(chanType)
{
    ui->setupUi(this);
    
    //The first time through the loop, keep the legend. All other times, delete it.
    if(!showlegend)
    {
        layout()->removeWidget(ui->legend0);
        layout()->removeWidget(ui->legend1);
        layout()->removeWidget(ui->legend2);
        layout()->removeWidget(ui->legend3);
        layout()->removeWidget(ui->legend4);
        layout()->removeWidget(ui->legend5);
        layout()->removeWidget(ui->legend6);
        delete ui->legend0;
        delete ui->legend1;
        delete ui->legend2;
        delete ui->legend3;
        delete ui->legend4;
        delete ui->legend5;
        delete ui->legend6;
    }

    if(!showSlider)
    {
        ui->channelNeutral->setHidden(true);
    }

    // Connect slots
    connect(ui->channelMin,SIGNAL(valueChanged(int)),this,SLOT(minMaxUpdated()));
    connect(ui->channelMax,SIGNAL(valueChanged(int)),this,SLOT(minMaxUpdated()));
    connect(ui->channelGroup,SIGNAL(currentIndexChanged(int)),this,SLOT(groupUpdated()));

    // This is awkward but since we want the UI to be a dropdown but the field is not an enum
    // so it breaks the UAUVObject widget relation of the task gadget.  Running the data through
    // a spin box fixes this
    connect(ui->channelNumberDropdown,SIGNAL(currentIndexChanged(int)),this,SLOT(channelDropdownUpdated(int)));
    connect(ui->channelNumber,SIGNAL(valueChanged(int)),this,SLOT(channelNumberUpdated(int)));

    disableMouseWheelEvents();
}


inputChannelForm::~inputChannelForm()
{
    delete ui;
}

void inputChannelForm::setName(QString &name)
{
    ui->channelName->setText(name);
    QFontMetrics metrics(ui->channelName->font());
    int width=metrics.width(name)+5;
    foreach(inputChannelForm * form,parent()->findChildren<inputChannelForm*>())
    {
        if(form==this)
            continue;
        if(form->ui->channelName->minimumSize().width()<width)
            form->ui->channelName->setMinimumSize(width,0);
        else
            width=form->ui->channelName->minimumSize().width();
    }
    ui->channelName->setMinimumSize(width,0);
}

/**
  * Update the direction of the slider and boundaries
  */
void inputChannelForm::minMaxUpdated()
{
    if (ui->channelMin->value() != ui->channelMax->value()) {
        bool reverse = (ui->channelMin->value() > ui->channelMax->value());
        if(reverse) {
            ui->channelNeutral->setMinimum(ui->channelMax->value());
            ui->channelNeutral->setMaximum(ui->channelMin->value());

            // Set the QSlider groove colors so that the fill is on the side of the minimum value
            ui->channelNeutral->setProperty("state", "inverted");
        } else {
            ui->channelNeutral->setMinimum(ui->channelMin->value());
            ui->channelNeutral->setMaximum(ui->channelMax->value());

            // Set the QSlider groove colors so that the fill is on the side of the minimum value
            ui->channelNeutral->setProperty("state", "normal");
        }
        ui->channelNeutral->setInvertedAppearance(reverse);

        // make sure slider is enabled (can be removed when "else" below is removed)
        ui->channelNeutral->setEnabled(true);
    }
    else {
        // when the min and max is equal, disable this slider to prevent crash
        // from Qt bug: https://bugreports.qt.io/browse/QTBUG-43398
        ui->channelNeutral->setMinimum(ui->channelMin->value() - 1);
        ui->channelNeutral->setMaximum(ui->channelMax->value() + 1);
        ui->channelNeutral->setEnabled(false);
    }

    // Force refresh of style sheet.
    ui->channelNeutral->setStyle(QApplication::style());
}

/**
  * Update the channel options based on the selected receiver type
  *
  * I fully admit this is terrible practice to embed data within UI
  * like this.  Open to suggestions. JC 2011-09-07
  */
void inputChannelForm::groupUpdated()
{
    ui->channelNumberDropdown->clear();
    ui->channelNumberDropdown->addItem("Disabled");

    quint8 count = 0;

    switch (m_chanType) {

    case inputChannelForm::CHANNELFUNC_RC:
        {
            switch (ui->channelGroup->currentIndex()) {
            case -1: // Nothing selected
                count = 0;
                break;
            case ManualControlSettings::CHANNELGROUPS_PWM:
                count = 8; // Need to make this 6 for CC
                break;
            case ManualControlSettings::CHANNELGROUPS_PPM:
            case ManualControlSettings::CHANNELGROUPS_DSM:
                count = 12;
                break;
            case ManualControlSettings::CHANNELGROUPS_SBUS:
                count = 18;
                break;
            case ManualControlSettings::CHANNELGROUPS_RFM22B:
                count = 9;
                break;
            case ManualControlSettings::CHANNELGROUPS_OPENLRS:
                count = 8;
                break;
            case ManualControlSettings::CHANNELGROUPS_GCS:
                count = GCSReceiver::CHANNEL_NUMELEM;
                break;
            case ManualControlSettings::CHANNELGROUPS_HOTTSUM:
                count = 32;
                break;
            case ManualControlSettings::CHANNELGROUPS_NONE:
                count = 0;
                break;
            default:
                Q_ASSERT(0);
            }
        }
        break;


    case inputChannelForm::CHANNELFUNC_RSSI:
        {
            switch (ui->channelGroup->currentIndex()) {
            case -1: // Nothing selected
                count = 0;
                break;
            case ManualControlSettings::RSSITYPE_PWM:
                count = 8;
                break;
            case ManualControlSettings::RSSITYPE_PPM:
                count = 12;
                break;
            case ManualControlSettings::RSSITYPE_SBUS:
                count = 18;
                break;
            case ManualControlSettings::RSSITYPE_ADC:
                count = 9;
                break;
            case ManualControlSettings::RSSITYPE_OPENLRS:
                count = 8;
                break;
            case ManualControlSettings::RSSITYPE_FRSKYPWM:
                count = 1;
                break;
            case ManualControlSettings::RSSITYPE_NONE:
                count = 0;
                break;
            default:
                Q_ASSERT(0);
            }
        }
        break;

    }

    if (m_chanType == inputChannelForm::CHANNELFUNC_RSSI &&
            ui->channelGroup->currentIndex() == ManualControlSettings::RSSITYPE_ADC) {
        QStringList names;
        Core::IBoardType *board = utilMngr->getBoardType();
        if (board)
            names = board->getAdcNames();

        for (int i = 0; i < count; i++) {
            QString name;
            if (i < names.length())
                name = names[i] + QString(" (ADC%1)").arg(i);
            else if (names.isEmpty())
                name = QString("ADC%1").arg(i);
            else
                name = QString("N/A (ADC%1)").arg(i);

            ui->channelNumberDropdown->addItem(name);
        }
    } else {
        for (int i = 0; i < count; i++)
            ui->channelNumberDropdown->addItem(QString(tr("Chan %1").arg(i+1)));
    }

    ui->channelNumber->setMaximum(count);
    ui->channelNumber->setMinimum(0);
}

/**
  * Update the dropdown from the hidden control
  */
void inputChannelForm::channelDropdownUpdated(int newval)
{
    ui->channelNumber->setValue(newval);
}

/**
  * Update the hidden control from the dropdown
  */
void inputChannelForm::channelNumberUpdated(int newval)
{
    ui->channelNumberDropdown->setCurrentIndex(newval);
}
