/**
 ******************************************************************************
 *
 * @file       configurepage.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup RfmBindWizard Setup Wizard
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

#include "configurepage.h"
#include "ui_configurepage.h"

ConfigurePage::ConfigurePage(RfmBindWizard *wizard, QWidget *parent) :
    AbstractWizardPage(wizard, parent), ui(new Ui::ConfigurePage)
{
    ui->setupUi(this);

    // These lists are a property of the modem hardware (RFM22b) and
    // should be common amongst all the boards
    QStringList MaxRfPowerEnumOptions;
    MaxRfPowerEnumOptions.append("1.25");
    MaxRfPowerEnumOptions.append("1.6");
    MaxRfPowerEnumOptions.append("3.16");
    MaxRfPowerEnumOptions.append("6.3");
    MaxRfPowerEnumOptions.append("12.6");
    MaxRfPowerEnumOptions.append("25");
    MaxRfPowerEnumOptions.append("50");
    MaxRfPowerEnumOptions.append("100");
    ui->cbRfPower->clear();
    ui->cbRfPower->insertItems(0, MaxRfPowerEnumOptions);

    QStringList MaxRfSpeedEnumOptions;
    MaxRfSpeedEnumOptions.append("9600");
    MaxRfSpeedEnumOptions.append("19200");
    MaxRfSpeedEnumOptions.append("32000");
    MaxRfSpeedEnumOptions.append("64000");
    MaxRfSpeedEnumOptions.append("100000");
    MaxRfSpeedEnumOptions.append("192000");
    ui->cbBaudRate->clear();
    ui->cbBaudRate->insertItems(0, MaxRfSpeedEnumOptions);
    ui->cbBaudRate->setCurrentText("100000");

    ui->cbLinkMode->clear();
    ui->cbLinkMode->addItem("Telemetry Only", QVariant(Core::IBoardType::LINK_TELEM));
    ui->cbLinkMode->addItem("Telemetry and PPM", QVariant(Core::IBoardType::LINK_TELEM_PPM));
    ui->cbLinkMode->addItem("PPM Only", QVariant(Core::IBoardType::LINK_PPM));
    ui->cbBaudRate->setCurrentIndex(0);

    ui->sbMinChannel->setValue(0);
    ui->sbMaxChannel->setValue(250);
}

ConfigurePage::~ConfigurePage()
{
    delete ui;
}

void ConfigurePage::initializePage()
{
    emit completeChanged();
}

bool ConfigurePage::isComplete() const
{
    return true;
}

//! Store the configuration data in the wizard
bool ConfigurePage::validatePage()
{
    quint32 bps = ui->cbBaudRate->currentText().toInt();
    quint32 maxRfPower = ui->cbRfPower->currentText().toFloat();
    quint8 minChannel = ui->sbMinChannel->value();
    quint8 maxChannel = ui->sbMaxChannel->value();
    enum Core::IBoardType::LinkMode linkMode;
    linkMode = (enum Core::IBoardType::LinkMode) ui->cbLinkMode->currentData().value<int>();

    getWizard()->setMaxBps(bps);
    getWizard()->setMaxRfPower(maxRfPower);
    getWizard()->setMinChannel(minChannel);
    getWizard()->setMaxChannel(maxChannel);
    getWizard()->setLinkMode(linkMode);

    return true;
}
