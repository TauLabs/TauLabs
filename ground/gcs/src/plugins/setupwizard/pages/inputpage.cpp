/**
 ******************************************************************************
 *
 * @file       inputpage.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup SetupWizard Setup Wizard
 * @{
 * @brief
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

#include "inputpage.h"
#include "ui_inputpage.h"
#include "setupwizard.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"

InputPage::InputPage(SetupWizard *wizard, QWidget *parent) :
    AbstractWizardPage(wizard, parent),

    ui(new Ui::InputPage)
{
    ui->setupUi(this);
}

InputPage::~InputPage()
{
    delete ui;
}

bool InputPage::validatePage()
{
    if (ui->pwmButton->isChecked()) {
        getWizard()->setInputType(Core::IBoardType::INPUT_TYPE_PWM);
    } else if (ui->ppmButton->isChecked()) {
        getWizard()->setInputType(Core::IBoardType::INPUT_TYPE_PPM);
    } else if (ui->sbusButton->isChecked()) {
        getWizard()->setInputType(Core::IBoardType::INPUT_TYPE_SBUS);
    } else if (ui->spectrumButton->isChecked()) {
        getWizard()->setInputType(Core::IBoardType::INPUT_TYPE_DSM);
    } else if (ui->hottsumdButton->isChecked()) {
        getWizard()->setInputType(Core::IBoardType::INPUT_TYPE_HOTTSUMD);
    } else {
        getWizard()->setInputType(Core::IBoardType::INPUT_TYPE_PWM);
    }
    getWizard()->setRestartNeeded(getWizard()->isRestartNeeded() || restartNeeded(getWizard()->getInputType()));

    return true;
}

/**
 * @brief InputPage::restartNeeded Check if the requested input type is currently
 * selected
 * @param selectedType the requested input type
 * @return true if changing input type and should restart, false otherwise
 */
bool InputPage::restartNeeded(Core::IBoardType::InputType selectedType)
{
    Core::IBoardType* board = getWizard()->getControllerType();
    Q_ASSERT(board);
    if (!board)
        return true;

    // Map from the enums used in SetupWizard to IBoardType
    Core::IBoardType::InputType boardInputType = board->getInputOnPort();
    return (selectedType != boardInputType);
}
