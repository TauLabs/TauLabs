/**
 ******************************************************************************
 * @file       foruminteractionform.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup Utils
 * @{
 * @addtogroup ForumInteractionForm
 * @{
 * @brief Utility to present a form to the user where he can input is forum
 * credentials and aircraft details
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

#include "foruminteractionform.h"
#include "ui_foruminteractionform.h"

namespace Utils {

ForumInteractionForm::ForumInteractionForm(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ForumInteractionForm)
{
    ui->setupUi(this);
}

ForumInteractionForm::~ForumInteractionForm()
{
    delete ui;
}

void ForumInteractionForm::setPassword(QString value)
{
    ui->passwordLE->setText(value);
    ui->saveCredentialsCB->setChecked(true);
}

void ForumInteractionForm::setUserName(QString value)
{
    ui->userNameLE->setText(value);
    ui->saveCredentialsCB->setChecked(true);
}

QString ForumInteractionForm::getUserName()
{
    return ui->userNameLE->text();
}

QString ForumInteractionForm::getPassword()
{
    return ui->passwordLE->text();
}

void ForumInteractionForm::setObservations(QString value)
{
    ui->observationsTE->setText(value);
}

void ForumInteractionForm::setAircraftDescription(QString value)
{
    ui->aircraftDescriptionTE->setText(value);
}

QString ForumInteractionForm::getObservations()
{
    return ui->observationsTE->toPlainText();
}

QString ForumInteractionForm::getAircraftDescription()
{
    return ui->aircraftDescriptionTE->toPlainText();
}

bool ForumInteractionForm::getSaveCredentials()
{
    return ui->saveCredentialsCB->isChecked();
}

}
