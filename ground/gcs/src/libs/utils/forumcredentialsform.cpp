/**
 ******************************************************************************
 * @file       forumcredentialsform.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup Utils
 * @{
 * @addtogroup ForumCredentialsForm
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

#include "forumcredentialsform.h"
#include "ui_forumcredentialsform.h"

namespace Utils {

ForumCredentialsForm::ForumCredentialsForm(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ForumCredentialsForm)
{
    ui->setupUi(this);
}

ForumCredentialsForm::~ForumCredentialsForm()
{
    delete ui;
}

void ForumCredentialsForm::setPassword(QString value)
{
    ui->passwordLE->setText(value);
    ui->saveCredentialsCB->setChecked(true);
}

void ForumCredentialsForm::setUserName(QString value)
{
    ui->userNameLE->setText(value);
    ui->saveCredentialsCB->setChecked(true);
}

QString ForumCredentialsForm::getUserName()
{
    return ui->userNameLE->text();
}

QString ForumCredentialsForm::getPassword()
{
    return ui->passwordLE->text();
}

void ForumCredentialsForm::setObservations(QString value)
{
    ui->observationsTE->setText(value);
}

void ForumCredentialsForm::setAircraftDescription(QString value)
{
    ui->aircraftDescriptionTE->setText(value);
}

QString ForumCredentialsForm::getObservations()
{
    return ui->observationsTE->toPlainText();
}

QString ForumCredentialsForm::getAircraftDescription()
{
    return ui->aircraftDescriptionTE->toPlainText();
}

bool ForumCredentialsForm::getSaveCredentials()
{
    return ui->saveCredentialsCB->isChecked();
}

}
