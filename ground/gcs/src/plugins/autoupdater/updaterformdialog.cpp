/**
 ******************************************************************************
 * @file       updaterformdialog.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup [Group]
 * @{
 * @addtogroup updaterFormDialog
 * @{
 * @brief [Brief]
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

#include "updaterformdialog.h"
#include "ui_updaterformdialog.h"
#include <QDebug>

updaterFormDialog::updaterFormDialog(QString releaseDetails, bool changedUAVO, QVariant context, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::updaterFormDialog),
    context(context)
{
    ui->setupUi(this);
    ui->updateDetails->setText(releaseDetails);
    ui->updateDetails2->setText(releaseDetails);
    ui->changedUAVOWarning->setVisible(changedUAVO);
    this->setAttribute(Qt::WA_DeleteOnClose, true);
    connect(ui->update, SIGNAL(clicked(bool)), this, SLOT(onStartUpdate()));
    connect(ui->downloadHide, SIGNAL(clicked(bool)), this, SLOT(onHide()));
    connect(ui->downloadCancel, SIGNAL(clicked(bool)), this, SIGNAL(cancelDownload()));
    connect(ui->downloadCancel, SIGNAL(clicked(bool)), this, SLOT(onCancelDownload()));
    connect(ui->cancel, SIGNAL(clicked(bool)), this, SLOT(onCancel()));
    this->setAttribute(Qt::WA_DeleteOnClose);
}

updaterFormDialog::~updaterFormDialog()
{
    delete ui;
}

void updaterFormDialog::onStartUpdate()
{
    ui->stackedWidget->setCurrentIndex(1);
    emit startUpdate();
}

void updaterFormDialog::onHide()
{
    this->setVisible(false);
    QCoreApplication::processEvents();
}

void updaterFormDialog::onCancelDownload()
{
    ui->progressBar->setValue(0);
    ui->progress->setText("");
    ui->stackedWidget->setCurrentIndex(0);
}

void updaterFormDialog::onCancel()
{
    this->close();
}

void updaterFormDialog::setProgress(QString text)
{
    ui->progress->setText(text);
    QCoreApplication::processEvents();
}

void updaterFormDialog::setProgress(int value)
{
    ui->progressBar->setValue(value);
    QCoreApplication::processEvents();
}

void updaterFormDialog::setOperation(QString text)
{
    ui->operation->setText(text);
}

void updaterFormDialog::closeEvent(QCloseEvent *event)
{
    emit dialogAboutToClose(ui->dontBotherAgain->isChecked());
    QDialog::closeEvent(event);
}
