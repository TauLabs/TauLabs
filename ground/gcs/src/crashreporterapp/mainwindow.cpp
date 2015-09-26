/**
 ******************************************************************************
 * @file       mainwindow.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 * @addtogroup crashreporterapp
 * @{
 * @addtogroup MainWindow
 * @{
 * @brief Sends a crash report to the forum
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

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDesktopWidget>
#include <QDesktopServices>
#include <QDir>
#include <QDebug>
#include <QDateTime>
#include <QFileDialog>
#include "utils/phpbb.h"
#include <QMessageBox>

#define FORUM_SHARING_FORUM 26
#define FORUM_SHARING_THREAD 830

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QRect screenGeometry = QApplication::desktop()->screenGeometry();
    int x = (screenGeometry.width() - width()) / 2;
    int y = (screenGeometry.height() - height()) / 2;
    move(x, y);
    connect(ui->textBrowser, SIGNAL(anchorClicked(QUrl)), this, SLOT(onShowReport()));
    connect(ui->pb_Save, SIGNAL(clicked(bool)), this, SLOT(onSaveReport()));
    connect(ui->pb_Cancel, SIGNAL(clicked(bool)), this, SLOT(close()));
    connect(ui->pb_CancelSend, SIGNAL(clicked(bool)), this, SLOT(onCancelSend()));
    connect(ui->pb_OK, SIGNAL(clicked(bool)), this, SLOT(onOkSend()));
    connect(ui->pb_SendReport, SIGNAL(clicked(bool)), this, SLOT(onSendReport()));

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setDumpFile(QString filename)
{
    dumpFile = filename;
}

void MainWindow::onShowReport()
{
    QString txtFile(QDir::temp().absolutePath() + QDir::separator() + "dump.txt");
    QFile::copy(dumpFile, txtFile);
    QDesktopServices::openUrl(QUrl("file:///" + txtFile, QUrl::TolerantMode));
}

void MainWindow::onSaveReport()
{
    QString file = QFileDialog::getSaveFileName(this, "Choose the filename", QDir::homePath(), "Dump File (*.dmp)");
    if(file.isEmpty())
        return;
    QString trash;
    QFile::copy(dumpFile, file);
}

void MainWindow::onOkSend()
{
    ui->stackedWidget->setCurrentIndex(1);
}

void MainWindow::onSendReport()
{
    ui->stackedWidget->setCurrentIndex(2);
    Utils::PHPBB php("http://forum.taulabs.org", this);
    if (!php.login(ui->le_Username->text(), ui->le_Password->text())) {
       QMessageBox::warning(this, tr("Forum login"), tr("Forum login failed, probably wrong username or password"));
       ui->stackedWidget->setCurrentIndex(1);
       return;
    }
    QList<Utils::PHPBB::fileAttachment> list;
    Utils::PHPBB::fileAttachment fileAttach;
    fileAttach.fileComment = "Crash Report";
    QFile file(dumpFile);
    file.open(QIODevice::ReadOnly);
    fileAttach.fileData = file.readAll();
    fileAttach.fileName = QFileInfo(file).fileName();
    QString idTxt = QFileInfo(file).absolutePath();
    idTxt = idTxt.split("/").last();
    fileAttach.fileTypeSpec = "application/vnd.tcpdump.pcap";
    list.append(fileAttach);
    connect(&php, SIGNAL(uploadProgress(qint64,qint64)), SLOT(onUploadProgress(qint64,qint64)));
    if(php.postReply(FORUM_SHARING_FORUM, FORUM_SHARING_THREAD, "",  "Crash report for " + idTxt + "\n" + ui->te_Description->toPlainText(), list)) {
        QMessageBox::information(this, tr("Crash report sent"), tr("Thank you!"));
        ui->stackedWidget->setCurrentIndex(0);
        this->close();
    } else {
        QMessageBox::warning(this, tr("Could not send the crash report"), tr("Ooops, something went wrong"));
        ui->stackedWidget->setCurrentIndex(0);
    }
}

void MainWindow::onCancelSend()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void MainWindow::onUploadProgress(qint64 progress, qint64 total)
{
    if(total == 0)
        return;
    ui->uploadLabel->setText("Uploading");
    ui->progressLabel->setText(QString("Uploaded %0 of %1 bytes").arg(progress).arg(total));
    int p = (progress * 100) / total;
    ui->progressBar->setValue(p);
}
