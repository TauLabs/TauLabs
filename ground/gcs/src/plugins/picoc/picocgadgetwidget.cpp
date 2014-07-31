/**
 ******************************************************************************
 * @file       picocgadgetwidget.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup PicoCGadgetPlugin PicoC Gadget Plugin
 * @{
 * @brief A gadget to edit PicoC scripts
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
#include "picocgadgetwidget.h"
#include "extensionsystem/pluginmanager.h"

#include <QString>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include <QFile>
#include <QFileDialog>
#include <QMessageBox>
#include <QEventLoop>
#include <QTimer>
#include <QDebug>

PicoCGadgetWidget::PicoCGadgetWidget(QWidget *parent) : QLabel(parent)
{
    ui = new Ui_PicoCWidget();
    ui->setupUi(this);

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm != NULL);
    objManager = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objManager != NULL);
    pcStatus = PicoCStatus::GetInstance(objManager);
    Q_ASSERT(pcStatus != NULL);
    pcSettings = PicoCSettings::GetInstance(objManager);
    Q_ASSERT(pcSettings != NULL);

    // Connect PicoCStatus
    connect(pcStatus, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(updatePicoCStatus(UAVObject *)));

}

PicoCGadgetWidget::~PicoCGadgetWidget()
{
   // Do nothing
}

/**
 * @brief PicoCGadgetWidget::on_tbNewFile_clicked Clear script in editor.
 */
void PicoCGadgetWidget::on_tbNewFile_clicked()
{
    ui->ptEditor->clear();
}

/**
 * @brief PicoCGadgetWidget::on_tbLoadFromFile_clicked Display file dialog and load
 * script from file to editor.
 */
void PicoCGadgetWidget::on_tbLoadFromFile_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Load Script File"), "", tr("C Source (*.c)"));

    if (fileName.isEmpty()) {
        return;
    }

    QFile file(fileName);
    if (file.open(QFile::ReadOnly|QFile::Text)) {
        ui->ptEditor->setPlainText(file.readAll());
    } else {
        QMessageBox::critical(0,
                              tr("PicoC Save"),
                              tr("Unable to load script: ") + fileName,
                              QMessageBox::Ok);
        return;
    }
}

/**
 * @brief PicoCGadgetWidget::on_tbSaveToFile_clicked Display file dialog and save
 * script from editor to file.
 */
void PicoCGadgetWidget::on_tbSaveToFile_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save Script File As"), "", tr("C Source (*.c)"));

    if (fileName.isEmpty()) {
        return;
    }

    QFile file(fileName);
    if (file.open(QFile::WriteOnly) && (file.write(ui->ptEditor->toPlainText().toLatin1())) != -1) {
        file.close();
    } else {
        QMessageBox::critical(0,
                              tr("PicoC Save"),
                              tr("Unable to save script: ") + fileName,
                              QMessageBox::Ok);
        return;
    }
}

/**
 * @brief PicoCGadgetWidget::on_tbFetchFromUAV_clicked Transfer script from UAV to editor.
 */
void PicoCGadgetWidget::on_tbFetchFromUAV_clicked()
{
    // disable the button
    ui->tbFetchFromUAV->setEnabled(false);

    // fetch script from uav
    fetchScript();

    // enable the button again
    ui->tbFetchFromUAV->setEnabled(true);
}

/**
 * @brief PicoCGadgetWidget::on_tbSendToUAV_clicked Transfer script from editor to UAV.
 */
void PicoCGadgetWidget::on_tbSendToUAV_clicked()
{
    // disable the button
    ui->tbSendToUAV->setEnabled(false);

    // send content of editor to uav
    sendScript();

    // enable the button again
    ui->tbSendToUAV->setEnabled(true);
}

/**
 * @brief PicoCGadgetWidget::on_tbFetchFromUAVROM_clicked Load script from UAV filesystem
 * and transfer it to editor.
 */
void PicoCGadgetWidget::on_tbFetchFromUAVROM_clicked()
{
    // disable the button
    ui->tbFetchFromUAVROM->setEnabled(false);

    // set FileID and execute load command
    pcStatus->setFileID(ui->sbFileID->value());
    pcStatus->setCommand(PicoCStatus::COMMAND_LOADFILE);
    pcStatus->updated();

    // wait for execution of the command
    waitForExecution();

    // check command status after execution
    if ((pcStatus->getCommand() == PicoCStatus::COMMAND_IDLE) && (pcStatus->getCommandError() == 0)) {
        fetchScript();
    } else {
        QMessageBox::critical(0,
                              tr("PicoC UAV ROM"),
                              tr("Fetch from UAV ROM failed."),
                              QMessageBox::Ok);
    }

    // enable the button again
    ui->tbFetchFromUAVROM->setEnabled(true);
}

/**
 * @brief PicoCGadgetWidget::on_tbSendToUAVROM_clicked Transfer script from editor to UAV
 * and save it on UAV filesystem
 */
void PicoCGadgetWidget::on_tbSendToUAVROM_clicked()
{
    // disable the button
    ui->tbSendToUAVROM->setEnabled(false);

    // send content of editor to model
    sendScript();

    // set FileID and execute save command
    pcStatus->setFileID(ui->sbFileID->value());
    pcStatus->setCommand(PicoCStatus::COMMAND_SAVEFILE);
    pcStatus->updated();

    // wait for execution of the command
    waitForExecution();

    // check command status after execution
    if ((pcStatus->getCommand() == PicoCStatus::COMMAND_IDLE) && (pcStatus->getCommandError() == 0)) {
        QMessageBox::information(0,
                              tr("PicoC UAV ROM"),
                              tr("Save to UAV ROM finished."),
                              QMessageBox::Ok);
    } else {
        QMessageBox::critical(0,
                              tr("PicoC UAV ROM"),
                              tr("Save to UAV ROM failed."),
                              QMessageBox::Ok);
    }

    // enable the button again
    ui->tbSendToUAVROM->setEnabled(true);
}

/**
 * @brief PicoCGadgetWidget::on_tbEraseFromUAVROM_clicked Erase script from UAV filesystem
 */
void PicoCGadgetWidget::on_tbEraseFromUAVROM_clicked()
{
    // disable the button
    ui->tbEraseFromUAVROM->setEnabled(false);

    // set FileID and execute load command
    pcStatus->setFileID(ui->sbFileID->value());
    pcStatus->setCommand(PicoCStatus::COMMAND_DELETEFILE);
    pcStatus->updated();

    // wait for execution of the command
    waitForExecution();

    // enable the button again
    ui->tbEraseFromUAVROM->setEnabled(true);
}

/**
 * @brief PicoCGadgetWidget::on_tbStartScript_clicked Execute actual script
 */
void PicoCGadgetWidget::on_tbStartScript_clicked()
{
    // disable the button
    ui->tbStartScript->setEnabled(false);

    // execute script via command
    pcStatus->setTestValue(ui->sbTestValue->value());
    pcStatus->setCommand(PicoCStatus::COMMAND_STARTSCRIPT);
    pcStatus->updated();
}

/**
 * @brief PicoCGadgetWidget::updatePicoCStatus Called when PicoCStatus changes
 * @param obj
 */
void PicoCGadgetWidget::updatePicoCStatus(UAVObject *obj)
{
    Q_UNUSED(obj);

    // update exit and test values
    ui->sbExitValue->setValue(pcStatus->getExitValue());
    ui->sbTestValue->setValue(pcStatus->getTestValue());

    // enable some buttons in idle state
    if (pcStatus->getCommand() == PicoCStatus::COMMAND_IDLE) {
        ui->tbStartScript->setEnabled(true);
    }
}

/**
 * @brief PicoCGadgetWidget::setProgress Show value in progress bar.
 */
void PicoCGadgetWidget::setProgress(int value)
{
    ui->pbPercent->setValue(value);
}

/**
 * @brief PicoCGadgetWidget::fetchScript Transfer script from UAV to
 * editor. It uses sector fragmentation and transfer commands.
 */
void PicoCGadgetWidget::fetchScript()
{
    QString text = "";
    int max = pcSettings->getMaxFileSize();
    int i = 0;
    char c = 0;
    do {
        setProgress(i * 100 / max);
        if (i % PicoCStatus::SECTOR_NUMELEM == 0) {
            pcStatus->setSectorID(i / PicoCStatus::SECTOR_NUMELEM);
            pcStatus->setCommand(PicoCStatus::COMMAND_GETSECTOR);
            pcStatus->updated();

            if (!waitForExecution()) {
                setProgress(0);
                return;
            }
        }
        c = pcStatus->getSector(i % PicoCStatus::SECTOR_NUMELEM);
        text.append(c);
        i++;
    } while ((c != 0) && (pcStatus->getCommand() == PicoCStatus::COMMAND_IDLE));

    setProgress(0);
    ui->ptEditor->setPlainText(text.toLatin1());
}

/**
 * @brief PicoCGadgetWidget::sendScript Transfer script from editor to
 * UAV. It uses sector fragmentation and transfer commands.
 */
void PicoCGadgetWidget::sendScript()
{
    // security check for AccessLevelSet
    if ((ui->ptEditor->toPlainText().contains("AccessLevelSet")) &&
        (QMessageBox::question(0,
                              tr("PicoC UAV"),
                              tr("It seems, that you have used AccessLevels in your script. "
                                 "This enables extended access to UAVOs and can lead to problems. "
                                 "A programming error can cause loss of control over your vehicle. "
                                 "Are you sure you want to upload the script?"),
                              QMessageBox::Yes|QMessageBox::No) == QMessageBox::No))
        return;

    int max = 1 + ui->ptEditor->toPlainText().length();
    int i = 0;
    char c = 0;
    do {
        setProgress(i * 100 / max);
        if (i < (ui->ptEditor->toPlainText().length())) {
            c = ui->ptEditor->toPlainText().at(i).toLatin1();
        } else {
            c = 0;
        }
        pcStatus->setSector(i % PicoCStatus::SECTOR_NUMELEM, c);
        pcStatus->setSectorID(i / PicoCStatus::SECTOR_NUMELEM);
        i++;

        // send sector to uav when filled
        if ((i % PicoCStatus::SECTOR_NUMELEM) == 0) {
            pcStatus->setCommand(PicoCStatus::COMMAND_SETSECTOR);
            pcStatus->updated();

            if (!waitForExecution()) {
                setProgress(0);
                return;
            }
        }
    } while ((c != 0) || (i % PicoCStatus::SECTOR_NUMELEM != 0));

    setProgress(0);
}

/**
 * @brief PicoCGadgetWidget::executeCmd Wait for execution acknowledge.
 * Timeout in 10 seconds.
 * @return True if succeed, false otherwise.
 */
bool PicoCGadgetWidget::waitForExecution()
{
    for (int i = 0; i < 100; i++) {
        if (pcStatus->getCommand() == PicoCStatus::COMMAND_IDLE)
            return true;
        sleep(100);
    }
    return false;
}

/**
 * @brief PicoCGadgetWidget::sleep Let the task sleep for a while
 * to give UAVOs time for update.
 * @param msec The delay time.
 */
void PicoCGadgetWidget::sleep(int msec)
{
    QEventLoop m_eventloop;
    QTimer::singleShot(msec, &m_eventloop, SLOT(quit()));
    m_eventloop.exec();
}

/**
  * @}
  * @}
  */
