/**
 ******************************************************************************
 *
 * @file       flightlogdownload.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @see        The GNU Public License (GPL) Version 3
 * @brief      Import/Export Plugin
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup   Logging
 * @{
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
#include "flightlogdownload.h"
#include "ui_flightlogdownload.h"

#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>

#include "loggingstats.h"

#include <QDateTime>
#include <QFile>
#include <QFileDialog>
#include <QDebug>

FlightLogDownload::FlightLogDownload(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FlightLogDownload)
{
    ui->setupUi(this);

    dl_state = DL_IDLE;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *uavoManager = pm->getObject<UAVObjectManager>();
    loggingStats = LoggingStats::GetInstance(uavoManager);
    Q_ASSERT(loggingStats);

    connect(ui->fileNameButton, SIGNAL(clicked()), this, SLOT(getFilename()));
    connect(ui->saveButton, SIGNAL(clicked()), this, SLOT(startDownload()));

    // Create default file name
    QString fileName = tr("TauLabs-%0.tll").arg(QDateTime::currentDateTime().toString("yyyy-MM-dd_hh-mm-ss"));
    ui->fileName->setText(QDir::current().relativeFilePath(fileName));

    // Get the current status
    connect(loggingStats, SIGNAL(objectUnpacked(UAVObject*)), this, SLOT(updateReceived()));
    loggingStats->requestUpdate();
}

FlightLogDownload::~FlightLogDownload()
{
    delete ui;
}

//! Present file selector dialog for downloading file
void FlightLogDownload::getFilename()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save log as..."),
                                       tr("TauLabs-%0.tll").arg(QDateTime::currentDateTime().toString("yyyy-MM-dd_hh-mm-ss")),
                                       tr("Log (*.tll)"));
    if (!fileName.isEmpty())
        ui->fileName->setText(fileName);
}

/**
 * @brief FlightLogDownload::updateReceived respond to updates
 * from the LoggingStats object and store the data during a
 * download
 */
void FlightLogDownload::updateReceived()
{
    LoggingStats::DataFields logging = loggingStats->getData();

    switch(dl_state) {
    case DL_IDLE:
        // Update the file selector
        ui->cbFileId->clear();
        for (int i = logging.MinFileId; i <= logging.MaxFileId; i++)
            ui->cbFileId->addItem(QString::number(i), QVariant(i));
        return;
    case DL_COMPLETE:
        return;
    case DL_DOWNLOADING:
        break;
    }

    UAVObject::Metadata mdata;

    switch (logging.Operation) {
    case LoggingStats::OPERATION_IDLE:
        log.append((char *) logging.FileSector, LoggingStats::FILESECTOR_NUMELEM);
        logging.Operation = LoggingStats::OPERATION_DOWNLOAD;
        logging.FileSectorNum++;
        loggingStats->setData(logging);
        loggingStats->updated();
        ui->sectorLabel->setText(QString::number(logging.FileSectorNum));
        qDebug() << "Requesting sector num: " << logging.FileSectorNum;
        break;
    case LoggingStats::OPERATION_COMPLETE:
    {
        log.append((char *) logging.FileSector, LoggingStats::FILESECTOR_NUMELEM);

        dl_state = DL_IDLE;

        mdata = loggingStats->getMetadata();
        UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_MANUAL);
        loggingStats->setMetadata(mdata);

        logFile->write(log);
        logFile->close();

        break;
    }
    case LoggingStats::OPERATION_ERROR:
        dl_state = DL_IDLE;

        mdata = loggingStats->getMetadata();
        UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_MANUAL);
        loggingStats->setMetadata(mdata);

        break;
    default:
        qDebug() << "Unhandled";
    }
}

/**
 * @brief FlightLogDownload::startDownload set up the metadata
 * on the logging object and start a download after checking the
 * file name is valid.
 */
void FlightLogDownload::startDownload()
{
    bool ok;
    qint32 file_id = ui->cbFileId->currentData().toInt(&ok);
    if (!ok)
        return;

    logFile = new QFile(ui->fileName->text(), this);
    if (!logFile->open(QIODevice::WriteOnly))
        return;

    log.clear();

    LoggingStats::DataFields logging = loggingStats->getData();

    // Stop any existing log file
    dl_state = DL_IDLE;
    logging.Operation = LoggingStats::OPERATION_IDLE;
    loggingStats->setData(logging);
    loggingStats->updated();

    UAVObject::Metadata mdata = loggingStats->getMetadata();
    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_ONCHANGE);
    loggingStats->setMetadata(mdata);

    qDebug() << "Download file id: " << file_id;
    dl_state = DL_DOWNLOADING;
    logging.Operation = LoggingStats::OPERATION_DOWNLOAD;
    logging.FileRequest = file_id;
    logging.FileSectorNum = 0;
    loggingStats->setData(logging);
    loggingStats->updated();
}

/**
 * @}
 * @}
 */
