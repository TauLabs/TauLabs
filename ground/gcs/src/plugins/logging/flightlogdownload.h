/**
 ******************************************************************************
 *
 * @file       flightlogdownload.h
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
#ifndef FLIGHTLOGDOWNLOAD_H
#define FLIGHTLOGDOWNLOAD_H

#include <QDialog>
#include <QByteArray>
#include <QFile>
#include "loggingstats.h"

namespace Ui {
class FlightLogDownload;
}

class FlightLogDownload : public QDialog
{
    Q_OBJECT
    
public:
    explicit FlightLogDownload(QWidget *parent = 0);
    ~FlightLogDownload();

private slots:
    void updateReceived();
    void startDownload();
    void getFilename();

private:
    LoggingStats *loggingStats;
    QByteArray log;
    QFile *logFile;

    enum LOG_DL_STATE {DL_IDLE, DL_DOWNLOADING, DL_COMPLETE} dl_state;

    Ui::FlightLogDownload *ui;
};

#endif // FLIGHTLOGDOWNLOAD_H

/**
 * @}
 * @}
 */
