/**
 ******************************************************************************
 *
 * @file       logfile.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Plugin for generating a logfile
 *
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

#ifndef LOGFILE_H
#define LOGFILE_H

#include <QIODevice>
#include <QTime>
#include <QTimer>
#include <QMutexLocker>
#include <QDebug>
#include <QBuffer>
#include "uavobjectmanager.h"
#include <math.h>

class LogFile : public QIODevice
{
    Q_OBJECT
public:
    explicit LogFile(QObject *parent = 0);
    qint64 bytesAvailable() const;
    qint64 bytesToWrite() const { return file.bytesToWrite(); }
    bool open(OpenMode mode);
    void setFileName(QString name) { file.setFileName(name); }
    void close();
    qint64 writeData(const char * data, qint64 dataSize);
    qint64 readData(char * data, qint64 maxlen);

    bool startReplay();
    bool stopReplay();

public slots:
    void setReplaySpeed(double val) { playbackSpeed = val; qDebug() << "New playback speed: " << playbackSpeed; }
    void setReplayTime(double val);
    void pauseReplay();
    void resumeReplay();

protected slots:
    void timerFired();

signals:
    void readReady();
    void replayStarted();
    void replayFinished();

protected:
    QByteArray dataBuffer;
    QTimer timer;
    QTime myTime;
    QFile file;
    quint32 lastTimeStamp;
    quint32 lastPlayTime;
    QMutex mutex;


    int lastPlayTimeOffset;
    double playbackSpeed;

private:
    QList<quint32> timestampBuffer;
    QList<quint32> timestampPos;
    quint32 timestampBufferIdx;
    quint32 lastTimeStampPos;
    quint32 firstTimestamp;
};

#endif // LOGFILE_H
