/**
 ******************************************************************************
 *
 * @file       logging.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
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

#include "loggingplugin.h"
#include "loggingdevice.h"
#include "logginggadgetfactory.h"
#include <QDebug>
#include <QtPlugin>
#include <QThread>
#include <QStringList>
#include <QDir>
#include <QFileDialog>
#include <QList>
#include <QErrorMessage>
#include <QWriteLocker>

#include <extensionsystem/pluginmanager.h>
#include <QKeySequence>
#include "uavobjectmanager.h"


LoggingConnection::LoggingConnection()
{
    // We only ever use one (virtual) logging device, so let's just
    // initialize it once upon connection init:
    logDevice.setDisplayName("Logfile replay...");
    logDevice.setName("Logfile replay...");


}

LoggingConnection::~LoggingConnection()
{
}

void LoggingConnection::onEnumerationChanged()
{
        emit availableDevChanged(this);
}

QList <Core::IDevice*> LoggingConnection::availableDevices()
{
    QList <Core::IDevice*> list;
    list.append(&logDevice);
    return list;
}

QIODevice* LoggingConnection::openDevice(IDevice *deviceName)
{
    Q_UNUSED(deviceName)

    if (logFile.isOpen()){
        logFile.close();
    }
    QString fileName = QFileDialog::getOpenFileName(NULL, tr("Open file"), QString(""), tr("Tau Labs Log (*.tll)"));
    if (!fileName.isNull()) {
        startReplay(fileName);
    }
    return &logFile;
}

void LoggingConnection::startReplay(QString file)
{
    logFile.setFileName(file);
    if(logFile.open(QIODevice::ReadOnly)) {
        qDebug() << "Replaying " << file;
        // state = REPLAY;
        logFile.startReplay();
    }
}

void LoggingConnection::closeDevice(const QString &deviceName)
{
    Q_UNUSED(deviceName);
    //we have to delete the serial connection we created
    if (logFile.isOpen()){
        logFile.close();
        m_deviceOpened = false;
    }
}


QString LoggingConnection::connectionName()
{
    return QString("Logfile replay");
}

QString LoggingConnection::shortName()
{
    return QString("Logfile");
}



/**
 * @brief LoggingThread::~LoggingThread Destructor
 */
LoggingThread::~LoggingThread()
{
    stopLogging();
}

/**
  * Sets the file to use for logging and takes the parent plugin
  * to connect to stop logging signal
  * @param[in] file File name to write to
  * @param[in] parent plugin
  */
bool LoggingThread::openFile(QString file, LoggingPlugin * parent)
{
    logFile.setFileName(file);
    logFile.open(QIODevice::WriteOnly);

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();

    uavTalk = new UAVTalk(&logFile, objManager);
    connect(parent,SIGNAL(stopLoggingSignal()),this,SLOT(stopLogging()));

    return true;
};

/**
  * Logs an object update to the file.  Data format is the
  * timestamp as a 32 bit uint counting ms from start of
  * file writing (flight time will be embedded in stream),
  * then object packet size, then the packed UAVObject.
  */
void LoggingThread::objectUpdated(UAVObject * obj)
{
    QWriteLocker locker(&lock);
    if(!uavTalk->sendObject(obj,false,false) )
        qDebug() << "Error logging " << obj->getName();
};

/**
  * Connect signals from all the objects updates to the write routine then
  * run event loop
  */
void LoggingThread::run()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();

    QVector< QVector<UAVObject*> > list = objManager->getObjectsVector();
    QVector< QVector<UAVObject*> >::const_iterator i;
    QVector<UAVObject*>::const_iterator j;
    int objects = 0;

    const QVector< QVector<UAVObject*> >::iterator iEnd = list.end();
    for (i = list.constBegin(); i != iEnd; ++i)
    {
        QVector<UAVObject*>::const_iterator jEnd = (*i).constEnd();
        for (j = (*i).constBegin(); j != jEnd; ++j)
        {
            connect(*j, SIGNAL(objectUpdated(UAVObject*)), (LoggingThread*) this, SLOT(objectUpdated(UAVObject*)));
            objects++;
        }
    }

    GCSTelemetryStats* gcsStatsObj = GCSTelemetryStats::GetInstance(objManager);
    GCSTelemetryStats::DataFields gcsStats = gcsStatsObj->getData();
    if ( gcsStats.Status == GCSTelemetryStats::STATUS_CONNECTED )
    {
        qDebug() << "Logging: connected already, ask for all settings";
        retrieveSettings();
    } else {
        qDebug() << "Logging: not connected, do no ask for settings";
    }


    exec();
}


/**
  * Pass this command to the correct thread then close the file
  */
void LoggingThread::stopLogging()
{
    QWriteLocker locker(&lock);

    // Disconnect all objects we registered with:
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();

    QVector< QVector<UAVObject*> > list = objManager->getObjectsVector();
    QVector< QVector<UAVObject*> >::const_iterator i;
    QVector<UAVObject*>::const_iterator j;

    const QVector< QVector<UAVObject*> >::iterator iEnd = list.end();
    for (i = list.constBegin(); i != iEnd; ++i)
    {
        QVector<UAVObject*>::const_iterator jEnd = (*i).constEnd();
        for (j = (*i).constBegin(); j != jEnd; ++j)
        {
            disconnect(*j, SIGNAL(objectUpdated(UAVObject*)), (LoggingThread*) this, SLOT(objectUpdated(UAVObject*)));
        }
    }

    logFile.close();
    qDebug() << "File closed";
    quit();
}

/**
 * Initialize queue with settings objects to be retrieved.
 */
void LoggingThread::retrieveSettings()
{
    // Clear object queue
    queue.clear();
    // Get all objects, add metaobjects, settings and data objects with OnChange update mode to the queue
    // Get UAVObjectManager instance
    ExtensionSystem::PluginManager* pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objMngr = pm->getObject<UAVObjectManager>();
    QVector< QVector<UAVDataObject*> > objs = objMngr->getDataObjectsVector();
    for (int n = 0; n < objs.size(); ++n)
    {
        UAVDataObject* obj = objs[n][0];
        if ( obj->isSettings() )
                {
                    queue.enqueue(obj);
                }
        }
    // Start retrieving
    qDebug() << tr("Logging: retrieve settings objects from the autopilot (%1 objects)")
                  .arg( queue.length());
    retrieveNextObject();
}


/**
 * Retrieve the next object in the queue
 */
void LoggingThread::retrieveNextObject()
{
    // If queue is empty return
    if ( queue.isEmpty() )
    {
        qDebug() << "Logging: Object retrieval completed";
        return;
    }
    // Get next object from the queue
    UAVObject* obj = queue.dequeue();
    // Connect to object
    connect(obj, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(transactionCompleted(UAVObject*,bool)));
    // Request update
    obj->requestUpdate();
}

/**
 * Called by the retrieved object when a transaction is completed.
 */
void LoggingThread::transactionCompleted(UAVObject* obj, bool success)
{
    Q_UNUSED(success);
    // Disconnect from sending object
    obj->disconnect(this);
    // Process next object if telemetry is still available
    // Get stats objects
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    GCSTelemetryStats* gcsStatsObj = GCSTelemetryStats::GetInstance(objManager);
    GCSTelemetryStats::DataFields gcsStats = gcsStatsObj->getData();
    if ( gcsStats.Status == GCSTelemetryStats::STATUS_CONNECTED )
    {
        retrieveNextObject();
    }
    else
    {
        qDebug() << "Logging: Object retrieval has been cancelled";
        queue.clear();
    }
}



/****************************************************************
    Logging plugin
 ********************************/


LoggingPlugin::LoggingPlugin() : state(IDLE)
{
    logConnection = new LoggingConnection();
}

LoggingPlugin::~LoggingPlugin()
{
    if (loggingThread)
        delete loggingThread;

    // Don't delete it, the plugin manager will do it:
    //delete logConnection;
}

/**
  * Add Logging plugin to File menu
  */
bool LoggingPlugin::initialize(const QStringList& args, QString *errMsg)
{
    Q_UNUSED(args);
    Q_UNUSED(errMsg);

    loggingThread = NULL;


    // Add Menu entry
    Core::ActionManager* am = Core::ICore::instance()->actionManager();
    Core::ActionContainer* ac = am->actionContainer(Core::Constants::M_TOOLS);

    // Command to start logging
    cmd = am->registerAction(new QAction(this),
                                            "LoggingPlugin.Logging",
                                            QList<int>() <<
                                            Core::Constants::C_GLOBAL_ID);
    cmd->setDefaultKeySequence(QKeySequence("Ctrl+L"));
    cmd->action()->setText("Start logging...");

    ac->menu()->addSeparator();
    ac->appendGroup("Logging");
    ac->addAction(cmd, "Logging");

    connect(cmd->action(), SIGNAL(triggered(bool)), this, SLOT(toggleLogging()));


    mf = new LoggingGadgetFactory(this);
    addAutoReleasedObject(mf);

    // Map signal from end of replay to replay stopped
    connect(getLogfile(),SIGNAL(replayFinished()), this, SLOT(replayStopped()));
    connect(getLogfile(),SIGNAL(replayStarted()), this, SLOT(replayStarted()));

    return true;
}

/**
  * The action that is triggered by the menu item which opens the
  * file and begins logging if successful
  */
void LoggingPlugin::toggleLogging()
{
    if(state == IDLE)
    {

        QString fileName = QFileDialog::getSaveFileName(NULL, tr("Start Log"),
                                    tr("TauLabs-%0.tll").arg(QDateTime::currentDateTime().toString("yyyy-MM-dd_hh-mm-ss")),
                                    tr("Tau Labs Log (*.tll)"));
        if (fileName.isEmpty())
            return;

        startLogging(fileName);
        cmd->action()->setText(tr("Stop logging"));

    }
    else if(state == LOGGING)
    {
        stopLogging();
        cmd->action()->setText(tr("Start logging..."));
    }
}


/**
  * Starts the logging thread to a certain file
  */
void LoggingPlugin::startLogging(QString file)
{
    qDebug() << "Logging to " << file;
    // We have to delete the previous logging thread if is was still there!
    if (loggingThread)
        delete loggingThread;
    loggingThread = new LoggingThread();
    if(loggingThread->openFile(file,this))
    {
        connect(loggingThread,SIGNAL(finished()),this,SLOT(loggingStopped()));
        state = LOGGING;
        loggingThread->start();
        emit stateChanged("LOGGING");
    } else {
        QErrorMessage err;
        err.showMessage("Unable to open file for logging");
        err.exec();
    }
}

/**
  * Send the stop logging signal to the LoggingThread
  */
void LoggingPlugin::stopLogging()
{
    emit stopLoggingSignal();
    disconnect( this,SIGNAL(stopLoggingSignal()),0,0);
}


/**
  * Receive the logging stopped signal from the LoggingThread
  * and change status to not logging
  */
void LoggingPlugin::loggingStopped()
{
    if(state == LOGGING)
        state = IDLE;

    emit stateChanged("IDLE");

    delete loggingThread;
    loggingThread = NULL;
}

/**
  * Received the replay stopped signal from the LogFile
  */
void LoggingPlugin::replayStopped()
{
    state = IDLE;
    emit stateChanged("IDLE");
}

/**
  * Received the replay started signal from the LogFile
  */
void LoggingPlugin::replayStarted()
{
    state = REPLAY;
    emit stateChanged("REPLAY");
}




void LoggingPlugin::extensionsInitialized()
{
    addAutoReleasedObject(logConnection);
}

void LoggingPlugin::shutdown()
{
    // Do nothing
}

/**
 * @}
 * @}
 */
