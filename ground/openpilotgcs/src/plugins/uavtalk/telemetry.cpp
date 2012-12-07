/**
 ******************************************************************************
 * @file       telemetry.cpp
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVTalkPlugin UAVTalk Plugin
 * @{
 * @brief Provide a higher level of telemetry control on top of UAVTalk
 * including setting up transactions and acknowledging their receipt or
 * timeout
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

#include "telemetry.h"
#include "qxtlogger.h"
#include "pipxsettings.h"
#include "objectpersistence.h"
#include <QTime>
#include <QtGlobal>
#include <stdlib.h>
#include <QDebug>

/**
 * @brief The TransactionKey class A key for the QMap to track transactions
 */
class TransactionKey {
public:
    TransactionKey(quint32 objId, quint32 instId, bool req) {
        this->objId = objId;
        this->instId = instId;
        this->req = req;
    }

    TransactionKey(UAVObject *obj, bool req) {
        this->objId = obj->getObjID();
        this->instId = obj->getInstID();
        this->req = req;
    }

    // See if this is an equivalent transaction key
    bool operator==(const TransactionKey & rhs) const {
        return (rhs.objId == objId && rhs.instId == instId && rhs.req == req);
    }

    bool operator<(const TransactionKey & rhs) const {
        return objId < rhs.objId || (objId == rhs.objId && instId < rhs.instId) ||
                (objId == rhs.objId && instId == rhs.instId && req != rhs.req);
    }

    quint32 objId;
    quint32 instId;
    bool req;
};

/**
 * Constructor
 */
Telemetry::Telemetry(UAVTalk* utalk, UAVObjectManager* objMngr)
{
    this->utalk = utalk;
    this->objMngr = objMngr;
    mutex = new QMutex(QMutex::Recursive);
    // Process all objects in the list
    QList< QList<UAVObject*> > objs = objMngr->getObjects();
    for (int objidx = 0; objidx < objs.length(); ++objidx)
    {
        registerObject(objs[objidx][0]); // we only need to register one instance per object type
    }
    // Listen to new object creations
    connect(objMngr, SIGNAL(newObject(UAVObject*)), this, SLOT(newObject(UAVObject*)));
    connect(objMngr, SIGNAL(newInstance(UAVObject*)), this, SLOT(newInstance(UAVObject*)));
    // Listen to transaction completions
    connect(utalk, SIGNAL(ackReceived(UAVObject*)), this, SLOT(transactionSuccess(UAVObject*)));
    connect(utalk, SIGNAL(nackReceived(UAVObject*)), this, SLOT(transactionFailure(UAVObject*)));
    // Get GCS stats object
    gcsStatsObj = GCSTelemetryStats::GetInstance(objMngr);
    // Setup and start the periodic timer
    timeToNextUpdateMs = 0;
    updateTimer = new QTimer(this);
    connect(updateTimer, SIGNAL(timeout()), this, SLOT(processPeriodicUpdates()));
    updateTimer->start(1000);
    // Setup and start the stats timer
    txErrors = 0;
    txRetries = 0;
}

Telemetry::~Telemetry()
{
    for (QMap<TransactionKey, ObjectTransactionInfo*>::iterator itr = transMap.begin(); itr != transMap.end(); ++itr) {
        delete itr.value();
    }
}

/**
 * Register a new object for periodic updates (if enabled)
 */
void Telemetry::registerObject(UAVObject* obj)
{
    // Setup object for periodic updates
    addObject(obj);

    // Setup object for telemetry updates
    updateObject(obj, EV_NONE);
}

/**
 * Add an object in the list used for periodic updates
 */
void Telemetry::addObject(UAVObject* obj)
{
    // Check if object type is already in the list
    for (int n = 0; n < objList.length(); ++n)
    {
        if ( objList[n].obj->getObjID() == obj->getObjID() )
        {
            // Object type (not instance!) is already in the list, do nothing
            return;
        }
    }

    // If this point is reached, then the object type is new, let's add it
    ObjectTimeInfo timeInfo;
    timeInfo.obj = obj;
    timeInfo.timeToNextUpdateMs = 0;
    timeInfo.updatePeriodMs = 0;
    objList.append(timeInfo);
}

/**
 * Update the object's timers
 */
void Telemetry::setUpdatePeriod(UAVObject* obj, qint32 periodMs)
{
    // Find object type (not instance!) and update its period
    for (int n = 0; n < objList.length(); ++n)
    {
        if ( objList[n].obj->getObjID() == obj->getObjID() )
        {
            objList[n].updatePeriodMs = periodMs;
            objList[n].timeToNextUpdateMs = quint32((float)periodMs * (float)qrand() / (float)RAND_MAX); // avoid bunching of updates
        }
    }
}

/**
 * Connect to all instances of an object depending on the event mask specified
 */
void Telemetry::connectToObjectInstances(UAVObject* obj, quint32 eventMask)
{
    QList<UAVObject*> objs = objMngr->getObjectInstances(obj->getObjID());
    for (int n = 0; n < objs.length(); ++n)
    {
        // Disconnect all
        objs[n]->disconnect(this);
        // Connect only the selected events
        if ( (eventMask&EV_UNPACKED) != 0)
        {
            connect(objs[n], SIGNAL(objectUnpacked(UAVObject*)), this, SLOT(objectUnpacked(UAVObject*)));
        }
        if ( (eventMask&EV_UPDATED) != 0)
        {
            connect(objs[n], SIGNAL(objectUpdatedAuto(UAVObject*)), this, SLOT(objectUpdatedAuto(UAVObject*)));
        }
        if ( (eventMask&EV_UPDATED_MANUAL) != 0)
        {
            connect(objs[n], SIGNAL(objectUpdatedManual(UAVObject*)), this, SLOT(objectUpdatedManual(UAVObject*)));
        }
        if ( (eventMask&EV_UPDATED_PERIODIC) != 0)
        {
            connect(objs[n], SIGNAL(objectUpdatedPeriodic(UAVObject*)), this, SLOT(objectUpdatedPeriodic(UAVObject*)));
        }
        if ( (eventMask&EV_UPDATE_REQ) != 0)
        {
            connect(objs[n], SIGNAL(updateRequested(UAVObject*)), this, SLOT(updateRequested(UAVObject*)));
        }
    }
}

/**
 * Update an object based on its metadata properties.
 *
 * This method updates the connections between object events and the telemetry
 * layer, depending on the object's metadata properties.
 *
 * Note (elafargue, 2012.11): we listen for "unpacked" events in every case, because we want
 * to track when we receive object updates after doing an object request.
 */
void Telemetry::updateObject(UAVObject* obj, quint32 eventType)
{
    // Get metadata
    UAVObject::Metadata metadata = obj->getMetadata();
    UAVObject::UpdateMode updateMode = UAVObject::GetGcsTelemetryUpdateMode(metadata);

    // Setup object depending on update mode
    qint32 eventMask;
    if ( updateMode == UAVObject::UPDATEMODE_PERIODIC )
    {
        // Set update period
        setUpdatePeriod(obj, metadata.gcsTelemetryUpdatePeriod);
        // Connect signals for all instances
        eventMask = EV_UPDATED_MANUAL | EV_UPDATE_REQ | EV_UPDATED_PERIODIC;
        eventMask |= EV_UNPACKED; // we also need to act on remote updates (unpack events)
        connectToObjectInstances(obj, eventMask);
    }
    else if ( updateMode == UAVObject::UPDATEMODE_ONCHANGE )
    {
        // Set update period
        setUpdatePeriod(obj, 0);
        // Connect signals for all instances
        eventMask = EV_UPDATED | EV_UPDATED_MANUAL | EV_UPDATE_REQ;
        eventMask |= EV_UNPACKED; // we also need to act on remote updates (unpack events)
        connectToObjectInstances(obj, eventMask);
    }
    else if ( updateMode == UAVObject::UPDATEMODE_THROTTLED )
    {
        // If we received a periodic update, we can change back to update on change
        if ((eventType == EV_UPDATED_PERIODIC) || (eventType == EV_NONE)) {
            // Set update period
            if (eventType == EV_NONE)
                 setUpdatePeriod(obj, metadata.gcsTelemetryUpdatePeriod);
            // Connect signals for all instances
            eventMask = EV_UPDATED | EV_UPDATED_MANUAL | EV_UPDATE_REQ | EV_UPDATED_PERIODIC;
        }
        else
        {
            // Otherwise, we just received an object update, so switch to periodic for the timeout period to prevent more updates
            // Connect signals for all instances
            eventMask = EV_UPDATED | EV_UPDATED_MANUAL | EV_UPDATE_REQ;
        }
        eventMask |= EV_UNPACKED; // we also need to act on remote updates (unpack events)
        connectToObjectInstances(obj, eventMask);
    }
    else if ( updateMode == UAVObject::UPDATEMODE_MANUAL )
    {
        // Set update period
        setUpdatePeriod(obj, 0);
        // Connect signals for all instances
        eventMask = EV_UPDATED_MANUAL | EV_UPDATE_REQ;
        eventMask |= EV_UNPACKED; // we also need to act on remote updates (unpack events)
        connectToObjectInstances(obj, eventMask);
    }
}

/**
 * Called when a transaction is completed with success(uavtalk event).
 * This happens either:
 *  - Because we received an ACK from the UAVTalk layer.
 *  - Because we received an UNPACK event from an object we had requested.
 */
void Telemetry::transactionSuccess(UAVObject* obj)
{
    if (updateTransactionMap(obj,false)) {
        qDebug() << "[uavtalk.cpp] Transaction succeeded " << obj->getObjID() << " " << obj->getInstID();
        obj->emitTransactionCompleted(true);
    } else {
        qDebug() << "[telemetry.cpp] Received a ACK we were not expecting";
    }
    // Process new object updates from queue
    processObjectQueue();
}


/**
 * Called when a transaction is completed with failure (uavtalk event).
 * This happens either:
 *  - Because we received a NACK from the UAVTalk layer.
 *  - Because we did not receive an UNPACK event from an object we had requested and
 *    the object request retries is exceeded.
 */
void Telemetry::transactionFailure(UAVObject* obj)
{
    // Here we need to check for true or false as a NAK can occur for OBJ_REQ or an
    // object set
    if (updateTransactionMap(obj, true) || updateTransactionMap(obj, false)) {
        qDebug() << "[uavtalk.cpp] Transaction failed " << obj->getObjID() << " " << obj->getInstID();
        obj->emitTransactionCompleted(false);
    } else {
        qDebug() << "[telemetry.cpp] Received a NACK we were not expecting";
    }
    // Process new object updates from queue
    processObjectQueue();
}

/**
 * Called when an object request transaction is completed
 * This happens either:
 *  - Because we received an UNPACK event from an object we had requested.
 */
void Telemetry::transactionRequestCompleted(UAVObject* obj)
{
    if (updateTransactionMap(obj,true)) {
        qDebug() << "[uavtalk.cpp] Transaction succeeded " << obj->getObjID() << " " << obj->getInstID();
        obj->emitTransactionCompleted(true);
    } else {
        qDebug() << "[telemetry.cpp] Received a ACK we were not expecting";
    }
    // Process new object updates from queue
    processObjectQueue();
}

/**
 * @brief Telemetry::updateTransactionMap
 *  Check whether the object is in our pending transactions map. If so, remove
 *  it, otherwise return an error (false)
 * @param obj pointer to the UAV Object
 */
bool Telemetry::updateTransactionMap(UAVObject* obj, bool request)
{
    TransactionKey key(obj, request);
    QMap<TransactionKey, ObjectTransactionInfo*>::iterator itr = transMap.find(key);
    if ( itr != transMap.end() )
    {
        ObjectTransactionInfo *transInfo = itr.value();
        // Remove this transaction as it is complete.
        transInfo->timer->stop();
        transMap.remove(key);
        delete transInfo;
        return true;
    }
    return false;
}


/**
 * Called when a transaction is not completed within the timeout period (timer event)
 */
void Telemetry::transactionTimeout(ObjectTransactionInfo *transInfo)
{
    transInfo->timer->stop();
    // Check if more retries are pending
    if (transInfo->retriesRemaining > 0)
    {
        qDebug() << "[telemetry.cpp] Object transaction timeout for " << transInfo->obj->getName() << ", re-requesting.";
        --transInfo->retriesRemaining;
        processObjectTransaction(transInfo);
        ++txRetries;
    }
    else
    {
        transInfo->timer->stop();
        transactionFailure(transInfo->obj);
        ++txErrors;
    }
}

/**
 * Start an object transaction with UAVTalk, all information is stored in transInfo.
 */
void Telemetry::processObjectTransaction(ObjectTransactionInfo *transInfo)
{

    // Initiate transaction
    if (transInfo->objRequest)
    {  // We are requesting an object from the remote end
         utalk->sendObjectRequest(transInfo->obj, transInfo->allInstances);
    }
    else
    {   // We are sending an object to the remote end
        utalk->sendObject(transInfo->obj, transInfo->acked, transInfo->allInstances);
    }
    // Start timer if a response is expected
    if ( transInfo->objRequest || transInfo->acked )
    {
        transInfo->timer->start(REQ_TIMEOUT_MS);
    }
    else
    {
        // Stop tracking this transaction, since we're not expecting a response:
        transMap.remove(TransactionKey(transInfo->obj, transInfo->objRequest));
        delete transInfo;
    }
}

/**
 * Process the event received from an object we are following. This method
 * only enqueues objects for later processing
 */
void Telemetry::processObjectUpdates(UAVObject* obj, EventMask event, bool allInstances, bool priority)
{
    // Push event into queue
    ObjectQueueInfo objInfo;
    objInfo.obj = obj;
    objInfo.event = event;
    objInfo.allInstances = allInstances;
    if (priority)
    {
        if ( objPriorityQueue.length() < MAX_QUEUE_SIZE )
        {
            objPriorityQueue.enqueue(objInfo);
        }
        else
        {
            ++txErrors;
            obj->emitTransactionCompleted(false);
            qxtLog->warning(tr("Telemetry: priority event queue is full, event lost (%1)").arg(obj->getName()));
        }
    }
    else
    {
        if ( objQueue.length() < MAX_QUEUE_SIZE )
        {
            objQueue.enqueue(objInfo);
        }
        else
        {
            ++txErrors;
            obj->emitTransactionCompleted(false);
        }
    }
    // Process the transaction queue
    processObjectQueue();
}

/**
 * Process events from the object queue.
 */
void Telemetry::processObjectQueue()
{
    if (objQueue.length() > 1)
        qDebug() << "[telemetry.cpp] **************** Object Queue above 1 in backlog ****************";
    // Get object information from queue (first the priority and then the regular queue)
    ObjectQueueInfo objInfo;
    if ( !objPriorityQueue.isEmpty() )
    {
        objInfo = objPriorityQueue.dequeue();
    }
    else if ( !objQueue.isEmpty() )
    {
        objInfo = objQueue.dequeue();
    }
    else
    {
        return;
    }

    // Check if a connection has been established, only process GCSTelemetryStats updates
    // (used to establish the connection)
    GCSTelemetryStats::DataFields gcsStats = gcsStatsObj->getData();
    if ( gcsStats.Status != GCSTelemetryStats::STATUS_CONNECTED )
    {
        objQueue.clear();
        if ( objInfo.obj->getObjID() != GCSTelemetryStats::OBJID && objInfo.obj->getObjID() != PipXSettings::OBJID  && objInfo.obj->getObjID() != ObjectPersistence::OBJID )
        {
            objInfo.obj->emitTransactionCompleted(false);
            return;
        }
    }

    // Setup transaction (skip if unpack event)
    UAVObject::Metadata metadata = objInfo.obj->getMetadata();
    UAVObject::UpdateMode updateMode = UAVObject::GetGcsTelemetryUpdateMode(metadata);
    if ( ( objInfo.event != EV_UNPACKED ) && ( ( objInfo.event != EV_UPDATED_PERIODIC ) || ( updateMode != UAVObject::UPDATEMODE_THROTTLED ) ) )
    {
        // We are either going to send an object, or are requesting one:
        if (transMap.contains(TransactionKey(objInfo.obj, objInfo.event == EV_UPDATE_REQ))) {
            qDebug() << "[telemetry.cpp] Warning: Got request for " << objInfo.obj->getName() << " for which a request is already in progress. Not doing it";
            // We will not re-request it, then, we should wait for a timeout or success...
        } else
        {
            UAVObject::Metadata metadata = objInfo.obj->getMetadata();
            ObjectTransactionInfo *transInfo = new ObjectTransactionInfo(this);
            transInfo->obj = objInfo.obj;
            transInfo->allInstances = objInfo.allInstances;
            transInfo->retriesRemaining = MAX_RETRIES;
            transInfo->acked = UAVObject::GetGcsTelemetryAcked(metadata);
            if ( objInfo.event == EV_UPDATED || objInfo.event == EV_UPDATED_MANUAL || objInfo.event == EV_UPDATED_PERIODIC )
            {
                transInfo->objRequest = false;
            }
            else if ( objInfo.event == EV_UPDATE_REQ )
            {
                transInfo->objRequest = true;
            }
            transInfo->telem = this;
            // Insert the transaction into the transaction map.
            TransactionKey key(objInfo.obj, transInfo->objRequest);
            transMap.insert(key, transInfo);
            processObjectTransaction(transInfo);
        }
    }

    // If this is a metaobject then make necessary telemetry updates
    // to the connections of this object to Telemetry (this) :
    UAVMetaObject* metaobj = dynamic_cast<UAVMetaObject*>(objInfo.obj);
    if ( metaobj != NULL )
    {
        updateObject( metaobj->getParentObject(), EV_NONE );
    }
    else if ( updateMode != UAVObject::UPDATEMODE_THROTTLED )
    {
        updateObject( objInfo.obj, objInfo.event );
    }

    // We received an "unpacked" event, check whether
    // this is for an object we were expecting
    if ( objInfo.event == EV_UNPACKED ) {
        // TODO: Check here this is for a OBJ_REQ
        if (transMap.contains(TransactionKey(objInfo.obj, true))) {
            qDebug() << "[telemetry.cpp] EV_UNPACKED " << objInfo.obj->getName() << " inst " << objInfo.obj->getInstID();
            transactionRequestCompleted(objInfo.obj);
        } else
        {
            processObjectQueue();
        }
    }
}

/**
 * Check is any objects are pending for periodic updates
 * TODO: Clean-up
 */
void Telemetry::processPeriodicUpdates()
{
    QMutexLocker locker(mutex);

    // Stop timer
    updateTimer->stop();

    // Iterate through each object and update its timer, if zero then transmit object.
    // Also calculate smallest delay to next update (will be used for setting timeToNextUpdateMs)
    qint32 minDelay = MAX_UPDATE_PERIOD_MS;
    ObjectTimeInfo *objinfo;
    qint32 elapsedMs = 0;
    QTime time;
    qint32 offset;
    for (int n = 0; n < objList.length(); ++n)
    {
        objinfo = &objList[n];
        // If object is configured for periodic updates
        if (objinfo->updatePeriodMs > 0)
        {
            objinfo->timeToNextUpdateMs -= timeToNextUpdateMs;
            // Check if time for the next update
            if (objinfo->timeToNextUpdateMs <= 0)
            {
                // Reset timer
                offset = (-objinfo->timeToNextUpdateMs) % objinfo->updatePeriodMs;
                objinfo->timeToNextUpdateMs = objinfo->updatePeriodMs - offset;
                // Send object
                time.start();
                processObjectUpdates(objinfo->obj, EV_UPDATED_PERIODIC, true, false);
                elapsedMs = time.elapsed();
                // Update timeToNextUpdateMs with the elapsed delay of sending the object;
                timeToNextUpdateMs += elapsedMs;
            }
            // Update minimum delay
            if (objinfo->timeToNextUpdateMs < minDelay)
            {
                minDelay = objinfo->timeToNextUpdateMs;
            }
        }
    }

    // Check if delay for the next update is too short
    if (minDelay < MIN_UPDATE_PERIOD_MS)
    {
        minDelay = MIN_UPDATE_PERIOD_MS;
    }

    // Done
    timeToNextUpdateMs = minDelay;

    // Restart timer
    updateTimer->start(timeToNextUpdateMs);
}

Telemetry::TelemetryStats Telemetry::getStats()
{
    QMutexLocker locker(mutex);

    // Get UAVTalk stats
    UAVTalk::ComStats utalkStats = utalk->getStats();

    // Update stats
    TelemetryStats stats;
    stats.txBytes = utalkStats.txBytes;
    stats.rxBytes = utalkStats.rxBytes;
    stats.txObjectBytes = utalkStats.txObjectBytes;
    stats.rxObjectBytes = utalkStats.rxObjectBytes;
    stats.rxObjects = utalkStats.rxObjects;
    stats.txObjects = utalkStats.txObjects;
    stats.txErrors = utalkStats.txErrors + txErrors;
    stats.rxErrors = utalkStats.rxErrors;
    stats.txRetries = txRetries;

    // Done
    return stats;
}

void Telemetry::resetStats()
{
    QMutexLocker locker(mutex);
    utalk->resetStats();
    txErrors = 0;
    txRetries = 0;
}

void Telemetry::objectUpdatedAuto(UAVObject* obj)
{
    QMutexLocker locker(mutex);
    processObjectUpdates(obj, EV_UPDATED, false, true);
}

void Telemetry::objectUpdatedManual(UAVObject* obj)
{
    QMutexLocker locker(mutex);
    processObjectUpdates(obj, EV_UPDATED_MANUAL, false, true);
}

void Telemetry::objectUpdatedPeriodic(UAVObject* obj)
{
    QMutexLocker locker(mutex);
    processObjectUpdates(obj, EV_UPDATED_PERIODIC, false, true);
}

void Telemetry::objectUnpacked(UAVObject* obj)
{
    QMutexLocker locker(mutex);
    processObjectUpdates(obj, EV_UNPACKED, false, true);
}

void Telemetry::updateRequested(UAVObject* obj)
{
    QMutexLocker locker(mutex);
    processObjectUpdates(obj, EV_UPDATE_REQ, false, true);
}

void Telemetry::newObject(UAVObject* obj)
{
    QMutexLocker locker(mutex);
    registerObject(obj);
}

void Telemetry::newInstance(UAVObject* obj)
{
    QMutexLocker locker(mutex);
    registerObject(obj);
}

ObjectTransactionInfo::ObjectTransactionInfo(QObject* parent):QObject(parent)
{
    obj = 0;
    allInstances = false;
    objRequest = false;
    retriesRemaining = 0;
    acked = false;
    telem = 0;
    // Setup transaction timer
    timer = new QTimer(this);
    timer->stop();
    connect(timer, SIGNAL(timeout()), this, SLOT(timeout()));
}

ObjectTransactionInfo::~ObjectTransactionInfo()
{
    telem = 0;
    timer->stop();
    delete timer;
}

void ObjectTransactionInfo::timeout()
{
    if (!telem.isNull())
        telem->transactionTimeout(this);
}
