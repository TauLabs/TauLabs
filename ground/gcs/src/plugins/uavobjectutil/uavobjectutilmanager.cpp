/**
 ******************************************************************************
 *
 * @file       uavobjectutilmanager.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @see        The GNU Public License (GPL) Version 3
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVObjectUtilPlugin UAVObjectUtil Plugin
 * @{
 * @brief      The UAVUObjectUtil GCS plugin
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

#include "uavobjectutilmanager.h"
#include "physical_constants.h"
#include "utils/homelocationutil.h"

#include <QMutexLocker>
#include <QDebug>
#include <QEventLoop>
#include <QTimer>
#include <objectpersistence.h>
#include <QInputDialog>

#include "firmwareiapobj.h"
#include "homelocation.h"
#include "gpsposition.h"

// ******************************
// constructor/destructor

UAVObjectUtilManager::UAVObjectUtilManager()
{
    mutex = new QMutex(QMutex::Recursive);
    saveState = IDLE;
    failureTimer.stop();
    failureTimer.setSingleShot(true);
    failureTimer.setInterval(1000);
    connect(&failureTimer, SIGNAL(timeout()),this,SLOT(objectPersistenceOperationFailed()));

    pm = NULL;
    obm = NULL;
    obum = NULL;

    pm = ExtensionSystem::PluginManager::instance();
    if (pm)
    {
        obm = pm->getObject<UAVObjectManager>();
        obum = pm->getObject<UAVObjectUtilManager>();
    }

}

UAVObjectUtilManager::~UAVObjectUtilManager()
{
//	while (!queue.isEmpty())
	{
	}

	disconnect();

	if (mutex)
	{
		delete mutex;
		mutex = NULL;
	}
}


UAVObjectManager* UAVObjectUtilManager::getObjectManager() {
    Q_ASSERT(obm);
    return obm;
}


// ******************************
// Flash memory saving
//

/**
 * @brief UAVObjectUtilManager::saveObjectToSD Add a new object to save in the queue
 * @param obj
 *
 * This method only tells the remote end to save the object to its persistent storage (flash
 * usually), but does not send the object over the telemetry link beforehand. It is therefore
 * up to the programmer to make sure the object is in the right state on the remote end before
 * calling this method - the "SmartSave" feature in the ConfigTaskWidget does this, for instance.
 *
 * We need to manage a save queue, because once a save request is sent, we need to wait until we
 * get an update from the remote end telling us whether the operation was successful. If we don't
 * manage the queue, then we will end up sending save requests one after another and won't be able
 * to monitor success or failure at all.
 *
 * The objectPersistence UAVObject works as follows:
 *  - Set the ObjecID and InstanceID for the target object
 *  - Set the operation ('save' for instance)
 *  - Then update the object, which sends it over the telemetry link
 *  - Once the objectPersistence UAVO is updated, the board will in turn update it again,
 *    once the operation is completed. We need therefore to listen to updates on the objectPersistence
 *    object, and check the "Operation" field, which should be set to "completed", or "error".
 */
void UAVObjectUtilManager::saveObjectToFlash(UAVObject *obj)
{
    // Add to queue
    queue.enqueue(obj);
    qDebug() << "Enqueue object: " << obj->getName();

    // If queue length is one, then start sending (call sendNextObject)
    // Otherwise, do nothing, because we are already sending.
    if (queue.length()==1)
        saveNextObject();
}


/**
 * @brief UAVObjectUtilManager::saveNextObject
 *
 * Processes the save queue.
 */
void UAVObjectUtilManager::saveNextObject()
{
    if ( queue.isEmpty() ) {
        return;
    }

    Q_ASSERT(saveState == IDLE);

    // Get next object from the queue (don't dequeue yet)
    UAVObject* obj = queue.head();
    Q_ASSERT(obj);
    qDebug() << "Send save object request to board " << obj->getName();

    ObjectPersistence * objectPersistence = ObjectPersistence::GetInstance(getObjectManager());
    Q_ASSERT(objectPersistence);

    // "transactionCompleted" is emitted once the objectPersistence object is sent over the telemetry link.
    // Since its metadata state that it should be ACK'ed on flight telemetry, the transactionCompleted signal
    // will be triggered once the GCS telemetry manager receives an ACK from the flight controller, or times
    // out. the success value will reflect success or failure.
    connect(objectPersistence, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(objectPersistenceTransactionCompleted(UAVObject*,bool)));
    // After we update the objectPersistence UAVO, we need to listen to its "objectUpdated" event, which will occur
    // once the flight controller sends this object back to the GCS, with the "Operation" field set to "Completed"
    // or "Error".
    connect(objectPersistence, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(objectPersistenceUpdated(UAVObject *)));
    saveState = AWAITING_ACK;
    qDebug() << "[saveObjectToFlash] Moving on to AWAITING_ACK";

    ObjectPersistence::DataFields data;
    data.Operation = ObjectPersistence::OPERATION_SAVE;
    data.Selection = ObjectPersistence::SELECTION_SINGLEOBJECT;
    data.ObjectID = obj->getObjID();
    data.InstanceID = obj->getInstID();
    objectPersistence->setData(data);
    objectPersistence->updated();
    // Now: we are going to get the following:
    // - two "objectUpdated" messages (one coming from GCS, one coming from Flight, which will confirm the object
    //   was properly received by both sides). This is because the metadata of objectPersistence has "FlightUpdateOnChange"
    //   set to true.
    //  - then one "transactionCompleted" indicating that the Flight side did not only receive the object but it did
    //    receive it without error (that message will be triggered by reception of the ACK).
    //  - Last we will get one last "objectUpdated" message coming from flight side, where we'll get the results of
    //    the objectPersistence operation we asked for (saved, other).
}


/**
  * @brief Process the transactionCompleted message from Telemetry indicating request sent successfully
  * @param[in] The object just transacted.  Must be ObjectPersistance
  * @param[in] success Indicates that the transaction did not time out
  *
  * After a failed transaction (usually timeout) simply drops the save request and emits a failure
  * signal (saveCompleted with 'false').  After a succesful
  * transaction it will then wait for a "save completed" update from the autopilot.
  */
void UAVObjectUtilManager::objectPersistenceTransactionCompleted(UAVObject* obj, bool success)
{
    if(success) {
        Q_ASSERT(obj->getName().compare("ObjectPersistence") == 0);
        Q_ASSERT(saveState == AWAITING_ACK);
        // Two things can happen then:
        // Either the Object Save Request did actually go through, and then we should get in
        // "AWAITING_COMPLETED" mode, or the Object Save Request did _not_ go through, for example
        // because the object does not exist and then we will never get a subsequent update - the flight firmware
        // does not update the objectPersistence UAVO with "Error", it just ignores the request.
        // For this reason, we will arm a 1 second timer to make provision for this and not block
        // the queue:
        saveState = AWAITING_COMPLETED;
        qDebug() << "[saveObjectToFlash] Moving on to AWAITING_COMPLETED";
        disconnect(obj, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(objectPersistenceTransactionCompleted(UAVObject*,bool)));
        failureTimer.start(2000); // Create a timeout
    } else {
        // Can be caused by timeout errors on sending.  Forget it and send next.
        qDebug() << "objectPersistenceTranscationCompleted (error)";
        ObjectPersistence * objectPersistence = ObjectPersistence::GetInstance(getObjectManager());
        Q_ASSERT(objectPersistence);

        objectPersistence->disconnect(this);
        queue.dequeue(); // We can now remove the object, it failed anyway.
        saveState = IDLE;
        // Careful below: objectPersistence contains a field 'ObjectID' that corresponds to
        // the objectID we wanted to save. Don't confuse with the OBJID of the Objectpersistence
        // UAVO itself!
        emit saveCompleted(objectPersistence->getObjectID(), false);
        saveNextObject();
    }
}


/**
  * @brief Object persistence operation failed, i.e. we never got an update
  * from the board saying "completed". This method is only called when we get a timeout.
  */
void UAVObjectUtilManager::objectPersistenceOperationFailed()
{
    if(saveState == AWAITING_COMPLETED) {

        ObjectPersistence * objectPersistence = ObjectPersistence::GetInstance(getObjectManager());
        Q_ASSERT(objectPersistence);

        UAVObject* obj = queue.dequeue(); // We can now remove the object, it failed anyway.
        Q_ASSERT(obj);

        objectPersistence->disconnect(this);

        saveState = IDLE;
        emit saveCompleted(obj->getObjID(), false);

        saveNextObject();
    }

}


/**
  * @brief Process the ObjectPersistence updated message to confirm the right object saved
  * then requests next object be saved.
  * @param[in] The object just received.  Must be ObjectPersistance
  */
void UAVObjectUtilManager::objectPersistenceUpdated(UAVObject * obj)
{
    Q_ASSERT(obj);
    Q_ASSERT(obj->getObjID() == ObjectPersistence::OBJID);

    if (saveState != AWAITING_COMPLETED) {
        // We'll get two of those before we're in AWAITING_COMPLETED state...
        return;
    }

    ObjectPersistence::DataFields objectPersistence = ((ObjectPersistence *)obj)->getData();

    if (objectPersistence.Operation == ObjectPersistence::OPERATION_ERROR) {
        failureTimer.stop();
        objectPersistenceOperationFailed();
    } else if (objectPersistence.Operation == ObjectPersistence::OPERATION_COMPLETED) {
        failureTimer.stop();
        // Check right object saved
        UAVObject* savingObj = queue.head();
        if(objectPersistence.ObjectID != savingObj->getObjID() ) {
            objectPersistenceOperationFailed();
            return;
        }

        obj->disconnect(this);
        queue.dequeue(); // We can now remove the object, it's done.
        saveState = IDLE;
        qDebug() << "[saveObjectToFlash] Object save succeeded";
        emit saveCompleted(objectPersistence.ObjectID, true);
        saveNextObject();
    }
}


/**
 * @brief UAVObjectUtilManager::readAllNonSettingsMetadata Convenience function for calling
 * readMetadata
 * @return QMap containing list of all requested UAVO metaadata
 */
QMap<QString, UAVObject::Metadata> UAVObjectUtilManager::readAllNonSettingsMetadata()
{
    return readMetadata(NONSETTINGS_METADATA_ONLY);
}


/**
 * @brief UAVObjectUtilManager::readMetadata Get metadata for UAVOs
 * @param metadataReadType Defines if all UAVOs are read, only settings, or only data types
 * @return QMap containing list of all requested UAVO metaadata
 */
QMap<QString, UAVObject::Metadata> UAVObjectUtilManager::readMetadata(metadataSetEnum metadataReadType)
{
    QMap<QString, UAVObject::Metadata> metaDataList;

    // Save all metadata objects.
    UAVObjectManager *objManager = getObjectManager();
    QVector< QVector<UAVDataObject*> > objList = objManager->getDataObjectsVector();
    foreach (QVector<UAVDataObject*> list, objList) {
        foreach (UAVDataObject* obj, list) {
            bool updateMetadataFlag = false;
            switch (metadataReadType){
            case ALL_METADATA:
                updateMetadataFlag = true;
                break;
            case SETTINGS_METADATA_ONLY:
                if(obj->isSettings()) {
                    updateMetadataFlag = true;
                }
                break;
            case NONSETTINGS_METADATA_ONLY:
                if(!obj->isSettings()) {
                    updateMetadataFlag = true;
                }
                break;
            }

            if (updateMetadataFlag){
                metaDataList.insert(obj->getName(), obj->getMetadata());
            }
        }
    }

    return metaDataList;
}


/**
 * @brief UAVObjectUtilManager::setAllNonSettingsMetadata Convenience function for calling setMetadata
 * @param metaDataList QMap containing list of all UAVO metaadata to be updated
 * @return
 */
bool UAVObjectUtilManager::setAllNonSettingsMetadata(QMap<QString, UAVObject::Metadata> metaDataList)
{
    metadataSetEnum metadataSetType = NONSETTINGS_METADATA_ONLY;
    setMetadata(metaDataList, metadataSetType);

    return true;
}


/**
 * @brief UAVObjectUtilManager::setMetadata Sets the metadata for all metadata in QMap
 * @param metaDataSetList QMap containing list of all UAVO metaadata to be updated
 * @param metadataSetType Controls if all metadata is updated, only settings metadata,
 * or only dynamic data metadata
 * @return
 */
bool UAVObjectUtilManager::setMetadata(QMap<QString, UAVObject::Metadata> metaDataSetList, metadataSetEnum metadataSetType)
{
    metadataChecklist = metaDataSetList;

    // Load all metadata objects.
    UAVObjectManager *objManager = getObjectManager();
    QVector< QVector<UAVDataObject*> > objList = objManager->getDataObjectsVector();
    foreach (QVector<UAVDataObject*> list, objList) {
        foreach (UAVDataObject* obj, list) {
            bool updateMetadataFlag = false;
            switch (metadataSetType){
            case ALL_METADATA:
                updateMetadataFlag = true;
                break;
            case SETTINGS_METADATA_ONLY:
                if(obj->isSettings()) {
                    updateMetadataFlag = true;
                }
                break;
            case NONSETTINGS_METADATA_ONLY:
                if(!obj->isSettings()) {
                    updateMetadataFlag = true;
                }
                break;
            }

            if (metaDataSetList.contains(obj->getName()) && updateMetadataFlag){
                obj->setMetadata(metaDataSetList.value(obj->getName()));

                // Connect to object
                connect(obj, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(metadataTransactionCompleted(UAVObject*,bool)));
                // Request update
                obj->requestUpdate();

            }
        }
    }

    return true;
}


/**
 * @brief UAVObjectUtilManager::resetMetadata Resets all metadata to defaults (from XML definitions)
 * @return
 */
bool UAVObjectUtilManager::resetMetadataToDefaults()
{
    QMap<QString, UAVObject::Metadata> metaDataList;

    // Load all metadata object defaults
    UAVObjectManager *objManager = getObjectManager();
    QVector< QVector<UAVDataObject*> > objList = objManager->getDataObjectsVector();
    foreach (QVector<UAVDataObject*> list, objList) {
        foreach (UAVDataObject* obj, list) {
            metaDataList.insert(obj->getName(), obj->getDefaultMetadata());
        }
    }

    // Save metadata
    metadataSetEnum metadataSetType = ALL_METADATA;
    bool ret = setMetadata(metaDataList, metadataSetType);

    return ret;
}


/**
 * @brief ConfigTaskWidget::metadataTransactionCompleted Called by the retrieved object when a transaction is completed.
 * @param uavoObject
 * @param success
 */
void UAVObjectUtilManager::metadataTransactionCompleted(UAVObject* uavoObject, bool success)
{
    Q_UNUSED(success);
    // Disconnect from sending UAVO
    disconnect(uavoObject, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(metadataTransactionCompleted(UAVObject*,bool)));


    // If the UAVO is on the list, check that the data was set correctly
    if(metadataChecklist.contains(uavoObject->getName()))
    {
        UAVObject::Metadata mdata = metadataChecklist.value(uavoObject->getName());
        if( uavoObject->getMetadata().flags == mdata.flags &&
                uavoObject->getMetadata().flightTelemetryUpdatePeriod == mdata.flightTelemetryUpdatePeriod &&
                uavoObject->getMetadata().gcsTelemetryUpdatePeriod == mdata.gcsTelemetryUpdatePeriod &&
                uavoObject->getMetadata().loggingUpdatePeriod == mdata.loggingUpdatePeriod)
        {
            metadataChecklist.take(uavoObject->getName());
        }
        else{
            qDebug() << "Writing metadata failed on " << uavoObject->getName();
            Q_ASSERT(0);
        }
    }

   if(metadataChecklist.empty()){
       //We're done, that was the last item checked off the list.
       emit completedMetadataWrite();
   }

   // TODO: Implement a timeout
}


/**
  * Helper function that makes sure FirmwareIAP is updated and then returns the data
  */
FirmwareIAPObj::DataFields UAVObjectUtilManager::getFirmwareIap()
{
    FirmwareIAPObj::DataFields dummy;

    FirmwareIAPObj *firmwareIap = FirmwareIAPObj::GetInstance(obm);
    Q_ASSERT(firmwareIap);
    if (!firmwareIap)
        return dummy;

    return firmwareIap->getData();
}

/**
  * Get the UAV Board model, for anyone interested. Return format is:
  * (Board Type << 8) + BoardRevision.
  *
  *  NOTE: Should get deprecated as a public method now, in favour
  *  of getBoardType and subsequent board capability queries.
  */
int UAVObjectUtilManager::getBoardModel()
{
    FirmwareIAPObj::DataFields firmwareIapData = getFirmwareIap();
    int ret=firmwareIapData.BoardType <<8;
    ret = ret + firmwareIapData.BoardRevision;
    return ret;
}

//! Get the IBoardType corresponding to the connected board
Core::IBoardType *UAVObjectUtilManager::getBoardType()
{
    int boardTypeNum = (getBoardModel() >> 8) & 0x00ff;
    QList <Core::IBoardType *> boards = pm->getObjects<Core::IBoardType>();
    foreach (Core::IBoardType *board, boards) {
        if (board->getBoardType() == boardTypeNum)
            return board;
    }
    return NULL;
}

/**
  * Get the UAV Board CPU Serial Number, for anyone interested. Return format is a byte array
  */
QByteArray UAVObjectUtilManager::getBoardCPUSerial()
{
    QByteArray cpuSerial;
    FirmwareIAPObj::DataFields firmwareIapData = getFirmwareIap();

    for (unsigned int i = 0; i < FirmwareIAPObj::CPUSERIAL_NUMELEM; i++)
        cpuSerial.append(firmwareIapData.CPUSerial[i]);

    return cpuSerial;
}

quint32 UAVObjectUtilManager::getFirmwareCRC()
{
    FirmwareIAPObj::DataFields firmwareIapData = getFirmwareIap();
    return firmwareIapData.crc;
}

/**
  * Get the UAV Board Description, for anyone interested.
  */
QByteArray UAVObjectUtilManager::getBoardDescription()
{
    QByteArray ret;
    FirmwareIAPObj::DataFields firmwareIapData = getFirmwareIap();

    for (unsigned int i = 0; i < FirmwareIAPObj::DESCRIPTION_NUMELEM; i++)
        ret.append(firmwareIapData.Description[i]);

    return ret;
}



// ******************************
// HomeLocation

int UAVObjectUtilManager::setHomeLocation(double LLA[3], bool save_to_sdcard)
{
    double Be[3];

    int retval = Utils::HomeLocationUtil().getDetails(LLA, Be);
    Q_ASSERT(retval >= 0);

    if (retval < 0)
        return -1;

    // ******************
    // save the new settings

    HomeLocation *homeLocation = HomeLocation::GetInstance(obm);
    Q_ASSERT(homeLocation != NULL);

    HomeLocation::DataFields homeLocationData = homeLocation->getData();
    homeLocationData.Latitude = LLA[0] * 1e7;
    homeLocationData.Longitude = LLA[1] * 1e7;
    homeLocationData.Altitude = LLA[2];

    homeLocationData.Be[0] = Be[0];
    homeLocationData.Be[1] = Be[1];
    homeLocationData.Be[2] = Be[2];

    homeLocationData.Set = HomeLocation::SET_TRUE;

    homeLocation->setData(homeLocationData);

    if (save_to_sdcard)
        saveObjectToFlash(homeLocation);

    return 0;
}

int UAVObjectUtilManager::getHomeLocation(bool &set, double LLA[3])
{
    HomeLocation *homeLocation = HomeLocation::GetInstance(obm);
    Q_ASSERT(homeLocation != NULL);

    HomeLocation::DataFields homeLocationData = homeLocation->getData();

    set = homeLocationData.Set;

    LLA[0] = homeLocationData.Latitude*1e-7;
    LLA[1] = homeLocationData.Longitude*1e-7;
    LLA[2] = homeLocationData.Altitude;

	if (LLA[0] != LLA[0]) LLA[0] = 0; // nan detection
	else
	if (LLA[0] >  90) LLA[0] =  90;
	else
	if (LLA[0] < -90) LLA[0] = -90;

	if (LLA[1] != LLA[1]) LLA[1] = 0; // nan detection
	else
	if (LLA[1] >  180) LLA[1] =  180;
	else
	if (LLA[1] < -180) LLA[1] = -180;

	if (LLA[2] != LLA[2]) LLA[2] = 0; // nan detection

	return 0;	// OK
}


// ******************************
// GPS

int UAVObjectUtilManager::getGPSPosition(double LLA[3])
{
    GPSPosition *gpsPosition = GPSPosition::GetInstance(obm);
    Q_ASSERT(gpsPosition != NULL);

    GPSPosition::DataFields gpsPositionData = gpsPosition->getData();

    LLA[0] = gpsPositionData.Latitude;
    LLA[1] = gpsPositionData.Longitude;
    LLA[2] = gpsPositionData.Altitude;

	if (LLA[0] != LLA[0]) LLA[0] = 0; // nan detection
	else
	if (LLA[0] >  90) LLA[0] =  90;
	else
	if (LLA[0] < -90) LLA[0] = -90;

	if (LLA[1] != LLA[1]) LLA[1] = 0; // nan detection
	else
	if (LLA[1] >  180) LLA[1] =  180;
	else
	if (LLA[1] < -180) LLA[1] = -180;

	if (LLA[2] != LLA[2]) LLA[2] = 0; // nan detection

	return 0;	// OK
}

deviceDescriptorStruct UAVObjectUtilManager::getBoardDescriptionStruct()
{
    deviceDescriptorStruct ret;
    descriptionToStructure(getBoardDescription(),ret);
    return ret;
}

bool UAVObjectUtilManager::descriptionToStructure(QByteArray desc, deviceDescriptorStruct & struc)
{
   if (desc.startsWith("TlFw") || desc.startsWith("OpFw")) {
       /*
        * This looks like a binary with a description at the end
        *  4 bytes: header: "TlFw"
        *  4 bytes: git commit hash (short version of SHA1)
        *  4 bytes: Unix timestamp of last git commit
        *  2 bytes: target platform. Should follow same rule as BOARD_TYPE and BOARD_REVISION in board define files.
        *  26 bytes: commit tag if it is there, otherwise "Unreleased". Zero-padded
        *  20 bytes: SHA1 sum of the firmware.
        *  20 bytes: SHA1 sum of the UAVO definition files.
        *  20 bytes: free for now.
        */

       // Note: the ARM binary is big-endian:
       quint32 gitCommitHash = desc.at(7) & 0xFF;
       for (int i = 1; i < 4; i++) {
           gitCommitHash = gitCommitHash << 8;
           gitCommitHash += desc.at(7-i) & 0xFF;
       }
       struc.gitHash = QString("%1").arg(gitCommitHash, 8, 16, QChar('0'));

       quint32 gitDate = desc.at(11) & 0xFF;
       for (int i = 1; i < 4; i++) {
           gitDate = gitDate << 8;
           gitDate += desc.at(11-i) & 0xFF;
       }
       struc.gitDate = QDateTime::fromTime_t(gitDate).toUTC().toString("yyyyMMdd HH:mm");

       QString gitTag = QString(desc.mid(14,26));
       struc.gitTag = gitTag;

       // TODO: check platform compatibility
       QByteArray targetPlatform = desc.mid(12,2);
       struc.boardType = (int)targetPlatform.at(0);
       struc.boardRevision = (int)targetPlatform.at(1);
       struc.fwHash.clear();
       struc.fwHash=desc.mid(40,20);
       struc.uavoHash.clear();
       struc.uavoHash=desc.mid(60,20);

       return true;
   }
   return false;
}

// ******************************
