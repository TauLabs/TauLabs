/**
 ******************************************************************************
 * @file       filtereduavtalk.c
 * @author     The PhoenixPilot Team, http://github.com/PhoenixPilot
 * * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVTalk relay plugin
 * @{
 *
 * @brief Relays UAVTalk data trough UDP to another GCS
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
#include "filtereduavtalk.h"
#include "gcstelemetrystats.h"

//! Construct a filtered uavtalk class
FilteredUavTalk::FilteredUavTalk(QIODevice *iodev, UAVObjectManager *objMngr,
                                 QHash<quint32,UavTalkRelayComon::accessType> rules,
                                 UavTalkRelayComon::accessType defaultRule) :
    UAVTalk(iodev,objMngr),m_rules(rules),m_defaultRule(defaultRule)
{
}

/**
 * @brief FilteredUavTalk::sendObjectSlot Called whenever an object is updated by
 * the local GCS and sends it to the remote GCS is the rules allow it.
 * @param obj The UAVObject to possibly send
 */
void FilteredUavTalk::sendObjectSlot(UAVObject *obj)
{
    UavTalkRelayComon::accessType access=m_rules.value(obj->getObjID(),m_defaultRule);
    if(access==UavTalkRelayComon::WriteOnly || access==UavTalkRelayComon::None)
        return;
    if(obj==GCSTelemetryStats::GetInstance(objMngr))
        return;
    sendObject(obj,false,false);
}

/**
 * @brief FilteredUavTalk::receiveObject Called by the parent parser whenever an object is deserialized
 * with a correct CRC.  Determines what to do based on the applicable rules.
 * @param type The type of UAVTalk message sent (TYPE_OBJ, TYPE_OBJ_ACK, TYPE_OBJ_REQ)
 * @param objId The ID of the object received
 * @param instId The instance ID of the received object
 * @param data The array of data received
 * @param length The length of the data received
 * @return True if the object passed the flitering, false otherwise
 */
bool FilteredUavTalk::receiveObject(quint8 type, quint32 objId, quint16 instId, quint8 *data, qint32 length)
{
    Q_UNUSED(length);
    UAVObject* obj = NULL;
    bool error = false;
    bool allInstances =  (instId == ALL_INSTANCES);
    UavTalkRelayComon::accessType access=m_rules.value(objId,m_defaultRule);
    if(access==UavTalkRelayComon::ReadOnly || access==UavTalkRelayComon::None)
        return false;
    if (obj == NULL)
        error = true;
    // Process message type
    switch (type) {
    case TYPE_OBJ: // We have received an object.
        // All instances, not allowed for OBJ messages
        if (!allInstances)
        {
            // Get object and update its data
            UAVObject* tobj = objMngr->getObject(objId);
            bool wasCon=disconnect(tobj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(sendObjectSlot(UAVObject*)));
            obj = updateObject(objId, instId, data);
            if(wasCon)
                connect(tobj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(sendObjectSlot(UAVObject*)));
            UAVMetaObject * mobj=dynamic_cast<UAVMetaObject*>(tobj);
            if(mobj)
                tobj->updated();
            if (obj == NULL)
                error = true;
        }
        else
        {
            error = true;
        }
        break;
    case TYPE_OBJ_ACK: // We have received an object and are asked for an ACK
        // All instances, not allowed for OBJ_ACK messages
        if (!allInstances)
        {
            // Get object and update its data
            UAVObject* tobj = objMngr->getObject(objId);
            bool wasCon=disconnect(tobj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(sendObjectSlot(UAVObject*)));
            obj = updateObject(objId, instId, data);
            if(wasCon)
                connect(tobj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(sendObjectSlot(UAVObject*)));
            UAVMetaObject * mobj=dynamic_cast<UAVMetaObject*>(tobj);
            if(mobj)
                tobj->updated();
            // Transmit ACK
            if ( obj != NULL )
            {
                transmitObject(obj, TYPE_ACK, false);
            }
            else
            {
                error = true;
            }
        }
        else
        {
            error = true;
        }
        break;
    case TYPE_OBJ_REQ:  // We are being asked for an object
        // Get object, if all instances are requested get instance 0 of the object
        if(access==UavTalkRelayComon::WriteOnly || access==UavTalkRelayComon::None)
            break;
        if (allInstances)
        {
            obj = objMngr->getObject(objId);
        }
        else
        {
            obj = objMngr->getObject(objId, instId);
        }
        // If object was found transmit it
        if (obj != NULL)
        {
            transmitObject(obj, TYPE_OBJ, allInstances);
        }
        else
        {
            // Object was not found, transmit a NACK with the
            // objId which was not found.
            transmitNack(objId);
            error = true;
        }
        break;
    case TYPE_NACK: // We have received a NACK for an object that does not exist on the far end.
                    // (but should exist on our end)
        // All instances, not allowed for NACK messages
        if (!allInstances)
        {
            // Get object
            obj = objMngr->getObject(objId, instId);
            // Check if object exists:
            if (obj != NULL)
            {
                emit nackReceived(obj);
            }
            else
            {
             error = true;
            }
        }
        break;
    case TYPE_ACK: // We have received a ACK, supposedly after sending an object with OBJ_ACK
        // All instances, not allowed for ACK messages
        if (!allInstances)
        {
            // Get object
            obj = objMngr->getObject(objId, instId);
            // Check if we actually know this object (tiny chance the ObjID
            // could be unknown and got through CRC check...)
            if (obj != NULL)
            {
                emit ackReceived(obj);
            }
            else
            {
                error = true;
            }
        }
        break;
    default:
        error = true;
    }
    // Done (exit value is "success", hence the "!" below)
    return !error;
}
