/**
 ******************************************************************************
 *
 * @file       uavobjectmanager.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @see        The GNU Public License (GPL) Version 3
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVObjectsPlugin UAVObjects Plugin
 * @{
 * @brief      The UAVUObjects GCS plugin
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
#include "uavobjectmanager.h"

/**
 * Constructor
 */
UAVObjectManager::UAVObjectManager()
{
    mutex = new QMutex(QMutex::Recursive);
}

UAVObjectManager::~UAVObjectManager()
{
    delete mutex;
}

/**
 * Register an object with the manager. This function must be called for all newly created instances.
 * A new instance can be created directly by instantiating a new object or by calling clone() of
 * an existing object. The object will be registered and will be properly initialized so that it can accept
 * updates.
 */
bool UAVObjectManager::registerObject(UAVDataObject* obj)
{
    QMutexLocker locker(mutex);
    // Check if this object type is already in the list
    quint32 objID = obj->getObjID();
    if (objects.contains(objID))//Known object ID
    {
        if (objects.value(objID).contains(obj->getInstID()))//Instance already present
            return false;
        if (obj->isSingleInstance())
            return false;
        if (obj->getInstID() >= MAX_INSTANCES)
            return false;
        UAVDataObject* refObj = dynamic_cast<UAVDataObject*>(objects.value(objID).first());
        if (refObj == NULL)
        {
            return false;
        }
        UAVMetaObject* mobj = refObj->getMetaObject();
        if (objects.value(objID).last()->getInstID() < obj->getInstID())//Space between last existent instance and new one, lets fill the gaps
        {
            for (quint32 instidx = objects.value(objID).last()->getInstID() + 1 ; instidx < obj->getInstID(); ++instidx)
            {
                UAVDataObject* cobj = obj->clone(instidx);
                cobj->initialize(instidx,mobj);
                QMap<quint32,UAVObject*> ppp;
                ppp.insert(instidx,cobj);
                objects[objID].insert(instidx,cobj);
                getObject(cobj->getObjID())->emitNewInstance(cobj);//TODO??
                emit newInstance(cobj);
            }
        }
        else if (obj->getInstID() == 0)
            obj->initialize(objects.value(objID).last()->getObjID() + 1, mobj);
        else
        {
            return false;
        }
        // Add the actual object instance in the list
        objects[objID].insert(obj->getInstID(),obj);
        getObject(objID)->emitNewInstance(obj);
        emit newInstance(obj);
        return true;
    }
    else
    {
        // If this point is reached then this is the first time this object type (ID) is added in the list
        // create a new list of the instances, add in the object collection and create the object's metaobject
        // Create metaobject
        QString mname = obj->getName();
        mname.append("Meta");
        UAVMetaObject* mobj = new UAVMetaObject(objID + 1, mname, obj);
        // Initialize object
        obj->initialize(0, mobj);
        // Add to list
        addObject(obj);
        addObject(mobj);
        return true;
    }
 }

/**
 * @brief unregisters an object instance and all instances bigger than the one passed as argument from the manager
 * @param obj pointer to the object to unregister
 * @return false if object is single instance
 */
bool UAVObjectManager::unRegisterObject(UAVDataObject* obj)
{
    QMutexLocker locker(mutex);
    // Check if this object type is already in the list
    quint32 objID = obj->getObjID();
    if(obj->isSingleInstance())
        return false;
    int instances = (quint32)objects.value(obj->getObjID()).count();
    for(quint32 x = obj->getInstID(); x < instances; ++x)
    {
        getObject(objects.value(objID).value(x)->getObjID())->emitInstanceRemoved(objects.value(objID).value(x));
        emit instanceRemoved(objects.value(objID).value(x));
        objects[objID].remove(x);
    }
    return true;
}

void UAVObjectManager::addObject(UAVObject* obj)
{
    // Add to list
    QMap<quint32,UAVObject*> list;
    list.insert(obj->getInstID(),obj);
    objects.insert(obj->getObjID(),list);
    emit newObject(obj);
}

/**
 * Get all objects. A two dimentional QVector is returned. Objects are grouped by
 * instances of the same object type.
 */
QVector< QVector<UAVObject*> > UAVObjectManager::getObjectsVector()
{
    QMutexLocker locker(mutex);
    QVector< QVector<UAVObject*> > vector;
    foreach(ObjectMap map,objects.values())
    {
        QVector<UAVObject*> vec = map.values().toVector();
        vector.append(vec);
    }
    return vector;
}

QHash<quint32, QMap<quint32, UAVObject *> > UAVObjectManager::getObjects()
{
    return objects;
}

/**
 * Same as getObjects() but will only return DataObjects.
 */
QVector< QVector<UAVDataObject*> > UAVObjectManager::getDataObjectsVector()
{
    QMutexLocker locker(mutex);
    QVector< QVector<UAVDataObject*> > vector;
    foreach(ObjectMap map,objects.values())
    {
        UAVDataObject* obj = dynamic_cast<UAVDataObject*>(map.first());
        if(obj!=NULL)
        {
            QVector<UAVDataObject*> vec;
            foreach(UAVObject* o,map)
            {
                UAVDataObject* dobj = dynamic_cast<UAVDataObject*>(o);
                if(dobj)
                    vec.append(dobj);
            }
            vector.append(vec);
        }
     }
    return vector;
}

/**
 * Same as getObjects() but will only return MetaObjects.
 */
QVector <QVector<UAVMetaObject*> > UAVObjectManager::getMetaObjectsVector()
{
    QMutexLocker locker(mutex);
    QVector< QVector<UAVMetaObject*> > vector;
    foreach(ObjectMap map,objects.values())
    {
        UAVMetaObject* obj = dynamic_cast<UAVMetaObject*>(map.first());
        if(obj!=NULL)
        {
            QVector<UAVMetaObject*> vec;
            foreach(UAVObject* o,map)
            {
                UAVMetaObject* mobj = dynamic_cast<UAVMetaObject*>(o);
                if(mobj)
                    vec.append(mobj);
            }
            vector.append(vec);
        }
     }
    return vector;
}

/**
 * Get a specific object given its name and instance ID
 * @returns The object is found or NULL if not
 */
UAVObject* UAVObjectManager::getObject(const QString& name, quint32 instId)
{
    return getObject(&name, 0, instId);
}

/**
 * Get a specific object given its object and instance ID
 * @returns The object is found or NULL if not
 */
UAVObject* UAVObjectManager::getObject(quint32 objId, quint32 instId)
{
    return getObject(NULL, objId, instId);
}

/**
 * Helper function for the public getObject() functions.
 */
UAVObject* UAVObjectManager::getObject(const QString* name, quint32 objId, quint32 instId)
{
    QMutexLocker locker(mutex);
    if(name != NULL)
    {
        foreach(ObjectMap map,objects)
        {
            if(map.first()->getName().compare(name) == 0)
                return map.value(instId,NULL);
        }
    }
    else if(objects.contains(objId))
        return objects.value(objId).value(instId);
    return NULL;
}

/**
 * Get all the instances of the object specified by name
 */
QVector<UAVObject*> UAVObjectManager::getObjectInstancesVector(const QString& name)
{
    return getObjectInstancesVector(&name, 0);
}

/**
 * Get all the instances of the object specified by its ID
 */
QVector<UAVObject*> UAVObjectManager::getObjectInstancesVector(quint32 objId)
{
    return getObjectInstancesVector(NULL, objId);
}

/**
 * Helper function for the public getObjectInstances()
 */
QVector<UAVObject*> UAVObjectManager::getObjectInstancesVector(const QString* name, quint32 objId)
{
    QMutexLocker locker(mutex);
    if(name != NULL)
    {
        foreach(ObjectMap map,objects)
        {
            if(map.first()->getName().compare(name) == 0)
                return map.values().toVector();
        }
    }
    else if(objects.contains(objId))
        return objects.value(objId).values().toVector();
    return  QVector<UAVObject*>();
}

/**
 * Get the number of instances for an object given its name
 */
qint32 UAVObjectManager::getNumInstances(const QString& name)
{
    return getNumInstances(&name, 0);
}

/**
 * Get the number of instances for an object given its ID
 */
qint32 UAVObjectManager::getNumInstances(quint32 objId)
{
    return getNumInstances(NULL, objId);
}

/**
 * Helper function for public getNumInstances
 */
qint32 UAVObjectManager::getNumInstances(const QString* name, quint32 objId)
{
    QMutexLocker locker(mutex);
    if(name != NULL)
    {
        foreach(ObjectMap map,objects)
        {
            if(map.first()->getName().compare(name) == 0)
                return map.count();
        }
    }
    else if(objects.contains(objId))
        return objects.value(objId).count();
    return -1;
}
