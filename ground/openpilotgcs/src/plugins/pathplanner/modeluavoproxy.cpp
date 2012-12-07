/**
 ******************************************************************************
 * @file       modeluavproxy.cpp
 * @author     PhoenixPilot Project, http://github.com/PhoenixPilot Copyright (C) 2012.
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Path Planner Plugin
 * @{
 * @brief The Path Planner plugin
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

#include <QDebug>
#include <QEventLoop>
#include <QTimer>
#include "modeluavoproxy.h"
#include "extensionsystem/pluginmanager.h"
#include <math.h>

#include "utils/coordinateconversions.h"
#include "homelocation.h"

//! Initialize the model uavo proxy
ModelUavoProxy::ModelUavoProxy(QObject *parent, FlightDataModel *model):QObject(parent),myModel(model)
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm != NULL);
    objManager = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objManager != NULL);
    waypointObj = Waypoint::GetInstance(objManager);
    Q_ASSERT(waypointObj != NULL);
}

/**
 * @brief ModelUavoProxy::modelToObjects Cast from the internal representation of a path
 * to the UAV objects required to represent it
 */
void ModelUavoProxy::modelToObjects()
{
    Waypoint *wp = Waypoint::GetInstance(objManager,0);
    Q_ASSERT(wp);
    if (wp == NULL)
        return;

    // Make sure the object is acked
    UAVObject::Metadata initialMeta = wp->getMetadata();
    UAVObject::Metadata meta = initialMeta;
    UAVObject::SetFlightTelemetryAcked(meta, true);
    wp->setMetadata(meta);

    for(int x=0;x<myModel->rowCount();++x)
    {
        Waypoint *wp = NULL;
        double distance;
        double bearing;
        double altitude;

        // Get the number of existing waypoints
        int instances=objManager->getNumInstances(Waypoint::OBJID);

        // Create new instances of waypoints if this is more than exist
        if(x>instances-1)
        {
            wp=new Waypoint;
            wp->initialize(x,wp->getMetaObject());
            objManager->registerObject(wp);
        }
        else
        {
            wp=Waypoint::GetInstance(objManager,x);
        }

        Q_ASSERT(wp);
        Waypoint::DataFields waypoint = wp->getData();

        // Fetch the data from the internal model
        distance=myModel->data(myModel->index(x,FlightDataModel::DISRELATIVE)).toDouble();
        bearing=myModel->data(myModel->index(x,FlightDataModel::BEARELATIVE)).toDouble();
        altitude=myModel->data(myModel->index(x,FlightDataModel::ALTITUDERELATIVE)).toFloat();
        waypoint.Velocity=myModel->data(myModel->index(x,FlightDataModel::VELOCITY)).toFloat();

        waypoint.Position[Waypoint::POSITION_NORTH]=distance*cos(bearing/180*M_PI);
        waypoint.Position[Waypoint::POSITION_EAST]=distance*sin(bearing/180*M_PI);
        waypoint.Position[Waypoint::POSITION_DOWN]=(-1.0f)*altitude;
        waypoint.Mode = myModel->data(myModel->index(x,FlightDataModel::MODE)).toInt();
        waypoint.ModeParameters = myModel->data(myModel->index(x,FlightDataModel::MODE_PARAMS)).toFloat();

        if (robustUpdate(waypoint, x))
            qDebug() << "Successfully updated";
        else {
            qDebug() << "Upload failed";
            break;
        }
    }
    wp->setMetadata(initialMeta);
}

/**
 * @brief robustUpdate Upload a waypoint and check for an ACK or retry.
 * @param data The data to set
 * @param instance The instance id
 * @return True if set succeed, false otherwise
 */
bool ModelUavoProxy::robustUpdate(Waypoint::DataFields data, int instance)
{
    Waypoint *wp = Waypoint::GetInstance(objManager, instance);
    connect(wp, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(waypointTransactionCompleted(UAVObject *, bool)));
    for (int i = 0; i < 10; i++) {
            QEventLoop m_eventloop;
            QTimer::singleShot(500, &m_eventloop, SLOT(quit()));
            connect(this, SIGNAL(waypointTransactionSucceeded()), &m_eventloop, SLOT(quit()));
            connect(this, SIGNAL(waypointTransactionFailed()), &m_eventloop, SLOT(quit()));
            waypointTransactionResult.insert(instance, false);
            wp->setData(data);
            wp->updated();
            m_eventloop.exec();
            if (waypointTransactionResult.value(instance)) {
                disconnect(wp, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(waypointTransactionCompleted(UAVObject *, bool)));
                return true;
            }

            // Wait a second for next attempt
            QTimer::singleShot(500, &m_eventloop, SLOT(quit()));
            m_eventloop.exec();
    }

    disconnect(wp, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(waypointTransactionCompleted(UAVObject *, bool)));

    // None of the attempt got an ack
    return false;
}

/**
 * @brief waypointTransactionCompleted Map from the transaction complete to whether it
 * did or not
 */
void ModelUavoProxy::waypointTransactionCompleted(UAVObject *obj, bool success) {
    Q_ASSERT(obj->getObjID() == Waypoint::OBJID);
    waypointTransactionResult.insert(obj->getInstID(), success);
    if (success) {
        qDebug() << "Success " << obj->getInstID();
        emit waypointTransactionSucceeded();
    } else {
        qDebug() << "Failed transaction " << obj->getInstID();
        emit waypointTransactionFailed();
    }
}

/**
 * @brief ModelUavoProxy::objectsToModel Take the existing UAV objects and
 * update the GCS model accordingly
 */
void ModelUavoProxy::objectsToModel()
{
    myModel->removeRows(0,myModel->rowCount());
    for(int x=0;x<objManager->getNumInstances(waypointObj->getObjID());++x)
    {
        Waypoint * wp;
        Waypoint::DataFields wpfields;
        double distance;
        double bearing;

        wp = Waypoint::GetInstance(objManager,x);
        Q_ASSERT(wp);

        if(!wp)
            continue;

        // Get the waypoint data from the object manager and prepare a row in the internal model
        wpfields = wp->getData();
        myModel->insertRow(x);

        // Calculate relative position and angle
        distance = sqrt(wpfields.Position[Waypoint::POSITION_NORTH]*wpfields.Position[Waypoint::POSITION_NORTH]+
                      wpfields.Position[Waypoint::POSITION_EAST]*wpfields.Position[Waypoint::POSITION_EAST]);
        bearing = atan2(wpfields.Position[Waypoint::POSITION_EAST],wpfields.Position[Waypoint::POSITION_NORTH])*180/M_PI;

        // Check for NAN
        if(bearing!=bearing)
            bearing=0;

        // Compute the coordinates in LLA
        HomeLocation *home = HomeLocation::GetInstance(objManager);
        Q_ASSERT(home);
        HomeLocation::DataFields homeLocation = home->getData();
        double homeLLA[3];
        homeLLA[0] = homeLocation.Latitude / 1e7;
        homeLLA[1] = homeLocation.Longitude / 1e7;
        homeLLA[2] = homeLocation.Altitude;
        double LLA[3];
        double NED[3] = {wpfields.Position[0], wpfields.Position[1], wpfields.Position[2]};

        Utils::CoordinateConversions().NED2LLA_HomeLLA(homeLLA, NED, LLA);

        // Store the data
        myModel->setData(myModel->index(x,FlightDataModel::LATPOSITION), LLA[0]);
        myModel->setData(myModel->index(x,FlightDataModel::LNGPOSITION), LLA[1]);
        myModel->setData(myModel->index(x,FlightDataModel::VELOCITY), wpfields.Velocity);
        myModel->setData(myModel->index(x,FlightDataModel::DISRELATIVE), distance);
        myModel->setData(myModel->index(x,FlightDataModel::BEARELATIVE), bearing);
        myModel->setData(myModel->index(x,FlightDataModel::ALTITUDERELATIVE),
                         (-1.0f)*wpfields.Position[Waypoint::POSITION_DOWN]);
        myModel->setData(myModel->index(x,FlightDataModel::ISRELATIVE), true);
        myModel->setData(myModel->index(x,FlightDataModel::MODE), wpfields.Mode);
        myModel->setData(myModel->index(x,FlightDataModel::MODE_PARAMS), wpfields.ModeParameters);
    }
}
