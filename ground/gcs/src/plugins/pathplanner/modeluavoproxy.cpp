/**
 ******************************************************************************
 * @file       modeluavproxy.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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
 * @return true if all wp were sent ok
 */
bool ModelUavoProxy::modelToObjects()
{
    Waypoint *wp = Waypoint::GetInstance(objManager,0);
    Q_ASSERT(wp);
    if (wp == NULL)
        return false;

    // Make sure the object is acked
    UAVObject::Metadata initialMeta = wp->getMetadata();
    UAVObject::Metadata meta = initialMeta;
    UAVObject::SetFlightTelemetryAcked(meta, true);
    wp->setMetadata(meta);

    double homeLLA[3];
    double NED[3];
    double LLA[3];
    getHomeLocation(homeLLA);
    bool newInstance;
    for(int x=0;x<myModel->rowCount();++x)
    {
        newInstance = false;
        Waypoint *wp = NULL;

        // Get the number of existing waypoints
        int instances = Waypoint::getNumInstances(objManager);

        // Create new instances of waypoints if this is more than exist
        if(x>instances-1)
        {
            newInstance = true;
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

        // Convert from LLA to NED for sending to the model
        LLA[0] = myModel->data(myModel->index(x,FlightDataModel::LATPOSITION)).toDouble();
        LLA[1] = myModel->data(myModel->index(x,FlightDataModel::LNGPOSITION)).toDouble();
        LLA[2] = myModel->data(myModel->index(x,FlightDataModel::ALTITUDE)).toDouble();
        Utils::CoordinateConversions().LLA2NED_HomeLLA(LLA, homeLLA, NED);

        // Fetch the data from the internal model
        waypoint.Velocity=myModel->data(myModel->index(x,FlightDataModel::VELOCITY)).toFloat();
        waypoint.Position[Waypoint::POSITION_NORTH] = NED[0];
        waypoint.Position[Waypoint::POSITION_EAST]  = NED[1];
        waypoint.Position[Waypoint::POSITION_DOWN]  = NED[2];
        waypoint.Mode = myModel->data(myModel->index(x,FlightDataModel::MODE), Qt::UserRole).toInt();
        waypoint.ModeParameters = myModel->data(myModel->index(x,FlightDataModel::MODE_PARAMS)).toFloat();

        if (robustUpdate(waypoint, x)) {
            qDebug() << "Successfully updated";
            emit sendPathPlanToUavProgress(100 * (x + 1) / myModel->rowCount());
        }
        else {
            qDebug() << "Upload failed";
            emit sendPathPlanToUavProgress(100 * (x + 1) / myModel->rowCount());
            return false;
            break;
        }
        if(newInstance)
        {
            QEventLoop m_eventloop;
            QTimer::singleShot(800, &m_eventloop, SLOT(quit()));
            m_eventloop.exec();
        }
    }
    wp->setMetadata(initialMeta);
    return true;
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
            QTimer::singleShot(1000, &m_eventloop, SLOT(quit()));
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
    double homeLLA[3];
    getHomeLocation(homeLLA);
    double LLA[3];

    myModel->removeRows(0,myModel->rowCount());
    for(int x=0; x < Waypoint::getNumInstances(objManager) ; ++x) {
        Waypoint * wp;
        Waypoint::DataFields wpfields;

        wp = Waypoint::GetInstance(objManager,x);
        Q_ASSERT(wp);

        if(!wp)
            continue;

        // Get the waypoint data from the object manager and prepare a row in the internal model
        wpfields = wp->getData();
        myModel->insertRow(x);

        // Compute the coordinates in LLA
        double NED[3] = {wpfields.Position[Waypoint::POSITION_NORTH], wpfields.Position[Waypoint::POSITION_EAST], wpfields.Position[Waypoint::POSITION_DOWN]};
        Utils::CoordinateConversions().NED2LLA_HomeLLA(homeLLA, NED, LLA);

        // Store the data
        myModel->setData(myModel->index(x,FlightDataModel::LATPOSITION), LLA[0]);
        myModel->setData(myModel->index(x,FlightDataModel::LNGPOSITION), LLA[1]);
        myModel->setData(myModel->index(x,FlightDataModel::ALTITUDE), LLA[2]);
        myModel->setData(myModel->index(x,FlightDataModel::VELOCITY), wpfields.Velocity);
        myModel->setData(myModel->index(x,FlightDataModel::MODE), wpfields.Mode);
        myModel->setData(myModel->index(x,FlightDataModel::MODE_PARAMS), wpfields.ModeParameters);
    }
}

/**
 * @brief ModelUavoProxy::getHomeLocation Take care of scaling the home location UAVO to
 * degrees (lat lon) and meters altitude
 * @param [out] home A 3 element double array to store resul in
 * @return True if successful, false otherwise
 */
bool ModelUavoProxy::getHomeLocation(double *homeLLA)
{
    // Compute the coordinates in LLA
    HomeLocation *home = HomeLocation::GetInstance(objManager);
    if (home == NULL)
        return false;

    HomeLocation::DataFields homeLocation = home->getData();
    homeLLA[0] = homeLocation.Latitude / 1e7;
    homeLLA[1] = homeLocation.Longitude / 1e7;
    homeLLA[2] = homeLocation.Altitude;

    return true;
}

