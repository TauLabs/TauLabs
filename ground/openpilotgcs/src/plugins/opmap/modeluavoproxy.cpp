/**
 ******************************************************************************
 *
 * @file       modeluavproxy.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup OPMapPlugin OpenPilot Map Plugin
 * @{
 * @brief The OpenPilot Map plugin
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
#include "modeluavoproxy.h"
#include "extensionsystem/pluginmanager.h"
#include <math.h>

#include "utils/coordinateconversions.h"
#include "homelocation.h"

//! Initialize the model uavo proxy
modelUavoProxy::modelUavoProxy(QObject *parent,flightDataModel * model):QObject(parent),myModel(model)
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm != NULL);
    objManager = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objManager != NULL);
    waypointObj = Waypoint::GetInstance(objManager);
    Q_ASSERT(waypointObj != NULL);
}

/**
 * @brief modelUavoProxy::modelToObjects Cast from the internal representation of a path
 * to the UAV objects required to represent it
 */
void modelUavoProxy::modelToObjects()
{
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
        distance=myModel->data(myModel->index(x,flightDataModel::DISRELATIVE)).toDouble();
        bearing=myModel->data(myModel->index(x,flightDataModel::BEARELATIVE)).toDouble();
        altitude=myModel->data(myModel->index(x,flightDataModel::ALTITUDERELATIVE)).toFloat();
        waypoint.Velocity=myModel->data(myModel->index(x,flightDataModel::VELOCITY)).toFloat();

        waypoint.Position[Waypoint::POSITION_NORTH]=distance*cos(bearing/180*M_PI);
        waypoint.Position[Waypoint::POSITION_EAST]=distance*sin(bearing/180*M_PI);
        waypoint.Position[Waypoint::POSITION_DOWN]=(-1.0f)*altitude;

        wp->setData(waypoint);
        wp->updated();
    }
}

/**
 * @brief modelUavoProxy::objectsToModel Take the existing UAV objects and
 * update the GCS model accordingly
 */
void modelUavoProxy::objectsToModel()
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
        myModel->setData(myModel->index(x,flightDataModel::LATPOSITION), LLA[0]);
        myModel->setData(myModel->index(x,flightDataModel::LNGPOSITION), LLA[1]);
        myModel->setData(myModel->index(x,flightDataModel::VELOCITY), wpfields.Velocity);
        myModel->setData(myModel->index(x,flightDataModel::DISRELATIVE), distance);
        myModel->setData(myModel->index(x,flightDataModel::BEARELATIVE), bearing);
        myModel->setData(myModel->index(x,flightDataModel::ALTITUDERELATIVE),
                         (-1.0f)*wpfields.Position[Waypoint::POSITION_DOWN]);
        myModel->setData(myModel->index(x,flightDataModel::ISRELATIVE), true);
        myModel->setData(myModel->index(x,flightDataModel::MODE), wpfields.Mode);
    }
}
