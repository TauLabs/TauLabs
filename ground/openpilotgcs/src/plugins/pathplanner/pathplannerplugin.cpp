/**
 ******************************************************************************
 * @file       pathplannerplugin.cpp
 * @author     PhoenixPilot Project, http://github.com/PhoenixPilot Copyright (C) 2012.
 * @addtogroup Path Planner GCS Plugins
 * @{
 * @addtogroup PathPlannerGadgetPlugin Waypoint Editor Gadget Plugin
 * @{
 * @brief A gadget to edit a list of waypoints
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
#include "pathplannerplugin.h"
#include "pathplannergadgetfactory.h"
#include <QDebug>
#include <QtPlugin>
#include <QStringList>
#include <extensionsystem/pluginmanager.h>


PathPlannerPlugin::PathPlannerPlugin()
{
   // Do nothing
}

PathPlannerPlugin::~PathPlannerPlugin()
{
   // Do nothing
}

/**
 * @brief PathPlannerPlugin::initialize Initialize the plugin which includes
 * creating the new data model
 */
bool PathPlannerPlugin::initialize(const QStringList& args, QString *errMsg)
{
    Q_UNUSED(args);
    Q_UNUSED(errMsg);

    // Create a factory for making gadgets
    mf = new PathPlannerGadgetFactory(this);
    addAutoReleasedObject(mf);

    // Create the data model for the flight plan
    dataModel = new FlightDataModel(this);
    addAutoReleasedObject(dataModel);

    // Create a selector and add it to the plugin
    selection = new QItemSelectionModel(dataModel);
    addAutoReleasedObject(selection);

    // Create a common dialog to be used by the map and path planner
    waypointDialog = new WaypointDialog(NULL, dataModel, selection);
    addAutoReleasedObject(waypointDialog);

    return true;
}

void PathPlannerPlugin::extensionsInitialized()
{
   // Do nothing
}

void PathPlannerPlugin::shutdown()
{
   // Do nothing
}
Q_EXPORT_PLUGIN(PathPlannerPlugin)

/**
  * @}
  * @}
  */
