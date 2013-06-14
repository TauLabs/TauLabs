/**
 ******************************************************************************
 * @file       geofenceeditorplugin.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup GeoFenceEditorGadgetPlugin Geo-fence Editor Gadget Plugin
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
#include "geofenceeditorplugin.h"
#include "geofenceeditorgadgetfactory.h"
#include <QtPlugin>
#include <QStringList>
#include <extensionsystem/pluginmanager.h>


GeoFenceEditorPlugin::GeoFenceEditorPlugin()
{
   // Do nothing
}

GeoFenceEditorPlugin::~GeoFenceEditorPlugin()
{
   // Do nothing
}

/**
 * @brief GeoFenceEditorPlugin::initialize Initialize the plugin which includes
 * creating the new data model
 */
bool GeoFenceEditorPlugin::initialize(const QStringList& args, QString *errMsg)
{
    Q_UNUSED(args);
    Q_UNUSED(errMsg);

    // Create a factory for making gadgets
    mf = new GeoFenceEditorGadgetFactory(this);
    addAutoReleasedObject(mf);

    // Create the data model for the flight plan
    verticesDataModel = new GeoFenceVerticesDataModel(this);
    addAutoReleasedObject(verticesDataModel);
    facesDataModel = new GeoFenceFacesDataModel(this);
    addAutoReleasedObject(facesDataModel);

    // Create a selector and add it to the plugin
    verticesSelection = new QItemSelectionModel(verticesDataModel);
    addAutoReleasedObject(verticesSelection);

    facesSelection = new QItemSelectionModel(facesDataModel);
    addAutoReleasedObject(facesSelection);

//    // Create a common dialog to be used by the map and path planner
//    waypointDialog = new WaypointDialog(NULL, verticesDataModel, verticesSelection);
//    addAutoReleasedObject(waypointDialog);

    return true;
}

void GeoFenceEditorPlugin::extensionsInitialized()
{
   // Do nothing
}

void GeoFenceEditorPlugin::shutdown()
{
   // Do nothing
}
Q_EXPORT_PLUGIN(GeoFenceEditorPlugin)

/**
  * @}
  * @}
  */
