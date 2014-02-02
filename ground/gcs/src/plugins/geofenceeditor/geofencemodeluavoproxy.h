/**
 ******************************************************************************
 * @file       geofencemodeluavproxy.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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
#ifndef GEOFENCEUAVOPROXY_H
#define GEOFENCEUAVOPROXY_H

#include <QObject>
#include "geofenceverticesdatamodel.h"
#include "geofencefacesdatamodel.h"
#include "geofencefaces.h"
#include "geofencevertices.h"

enum GeoFenceModelPolyhedron {
    GEOFENCE_CLOSED,
    GEOFENCE_OPEN_UNEVEN_NUMBER_OF_FACES,
    GEOFENCE_OPEN_BOUNDARY_EDGES,
    GEOFENCE_OPEN_UNUSED_VERTICES,
    GEOFENCE_DUPLICATE_FACES,
    GEOFENCE_DUPLICATE_VERTICES,
};

class GeoFenceModelUavoProxy:public QObject
{

    Q_OBJECT
public:
    explicit GeoFenceModelUavoProxy(QObject *parent, GeoFenceVerticesDataModel *verticesModel, GeoFenceFacesDataModel *facesModel);

private:
    //! Test if geofence is closed
    GeoFenceModelPolyhedron isPolyhedronClosed();

    //! Robustly upload a geofence (like smart save)
    bool robustUpdateVertices(GeoFenceVertices::DataFields geofenceVerticesData, int instance);
    bool robustUpdateFaces(GeoFenceFaces::DataFields geofenceFacesData, int instance);

public slots:
    //! Cast from the internal representation to the UAVOs
    void modelToObjects();

    //! Cast from the UAVOs to the internal representation
    void objectsToModel();

    //! Whenever a geofence transaction is completed
    void geofenceTransactionCompleted(UAVObject *, bool);

signals:
    void geofenceTransactionSucceeded();
    void geofenceTransactionFailed();

private:
    UAVObjectManager *objManager;
    GeoFenceFaces *geofenceFaces;
    GeoFenceVertices *geofenceVertices;
    GeoFenceVerticesDataModel *myVerticesModel;
    GeoFenceFacesDataModel *myFacesModel;

    //! Track if each geofence was updated
    QMap<int, bool>  geofenceTransactionResult;

};

#endif // GEOFENCEUAVOPROXY_H
