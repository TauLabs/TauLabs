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
#include <QMessageBox>
#include <QTimer>
#include "geofencemodeluavoproxy.h"
#include "extensionsystem/pluginmanager.h"
#include <math.h>

#include "utils/coordinateconversions.h"
#include "homelocation.h"

//! Initialize the model uavo proxy
GeoFenceModelUavoProxy::GeoFenceModelUavoProxy(QObject *parent, GeoFenceVerticesDataModel *verticesModel, GeoFenceFacesDataModel *facesModel):
    QObject(parent),
    myVerticesModel(verticesModel),
    myFacesModel(facesModel)
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    objManager = pm->getObject<UAVObjectManager>();
    geofenceFaces = GeoFenceFaces::GetInstance(objManager);
    geofenceVertices = GeoFenceVertices::GetInstance(objManager);
}

/**
 * @brief GeoFenceModelUavoProxy::modelToObjects Cast from the internal representation of a path
 * to the UAV objects required to represent it
 */
void GeoFenceModelUavoProxy::modelToObjects()
{
    // Test if polyhedron is closed
    GeoFenceModelPolyhedron ret = isPolyhedronClosed();

    switch (ret) {
    case GEOFENCE_DUPLICATE_VERTICES:
        QMessageBox::critical(new QWidget(),"Duplicate VertexID", "The geofence cannot be uploaded. There are duplicate Vertex IDs.");
        return;
        break;
    case GEOFENCE_DUPLICATE_FACES:
        QMessageBox::critical(new QWidget(),"Duplicate FaceID", "The geofence cannot be uploaded. There are duplicate Face IDs.");
        return;
        break;
    case GEOFENCE_OPEN_UNEVEN_NUMBER_OF_FACES:
        QMessageBox::critical(new QWidget(),"Open polyhedron", "An open geofence cannot be uploaded. It is not closed because it has an uneven number of faces.");
        return;
        break;
    case GEOFENCE_OPEN_BOUNDARY_EDGES:
        QMessageBox::critical(new QWidget(),"Open polyhedron", "An open geofence cannot be uploaded. It is not closed because at least one face edge is a boundary.");
        return;
        break;
    case GEOFENCE_OPEN_UNUSED_VERTICES:
        QMessageBox::critical(new QWidget(),"Unused vertices", "The geofence includes unused vertices, and cannot be uploaded.");
        return;
        break;
    case GEOFENCE_CLOSED:
        break;
    }

    // Set metadata
    GeoFenceVertices *gfV = GeoFenceVertices::GetInstance(objManager, 0);
    GeoFenceFaces *gfF = GeoFenceFaces::GetInstance(objManager, 0);

    Q_ASSERT(gfF != NULL && gfV != NULL);
    if (gfF == NULL || gfV == NULL)
        return;

    // Make sure the objects are acked
    UAVObject::Metadata initialGFFMeta = gfF->getMetadata();
    UAVObject::Metadata meta = initialGFFMeta;
    UAVObject::SetFlightTelemetryAcked(meta, true);
    gfF->setMetadata(meta);

    UAVObject::Metadata initialGFVMeta = gfV->getMetadata();
    meta = initialGFVMeta;
    UAVObject::SetFlightTelemetryAcked(meta, true);
    gfV->setMetadata(meta);

    for(int x =0; x < myVerticesModel->rowCount(); ++x)
    {
        GeoFenceVertices *gfV = NULL;

        // Get the number of existing waypoints
        int vertexInstances=objManager->getNumInstances(geofenceVertices->getObjID());

        // Create new instances of vertices if this is more than exist
        if (x > vertexInstances-1)
        {
            gfV = new GeoFenceVertices;
            gfV->initialize(x, gfV->getMetaObject());
            objManager->registerObject(gfV);
        }
        else
        {
            gfV=GeoFenceVertices::GetInstance(objManager, x);
        }

        Q_ASSERT(gfV);
        if (gfV == NULL) {
            qDebug() << "Fetching of GeoFenceVertices UAVO failed";
            return;
        }


        // Fetch the data from the internal model
        GeoFenceVertices::DataFields geofenceVerticesData = gfV->getData();

        geofenceVerticesData.Latitude = lround(myVerticesModel->data(myVerticesModel->index(x, GeoFenceVerticesDataModel::GEO_LATITUDE)).toDouble() * 10e6);
        geofenceVerticesData.Longitude = lround(myVerticesModel->data(myVerticesModel->index(x, GeoFenceVerticesDataModel::GEO_LONGITUDE)).toDouble() * 10e6);
        geofenceVerticesData.Altitude = myVerticesModel->data(myVerticesModel->index(x, GeoFenceVerticesDataModel::GEO_ALTITUDE)).toDouble();

        qDebug() << "Another vertices: " << x << " out of: "<< myVerticesModel->rowCount();

        // Send update
        bool ret = robustUpdateVertices(geofenceVerticesData, x);
        if (! ret) {
            qDebug() << "Geo-fence vertices failed";
//            break; //FixMe: Not receiving signal about successfully updated object
        }
    }

    for(int x = 0; x < myFacesModel->rowCount(); ++x)
    {
        GeoFenceFaces *gfF = NULL;

        // Get the number of existing waypoints
        int faceInstances=objManager->getNumInstances(geofenceFaces->getObjID());

        // Create new instances of faces if this is more than exist
        if(x>faceInstances-1)
        {
            gfF = new GeoFenceFaces;
            gfF->initialize(x, gfF->getMetaObject());
            objManager->registerObject(gfF);
        }
        else
        {
            gfF=GeoFenceFaces::GetInstance(objManager, x);
        }

        Q_ASSERT(gfF);
        if (gfF == NULL) {
            qDebug() << "Fetching of GeoFenceFaces UAVO failed";
            return;
        }


        // Fetch the data from the internal model
        GeoFenceFaces::DataFields geofenceFacesData = gfF->getData();

        geofenceFacesData.Vertices[GeoFenceFaces::VERTICES_A] = myFacesModel->data(myFacesModel->index(x, GeoFenceFacesDataModel::GEO_VERTEX_A)).toUInt();
        geofenceFacesData.Vertices[GeoFenceFaces::VERTICES_B] = myFacesModel->data(myFacesModel->index(x, GeoFenceFacesDataModel::GEO_VERTEX_B)).toUInt();
        geofenceFacesData.Vertices[GeoFenceFaces::VERTICES_C] = myFacesModel->data(myFacesModel->index(x, GeoFenceFacesDataModel::GEO_VERTEX_C)).toUInt();

        qDebug() << "Another fences: " << x << " out of: "<< myVerticesModel->rowCount();

        // Send update
        bool ret = robustUpdateFaces(geofenceFacesData, x);
        if (ret)
            qDebug() << "Successfully updated geo-fence: " << ret;
        else {
            qDebug() << "Geo-fence faces failed";
//            break; //FixMe: Not receiving signal about successfully updated object
        }
    }

    // Reset metadata
    gfV->setMetadata(initialGFVMeta);
    gfF->setMetadata(initialGFFMeta);
}

/**
 * @brief robustUpdateVertices Upload vertices and check for an ACK or retry.
 * @param data The data to set
 * @param instance The instance id
 * @return True if set succeed, false otherwise
 */
bool GeoFenceModelUavoProxy::robustUpdateVertices(GeoFenceVertices::DataFields geofenceVerticesData, int instance)
{
    uint8_t success = 0;

    GeoFenceVertices *gfV = GeoFenceVertices::GetInstance(objManager, instance);
    connect(gfV, SIGNAL(transactionCompleted(UAVObject*, bool)), this, SLOT(geofenceTransactionCompleted(UAVObject *, bool)));
    for (int i = 0; i < 10 && success == 0; i++) {
        QEventLoop m_eventloop;
        QTimer::singleShot(500, &m_eventloop, SLOT(quit())); // Allow 500ms for the transaction to complete.
        connect(this, SIGNAL(geofenceTransactionSucceeded()), &m_eventloop, SLOT(quit()));
        connect(this, SIGNAL(geofenceTransactionFailed()), &m_eventloop, SLOT(quit()));
        geofenceTransactionResult.insert(instance, false);
        gfV->setData(geofenceVerticesData);
        gfV->updated();
        qDebug() << "Stupid loop";
        m_eventloop.exec();
        if (geofenceTransactionResult.value(instance)) {
            success++;
            break;
        }
        else {
            // Wait a bit before next attempt
            QTimer::singleShot(500, &m_eventloop, SLOT(quit()));
            m_eventloop.exec();
        }

        break; //<--- FIXME: REMOVEME
    }
    disconnect(gfV, SIGNAL(transactionCompleted(UAVObject*, bool)), this, SLOT(geofenceTransactionCompleted(UAVObject *, bool)));

    return success;
}


/**
 * @brief robustUpdateFaces Upload faces and check for an ACK or retry.
 * @param data The data to set
 * @param instance The instance id
 * @return True if set succeed, false otherwise
 */
bool GeoFenceModelUavoProxy::robustUpdateFaces(GeoFenceFaces::DataFields geofenceFacesData, int instance)
{
    uint8_t success = 0;

    GeoFenceFaces *gfF = GeoFenceFaces::GetInstance(objManager, instance);
    connect(gfF, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(geofenceTransactionCompleted(UAVObject *, bool)));
    for (int i = 0; i < 10 && success == 0; i++) {
        QEventLoop m_eventloop;
        QTimer::singleShot(500, &m_eventloop, SLOT(quit()));
        connect(this, SIGNAL(geofenceTransactionSucceeded()), &m_eventloop, SLOT(quit()));
        connect(this, SIGNAL(geofenceTransactionFailed()), &m_eventloop, SLOT(quit()));
        geofenceTransactionResult.insert(instance, false);
        gfF->setData(geofenceFacesData);
        gfF->updated();
        qDebug() << "Even stupider loop";
        m_eventloop.exec();
        if (geofenceTransactionResult.value(instance)) {
            success++;
            break;
        }
        else {
            // Wait a bit before next attempt
            QTimer::singleShot(500, &m_eventloop, SLOT(quit()));
            m_eventloop.exec();
        }

        break; //<--- FIXME: REMOVEME
    }
    disconnect(gfF, SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(geofenceTransactionCompleted(UAVObject *, bool)));

    return success;
}


/**
 * @brief geofenceTransactionCompleted Map from the transaction complete to whether it
 * did or not
 */
void GeoFenceModelUavoProxy::geofenceTransactionCompleted(UAVObject *obj, bool success)
{
    geofenceTransactionResult.insert(obj->getInstID(), success);
    if (success) {
        qDebug() << "Success " << obj->getInstID();
        emit geofenceTransactionSucceeded();
    } else {
        qDebug() << "Failed transaction " << obj->getInstID();
        emit geofenceTransactionFailed();
    }
}


/**
 * @brief GeoFenceModelUavoProxy::objectsToModel Take the existing UAV objects and
 * update the GCS model accordingly
 */
void GeoFenceModelUavoProxy::objectsToModel()
{
    // Remove all rows
    myVerticesModel->removeRows(0, myVerticesModel->rowCount());
    myFacesModel->removeRows(0, myFacesModel->rowCount());

    // Add vertices rows back
    for(int x = 0; x < objManager->getNumInstances(GeoFenceVertices::OBJID); ++x) {
        GeoFenceVertices *gfV;
        GeoFenceVertices::DataFields gfVdata;

        gfV = GeoFenceVertices::GetInstance(objManager,x);

        Q_ASSERT(gfV);
        if(!gfV)
            continue;

        // Get the waypoint data from the object manager and prepare a row in the internal model
        gfVdata = gfV->getData();
        myVerticesModel->insertRow(x);

        // Store the data
        myVerticesModel->setData(myVerticesModel->index(x, GeoFenceVerticesDataModel::GEO_LATITUDE),  gfVdata.Latitude/10e6);
        myVerticesModel->setData(myVerticesModel->index(x, GeoFenceVerticesDataModel::GEO_LONGITUDE), gfVdata.Longitude/10e6);
        myVerticesModel->setData(myVerticesModel->index(x, GeoFenceVerticesDataModel::GEO_ALTITUDE),  gfVdata.Altitude);
        myVerticesModel->setData(myVerticesModel->index(x, GeoFenceVerticesDataModel::GEO_VERTEX_ID), x);
    }

    // Add faces rows back
    for(int x = 0; x < objManager->getNumInstances(GeoFenceFaces::OBJID); ++x) {
        GeoFenceFaces *gfF;
        GeoFenceFaces::DataFields gfFdata;

        gfF = GeoFenceFaces::GetInstance(objManager, x);

        Q_ASSERT(gfF);
        if(!gfF)
            continue;

        // Get the waypoint data from the object manager and prepare a row in the internal model
        gfFdata = gfF->getData();
        myFacesModel->insertRow(x);

        // Store the data
        myFacesModel->setData(myFacesModel->index(x, GeoFenceFacesDataModel::GEO_VERTEX_A), gfFdata.Vertices[GeoFenceFaces::VERTICES_A]);
        myFacesModel->setData(myFacesModel->index(x, GeoFenceFacesDataModel::GEO_VERTEX_B), gfFdata.Vertices[GeoFenceFaces::VERTICES_B]);
        myFacesModel->setData(myFacesModel->index(x, GeoFenceFacesDataModel::GEO_VERTEX_C), gfFdata.Vertices[GeoFenceFaces::VERTICES_C]);
        myFacesModel->setData(myFacesModel->index(x, GeoFenceFacesDataModel::GEO_FACE_ID),  x);
    }
}

bool sortRows (QVector<double> i, QVector<double> j)
{
    if (i[0] < j[0])
        return true;
    else if (i[0] > j[0])
        return false;
    else if (i[1] < j[1])
        return true;
    else
        return false;
}


/**
 * @brief GeoFenceModelUavoProxy::isPolyhedronClosed  Test if polyhedron is closed, i.e. if it's a polytope, i.e. if it's a closed manifold.
 * @return
 */
GeoFenceModelPolyhedron GeoFenceModelUavoProxy::isPolyhedronClosed()
{
    // A polyhedron is closed if and only if each edge is shared by exactly two triangles
    QVector< QVector<double> > edgeList;

    // Check that all vertice and face IDs are unique
    for (int i=0; i<myVerticesModel->rowCount(); i++ ) {
        myVerticesModel->data(myVerticesModel->index(i, GeoFenceVerticesDataModel::GEO_VERTEX_ID)).toUInt();
        for (int j=i+1; j<myVerticesModel->rowCount(); j++ ) {
            if (myVerticesModel->data(myVerticesModel->index(i, GeoFenceFacesDataModel::GEO_FACE_ID)).toUInt() == myVerticesModel->data(myVerticesModel->index(j, GeoFenceFacesDataModel::GEO_FACE_ID)).toUInt())
                return GEOFENCE_DUPLICATE_VERTICES;
        }
    }
    for (int i=0; i<myFacesModel->rowCount(); i++ ) {
        for (int j=i+1; j<myFacesModel->rowCount(); j++ ) {
            if (myFacesModel->data(myFacesModel->index(i, GeoFenceFacesDataModel::GEO_FACE_ID)).toUInt() == myFacesModel->data(myFacesModel->index(j, GeoFenceFacesDataModel::GEO_FACE_ID)).toUInt())
                return GEOFENCE_DUPLICATE_FACES;
        }
    }

    // Compile list of all edges
    for (int i=0; i<myFacesModel->rowCount(); i++ ) {
        QVector<double> edge(2);

        edge[0] = myFacesModel->data(myFacesModel->index(i, GeoFenceFacesDataModel::GEO_VERTEX_A)).toUInt();
        edge[1] = myFacesModel->data(myFacesModel->index(i, GeoFenceFacesDataModel::GEO_VERTEX_B)).toUInt();
        edgeList.append(edge);

        edge[0] = myFacesModel->data(myFacesModel->index(i, GeoFenceFacesDataModel::GEO_VERTEX_B)).toUInt();
        edge[1] = myFacesModel->data(myFacesModel->index(i, GeoFenceFacesDataModel::GEO_VERTEX_C)).toUInt();
        edgeList.append(edge);

        edge[0] = myFacesModel->data(myFacesModel->index(i, GeoFenceFacesDataModel::GEO_VERTEX_A)).toUInt();
        edge[1] = myFacesModel->data(myFacesModel->index(i, GeoFenceFacesDataModel::GEO_VERTEX_C)).toUInt();
        edgeList.append(edge);
    }

    // Check if all vertices are used
    for (int i=0; i<myVerticesModel->rowCount(); i++ ) {
        bool isUsed = false;
        foreach(QVector<double> edge, edgeList) {
            if(edge.contains(myVerticesModel->data(myVerticesModel->index(i, GeoFenceVerticesDataModel::GEO_VERTEX_ID)).toUInt())) {
                isUsed = true;
                break;
            }
        }
        if (!isUsed)
            return GEOFENCE_OPEN_UNUSED_VERTICES;
    }

    // If the number of edges is odd, then the manifold has a boundary, i.e. it is open
    if ((edgeList.size() % 2) != 0)
        return GEOFENCE_OPEN_UNEVEN_NUMBER_OF_FACES;

    // If all edges aren't used exactly twice, then the manifold has a boundary.
    // 1) Sort left-to-right
    for(int i=0; i<edgeList.size(); i++) {
        QVector<double> edge = edgeList.at(i);
        if(edge[0] > edge[1]) {
            double tmp = edge[0];
            edge[0] = edge[1];
            edge[1] = tmp;
        }
        edgeList.replace(i, edge);
     }

    // 2) Sort rows
    std::sort(edgeList.begin(), edgeList.end(), sortRows);

    // 3) Test if each item appears in the list exactly twice.
    for (int i=0; i < edgeList.size()-1; i+=2) {
        if ((edgeList[i][0] != edgeList[i+1][0]) && (edgeList[i][1] != edgeList[i+1][1])) {
            return GEOFENCE_OPEN_BOUNDARY_EDGES;
        }
    }

    return GEOFENCE_CLOSED;
}
