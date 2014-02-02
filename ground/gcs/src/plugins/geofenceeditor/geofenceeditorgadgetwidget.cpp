/**
 ******************************************************************************
 * @file       geofenceeditorgadgetwidget.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013.
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
#include "geofenceeditorgadgetwidget.h"
//#include "waypointdialog.h"
//#include "waypointdelegate.h"
#include "ui_geofence_dialog.h"

#include <QDomDocument>
#include <QFileDialog>
#include <QMessageBox>
#include <QPushButton>
#include <QString>
#include <QStringList>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>
#include <QDebug>
#include <math.h>

#include "extensionsystem/pluginmanager.h"


void GeoFenceEditorGadgetWidget::WalkGeometry(const GeometryPtr& geometry) {
    if (!geometry) {
        return;
    }

    // Print the Geometry type.
    switch(geometry->Type()) {
    case kmldom::Type_Polygon:
    {
        const PolygonPtr polygon = kmldom::AsPolygon(geometry);
        const OuterBoundaryIsPtr outerBoundary = polygon->get_outerboundaryis();

        const LinearRingPtr linearRing = outerBoundary->get_linearring();

        const CoordinatesPtr coordinates = linearRing->get_coordinates();

        int numCoordinatesTuples = coordinates->get_coordinates_array_size();

        if (numCoordinatesTuples == 4 &&
                !(coordinates->get_coordinates_array_at(0) == coordinates->get_coordinates_array_at(1)) && //Note: libkml does not implement a != operator for kmlbase::Vec3
                !(coordinates->get_coordinates_array_at(0) == coordinates->get_coordinates_array_at(2)) &&
                !(coordinates->get_coordinates_array_at(1) == coordinates->get_coordinates_array_at(2)) &&
                coordinates->get_coordinates_array_at(0) == coordinates->get_coordinates_array_at(3)) // Note that the first and fourth coordinates must equal each other.
        {
            GeoFenceFacesData newFace;
            for (int i=0; i<3; i++) {
                kmlbase::Vec3 coordinates_vec = coordinates->get_coordinates_array_at(i);
                qDebug() << "Coords: " << coordinates_vec.get_latitude() << ", "  << coordinates_vec.get_longitude()  <<  ", "  << coordinates_vec.get_altitude();

#define EPSILON  1e-6
                int existingVertex = importedVertices.size();
                for(int j=0; j<importedVertices.size(); j++) {
                    GeoFenceVerticesData vertex = importedVertices.at(j);
                    // Check if this is a new vertex
                    if (fabs(vertex.latitude -coordinates_vec.get_latitude()) < EPSILON  &&
                            fabs(vertex.longitude - coordinates_vec.get_longitude())  < EPSILON &&
                            fabs(vertex.altitude - coordinates_vec.get_altitude())  < EPSILON) {
                        // This is a previously existing vertex
                        existingVertex = j;
                        break;
                    }
                }

                // If the vertex is new, add it to the list
                if (existingVertex == importedVertices.size()) {
                    GeoFenceVerticesData newVertex;

                    newVertex.latitude = coordinates_vec.get_latitude();
                    newVertex.longitude = coordinates_vec.get_longitude();
                    newVertex.altitude = coordinates_vec.get_altitude();
                    newVertex.vertexId = importedVertices.size();
                    importedVertices.append(newVertex);
                }

                // Populate face with vertices, and sorting the vertices by ascending order
                switch (i) {
                case 0:
                    newFace.vertexA = existingVertex;
                    break;
                case 1:
                    if (existingVertex > newFace.vertexA)
                        newFace.vertexB = existingVertex;
                    else {
                        newFace.vertexB = newFace.vertexA;
                        newFace.vertexA = existingVertex;
                    }
                    break;
                case 2:
                    if (existingVertex > newFace.vertexB)
                        newFace.vertexC = existingVertex;
                    else if (existingVertex > newFace.vertexA) {
                        newFace.vertexC = newFace.vertexB;
                        newFace.vertexB = existingVertex;
                    }
                    else {
                        newFace.vertexC = newFace.vertexB;
                        newFace.vertexB = newFace.vertexA;
                        newFace.vertexA = existingVertex;
                    }
                    break;
                }
            }

            newFace.faceID = importedFaces.size();
            importedFaces.append(newFace);
        }
        else {
            //TODO: This KML file is bad. Quit the whole import process
            qDebug() << "Bad KML file. Halting import.";
            Q_ASSERT(0);
        }

        break;
    }
    case kmldom::Type_MultiGeometry:
    case kmldom::Type_Model:
    case kmldom::Type_Point:
    case kmldom::Type_LineString:
    case kmldom::Type_LinearRing:
    default:  // KML has 6 types of Geometry
      break;
  }

    // Recurse into <MultiGeometry>
    const MultiGeometryPtr multigeometry = kmldom::AsMultiGeometry(geometry);
    if (multigeometry != NULL) {
          for (size_t i=0; i < multigeometry->get_geometry_array_size(); ++i) {
              WalkGeometry(multigeometry->get_geometry_array_at(i));
          }
    }
}

void GeoFenceEditorGadgetWidget::WalkFeature(const FeaturePtr& feature) {
  if (feature) {
    if (const ContainerPtr container = kmldom::AsContainer(feature)) {
      WalkContainer(container);
    } else if (const PlacemarkPtr placemark = kmldom::AsPlacemark(feature)) {
      WalkGeometry(placemark->get_geometry());
    }
  }
}

void GeoFenceEditorGadgetWidget::WalkContainer(const ContainerPtr& container) {
  for (size_t i = 0; i < container->get_feature_array_size(); ++i) {
    WalkFeature(container->get_feature_array_at(i));
  }
}


GeoFenceEditorGadgetWidget::GeoFenceEditorGadgetWidget(QWidget *parent) :
    QLabel(parent)
{
    ui = new Ui_GeoFenceDialog();
    ui->setupUi(this);

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    GeoFenceVerticesDataModel *verticesModel = pm->getObject<GeoFenceVerticesDataModel>();
    GeoFenceFacesDataModel *facesModel = pm->getObject<GeoFenceFacesDataModel>();
    Q_ASSERT(verticesModel);
    Q_ASSERT(facesModel);

    QItemSelectionModel *verticesSelection = pm->getObject<QItemSelectionModel>();
    QItemSelectionModel *facesSelection = pm->getObject<QItemSelectionModel>();
    Q_ASSERT(verticesSelection);
    Q_ASSERT(facesSelection);
    setModel(verticesModel, verticesSelection, facesModel, facesSelection);
}

GeoFenceEditorGadgetWidget::~GeoFenceEditorGadgetWidget()
{
   // Do nothing
}


void GeoFenceEditorGadgetWidget::setModel(GeoFenceVerticesDataModel *verticesModel, QItemSelectionModel *verticesSelection, GeoFenceFacesDataModel *facesModel, QItemSelectionModel *facesSelection)
{
    verticesDataModel = verticesModel;
    ui->tvGeoVerticesFence->setModel(verticesDataModel);
    ui->tvGeoVerticesFence->setSelectionModel(verticesSelection);
    ui->tvGeoVerticesFence->setSelectionBehavior(QAbstractItemView::SelectRows);
    connect(verticesDataModel, SIGNAL(rowsInserted(const QModelIndex&, int, int)), this, SLOT(rowsInserted(const QModelIndex&, int, int)));
//    ui->tvGeoFence->resizeColumnsToContents();

    facesDataModel = facesModel;
    ui->tvGeoFacesFence->setModel(facesDataModel);
    ui->tvGeoFacesFence->setSelectionModel(facesSelection);
    ui->tvGeoFacesFence->setSelectionBehavior(QAbstractItemView::SelectRows);
    connect(facesDataModel, SIGNAL(rowsInserted(const QModelIndex&, int, int)), this, SLOT(rowsInserted(const QModelIndex&, int, int)));

    proxy = new GeoFenceModelUavoProxy(this, verticesDataModel, facesDataModel);
}


void GeoFenceEditorGadgetWidget::on_tbReadFromFile_clicked()
{
    if(!verticesDataModel || !facesDataModel)
        return;

    importedVertices.clear();
    importedFaces.clear();


    // Get file name from file picker dialog
    QString filters = tr("KML files (*.kml *.kmz);; XML files (*.xml)");
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), "", filters);

    std::string file_data;
    if (!kmlbase::File::ReadFileToString(fileName.toStdString(), &file_data)) {
        qDebug() << "error: reading of " << fileName << " failed";
      return;
    }

    // If the file was KMZ, extract the KML file.
    std::string kml;
    if (KmzFile::IsKmz(file_data)) {
      boost::scoped_ptr<KmzFile> kmz_file(KmzFile::OpenFromString(file_data));
      if (!kmz_file.get()) {
        qDebug() << "Failed while opening KMZ file";
        return;
      }
      if (!kmz_file->ReadKml(&kml)) {
          qDebug() << "Failed to read KML from KMZ";
        return;
      }
    } else {
      kml = file_data;
    }

    std::string errors;
    KmlFilePtr kml_file = KmlFile::CreateFromParse(kml, &errors);
    if (!kml_file || !errors.empty()) {
        qDebug() << "parse of " << fileName << "failed: " << QString().fromStdString(errors);
        return;
    }

    ElementPtr root = kmldom::Parse(kml, &errors);
    Q_ASSERT(root);
    if (!root) {
        return;
    }

    const FeaturePtr feature = GetRootFeature(root);
    WalkFeature(feature);

    verticesDataModel->replaceModel(importedVertices);
    facesDataModel->replaceModel(importedFaces);
    return;
}

void GeoFenceEditorGadgetWidget::on_tbDelete_clicked()
{
//    ui->tableView->model()->removeRow(ui->tableView->selectionModel()->currentIndex().row());
}

void GeoFenceEditorGadgetWidget::on_tbInsert_clicked()
{
//    ui->tableView->model()->insertRow(ui->tableView->selectionModel()->currentIndex().row());
}


void GeoFenceEditorGadgetWidget::on_tbSaveToFile_clicked()
{
//    if(!model)
//        return;
//    QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"));
//    model->writeToFile(fileName);
}

/**
 * @brief GeoFenceEditorGadgetWidget::on_tbDetails_clicked Display a dialog to show
 * and edit details of a waypoint.  The waypoint selected initially will be the
 * highlighted one.
 */
void GeoFenceEditorGadgetWidget::on_tbDetails_clicked()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm);
    if (pm == NULL)
        return;

//    WaypointDialog *dialog =  pm->getObject<WaypointDialog>();
//    Q_ASSERT(dialog);
//    dialog->show();
}

/**
 * @brief GeoFenceEditorGadgetWidget::on_tbSendToUAV_clicked Use the proxy to send
 * the data from the flight model to the UAV
 */
void GeoFenceEditorGadgetWidget::on_tbSendToUAV_clicked()
{
    proxy->modelToObjects();
}

/**
 * @brief GeoFenceEditorGadgetWidget::on_tbFetchFromUAV_clicked Use the flight model to
 * get data from the UAV
 */
void GeoFenceEditorGadgetWidget::on_tbFetchFromUAV_clicked()
{
    proxy->objectsToModel();
//    ui->tableView->resizeColumnsToContents();
}


/**
  * @}
  * @}
  */
