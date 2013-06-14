/**
 ******************************************************************************
 * @file       geofenceeditorgadgetwidget.h
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

#ifndef GEOFENCEEDITORGADGETWIDGET_H
#define GEOFENCEEDITORGADGETWIDGET_H

#include <QLabel>
#include <QItemSelectionModel>
#include "geofenceverticesdatamodel.h"
#include "geofencefacesdatamodel.h"
#include "geofencemodeluavoproxy.h"
#include "geofenceeditor_global.h"

#include "kml/base/file.h"
#include "kml/dom.h"
#include "kml/engine.h"

using kmldom::KmlFactory;
using kmldom::BalloonStylePtr;
using kmldom::ContainerPtr;
using kmldom::DocumentPtr;
using kmldom::ElementPtr;
using kmldom::PlacemarkPtr;
using kmldom::PolygonPtr;
using kmldom::FeaturePtr;
using kmldom::GeometryPtr;
using kmldom::MultiGeometryPtr;
using kmldom::OuterBoundaryIsPtr;
using kmldom::LinearRingPtr;
using kmldom::CoordinatesPtr;
using kmldom::SnippetPtr;
using kmldom::StylePtr;
//using kmldom::SerializePretty;
//using kmlengine::CreateBalloonText;
using kmlengine::FeatureVisitor;
using kmlengine::GetRootFeature;
using kmlengine::KmlFile;
using kmlengine::KmzFile;
using kmlengine::KmlFilePtr;
//using kmlengine::VisitFeatureHierarchy;
using kmldom::KmlPtr;

class Ui_GeoFenceDialog;


class GEOFENCEEDITOR_EXPORT GeoFenceEditorGadgetWidget : public QLabel
{
    Q_OBJECT

public:
    GeoFenceEditorGadgetWidget(QWidget *parent = 0);
    ~GeoFenceEditorGadgetWidget();

    void setModel(GeoFenceVerticesDataModel *verticesModel, QItemSelectionModel *verticesSelection, GeoFenceFacesDataModel *facesModel, QItemSelectionModel *facesSelection);
private slots:
    void on_tbDelete_clicked();

    void on_tbInsert_clicked();

    void on_tbReadFromFile_clicked();

    void on_tbSaveToFile_clicked();

    void on_tbDetails_clicked();

    void on_tbSendToUAV_clicked();

    void on_tbFetchFromUAV_clicked();

private:
    void WalkGeometry(const GeometryPtr& geometry);
    void WalkFeature(const FeaturePtr& feature);
    void WalkContainer(const ContainerPtr& container);

    Ui_GeoFenceDialog  *ui;
    GeoFenceVerticesDataModel *verticesDataModel;
    GeoFenceFacesDataModel *facesDataModel;
    GeoFenceModelUavoProxy  *proxy;

    QVector<GeoFenceVerticesData> importedVertices;
    QVector<GeoFenceFacesData>    importedFaces;

signals:
    void sendPathPlanToUAV();
    void receivePathPlanFromUAV();
};

#endif /* GEOFENCEEDITORGADGETWIDGET_H */
