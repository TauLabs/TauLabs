/**
 ******************************************************************************
 *
 * @file       geofencefacesdatamodel.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013.
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
#ifndef GEOFENCEFACESDATAMODEL_H
#define GEOFENCEFACESDATAMODEL_H

#include <QAbstractTableModel>

struct GeoFenceFacesData{
    int faceID;
    int vertexA;
    int vertexB;
    int vertexC;
};

class QDomElement;

class GeoFenceFacesDataModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    enum GeoFenceFacesDataEnum {
        GEO_FACE_ID,
        GEO_VERTEX_A,
        GEO_VERTEX_B,
        GEO_VERTEX_C
    };

    explicit GeoFenceFacesDataModel(QObject *parent = 0);
    int rowCount(const QModelIndex &parent = QModelIndex()) const ;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;

    bool setData(const QModelIndex & index, const QVariant & value, int role = Qt::EditRole);
    Qt::ItemFlags flags(const QModelIndex & index) const ;
    bool insertRows ( int row, int count, const QModelIndex & parent = QModelIndex() );
    bool removeRows ( int row, int count, const QModelIndex & parent = QModelIndex() );

    void replaceModel(QVector<GeoFenceFacesData>);
    
signals:
    
public slots:

private:
    QVariant getColumnByIndex(const GeoFenceFacesData *row, const int index) const;
    bool setColumnByIndex(GeoFenceFacesData *row, const int index, const QVariant value);

    QList<GeoFenceFacesData*> dataStorage;
    int nextIndex;
};

#endif // GEOFENCEFACESDATAMODEL_H
