/**
 ******************************************************************************
 *
 * @file       geofencefacesdatamodel.cpp
 * @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
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

#include <QFile>
#include <QMessageBox>
#include <QDomDocument>
#include "geofencefacesdatamodel.h"

GeoFenceFacesDataModel::GeoFenceFacesDataModel(QObject *parent) :
    QAbstractTableModel(parent),
    nextIndex(0)
{
}

int GeoFenceFacesDataModel::rowCount(const QModelIndex &/*parent*/) const
{
    return dataStorage.length();
}

int GeoFenceFacesDataModel::columnCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;
    return 4; //FixMe: Ugly to hardcode this like that.
}

QVariant GeoFenceFacesDataModel::data(const QModelIndex &index, int role) const
{
    if (role == Qt::DisplayRole||role==Qt::EditRole)
    {
        int rowNumber=index.row();
        int columnNumber=index.column();
        if(rowNumber>dataStorage.length()-1 || rowNumber<0)
            return QVariant::Invalid;
        GeoFenceFacesData * myRow=dataStorage.at(rowNumber);
        QVariant ret=getColumnByIndex(myRow,columnNumber);
        return ret;
    }
    else {
        return QVariant::Invalid;
    }
}

QVariant GeoFenceFacesDataModel::headerData(int section, Qt::Orientation orientation, int role) const
{
     if (role == Qt::DisplayRole) {
         switch (orientation) {
         case Qt::Vertical:
             return QString::number(section+1);
             break;
         case Qt::Horizontal:
             switch (section) {
             case GEO_FACE_ID:
                 return QString(tr("Face ID"));
                 break;
             case GEO_VERTEX_A:
                 return QString(tr("Vertex A"));
                 break;
             case GEO_VERTEX_B:
                 return QString(tr("Vertex B"));
                 break;
             case GEO_VERTEX_C:
                 return QString(tr("Vertex C"));
                 break;
             default:
                 return QString();
                 break;
             }
             break;
         }
     }
     else {
         return QAbstractTableModel::headerData(section, orientation, role);
     }
}

bool GeoFenceFacesDataModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (role == Qt::EditRole)
    {
        int columnIndex = index.column();
        int rowIndex = index.row();
        if(rowIndex > dataStorage.length()-1)
            return false;
        GeoFenceFacesData *myRow = dataStorage.at(rowIndex);
        setColumnByIndex(myRow, columnIndex, value);
        emit dataChanged(index, index);
    }
    return true;
}

Qt::ItemFlags GeoFenceFacesDataModel::flags(const QModelIndex & /*index*/) const
 {
    return Qt::ItemIsSelectable |  Qt::ItemIsEditable | Qt::ItemIsEnabled ;
}

bool GeoFenceFacesDataModel::insertRows(int row, int count, const QModelIndex &/*parent*/)
{
    GeoFenceFacesData * data;
    beginInsertRows(QModelIndex(),row,row+count-1);
    for(int x=0; x<count; ++x)
    {
        data=new GeoFenceFacesData;
        data->faceID=0;
        data->vertexA=0;
        data->vertexB=0;
        data->vertexC = 0;

        if(rowCount()>0)
        {
            data->faceID=this->data(this->index(rowCount()-1, GEO_FACE_ID)).toUInt(); // Fixme: Should this be faceID++ ?
            data->vertexA=this->data(this->index(rowCount()-1, GEO_VERTEX_A)).toUInt();
            data->vertexB=this->data(this->index(rowCount()-1, GEO_VERTEX_B)).toUInt();
            data->vertexC=this->data(this->index(rowCount()-1, GEO_VERTEX_C)).toUInt();
        }
        dataStorage.insert(row,data);
    }
    endInsertRows();

    return true;
}

bool GeoFenceFacesDataModel::removeRows(int row, int count, const QModelIndex &/*parent*/)
{
    if(row<0)
        return false;
    beginRemoveRows(QModelIndex(),row,row+count-1);
    for(int x=0; x<count;++x)
    {
        delete dataStorage.at(row);
        dataStorage.removeAt(row);
    }
    endRemoveRows();

    return true;
}


void GeoFenceFacesDataModel::replaceModel(QVector<GeoFenceFacesData> importedFaces)
{
    removeRows(0,rowCount());

    foreach(GeoFenceFacesData data, importedFaces) {
        // Create a local copy. This is necessary because the data model handles deleting the local data
        GeoFenceFacesData *data_ptr = new GeoFenceFacesData;
        data_ptr->faceID  = data.faceID;
        data_ptr->vertexA = data.vertexA;
        data_ptr->vertexB = data.vertexB;
        data_ptr->vertexC = data.vertexC;

        beginInsertRows(QModelIndex(), dataStorage.length(), dataStorage.length());
        dataStorage.append(data_ptr);
        endInsertRows();
    }
}


bool GeoFenceFacesDataModel::setColumnByIndex(GeoFenceFacesData  *row, const int index, const QVariant value)
{
    bool retVal = false;
    switch(index)
    {
    case GEO_FACE_ID:
        row->faceID = value.toUInt();
        retVal = true;
        break;
    case GEO_VERTEX_A:
        row->vertexA = value.toUInt();
        retVal = true;
        break;
    case GEO_VERTEX_B:
        row->vertexB = value.toUInt();
        retVal = true;
        break;
    case GEO_VERTEX_C:
        row->vertexC = value.toUInt();
        retVal = true;
        break;
    }
    return retVal;
}

QVariant GeoFenceFacesDataModel::getColumnByIndex(const GeoFenceFacesData *row, const int index) const
{
    switch(index)
    {
    case GEO_FACE_ID:
        return row->faceID;
        break;
    case GEO_VERTEX_A:
        return row->vertexA;
        break;
    case GEO_VERTEX_B:
        return row->vertexB;
        break;
    case GEO_VERTEX_C:
        return row->vertexC;
        break;
    }
}
