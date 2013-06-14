/**
 ******************************************************************************
 *
 * @file       geofenceverticesdatamodel.cpp
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
#include "geofenceverticesdatamodel.h"

GeoFenceVerticesDataModel::GeoFenceVerticesDataModel(QObject *parent) :
    QAbstractTableModel(parent),
    nextIndex(0)
{
}

int GeoFenceVerticesDataModel::rowCount(const QModelIndex &/*parent*/) const
{
    return dataStorage.length();
}

int GeoFenceVerticesDataModel::columnCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;
    return 4; //FixMe: Ugly to hardcode this like that.
}

QVariant GeoFenceVerticesDataModel::data(const QModelIndex &index, int role) const
{
    if (role == Qt::DisplayRole||role==Qt::EditRole)
    {
        int rowNumber=index.row();
        int columnNumber=index.column();
        if(rowNumber>dataStorage.length()-1 || rowNumber<0)
            return QVariant::Invalid;
        GeoFenceVerticesData * myRow=dataStorage.at(rowNumber);
        QVariant ret=getColumnByIndex(myRow,columnNumber);
        return ret;
    }
    else {
        return QVariant::Invalid;
    }
}

QVariant GeoFenceVerticesDataModel::headerData(int section, Qt::Orientation orientation, int role) const
{
     if (role == Qt::DisplayRole) {
         switch (orientation) {
         case Qt::Vertical:
             return QString::number(section+1);
             break;
         case Qt::Horizontal:
             switch (section) {
             case GEO_LATITUDE:
                 return QString(tr("Latitude"));
                 break;
             case GEO_LONGITUDE:
                 return QString(tr("Longitude"));
                 break;
             case GEO_ALTITUDE:
                 return QString(tr("Altitude"));
                 break;
             case GEO_VERTEX_ID:
                 return QString(tr("Vertex ID"));
                 break;
             default:
                 return QString();
                 break;
             }
         }
     }
     else {
         return QAbstractTableModel::headerData(section, orientation, role);
     }
}

bool GeoFenceVerticesDataModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (role == Qt::EditRole)
    {
        int columnIndex = index.column();
        int rowIndex = index.row();
        if(rowIndex > dataStorage.length()-1)
            return false;
        GeoFenceVerticesData *myRow = dataStorage.at(rowIndex);
        setColumnByIndex(myRow, columnIndex, value);
        emit dataChanged(index, index);
    }
    return true;
}

Qt::ItemFlags GeoFenceVerticesDataModel::flags(const QModelIndex & /*index*/) const
 {
    return Qt::ItemIsSelectable |  Qt::ItemIsEditable | Qt::ItemIsEnabled ;
}

bool GeoFenceVerticesDataModel::insertRows(int row, int count, const QModelIndex &/*parent*/)
{
    GeoFenceVerticesData * data;
    beginInsertRows(QModelIndex(),row,row+count-1);
    for(int x=0; x<count;++x)
    {
        data=new GeoFenceVerticesData;
        data->latitude=0;
        data->longitude=0;
        data->altitude=0;
        data->vertexId = 0;

        if(rowCount()>0)
        {
            data->latitude=this->data(this->index(rowCount()-1, GEO_LATITUDE)).toDouble();
            data->longitude=this->data(this->index(rowCount()-1, GEO_LONGITUDE)).toDouble();
            data->altitude=this->data(this->index(rowCount()-1, GEO_ALTITUDE)).toDouble();
            data->vertexId = nextIndex++;
        }
        dataStorage.insert(row,data);
    }
    endInsertRows();

    return true;
}

bool GeoFenceVerticesDataModel::removeRows(int row, int count, const QModelIndex &/*parent*/)
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

bool GeoFenceVerticesDataModel::writeToFile(QString fileName)
{

    QFile file(fileName);

    if (!file.open(QIODevice::WriteOnly)) {
        QMessageBox::information(NULL, tr("Unable to open file"), file.errorString());
        return false;
    }
    QDataStream out(&file);
    QDomDocument doc("Geofence");
    QDomElement root = doc.createElement("dataModel");
    doc.appendChild(root);
    QDomElement vertices = doc.createElement("vertices");
    root.appendChild(vertices);

    foreach(GeoFenceVerticesData * obj,dataStorage)
    {
        QDomElement geofence = doc.createElement("vertex");
        geofence.setAttribute("number",dataStorage.indexOf(obj));
        vertices.appendChild(geofence);
        QDomElement field=doc.createElement("field");
        field.setAttribute("value", obj->latitude);
        field.setAttribute("name","latitude");
        geofence.appendChild(field);

        field=doc.createElement("field");
        field.setAttribute("value", obj->longitude);
        field.setAttribute("name","longitude");
        geofence.appendChild(field);

        field=doc.createElement("field");
        field.setAttribute("value", obj->altitude);
        field.setAttribute("name","altitude");
        geofence.appendChild(field);

        field=doc.createElement("field");
        field.setAttribute("value", obj->vertexId);
        field.setAttribute("name","vertexId");
        geofence.appendChild(field);
    }

    // End file write
    file.write(doc.toString().toAscii());
    file.close();

    return true;
}

void GeoFenceVerticesDataModel::replaceModel(QVector<GeoFenceVerticesData> importedVertices)
{
    removeRows(0,rowCount());

    foreach(GeoFenceVerticesData data, importedVertices) {
        // Create a local copy. This is necessary because the data model handles deleting the local data
        GeoFenceVerticesData *data_ptr = new GeoFenceVerticesData;
        data_ptr->latitude  = data.latitude;
        data_ptr->longitude = data.longitude;
        data_ptr->altitude  = data.altitude;
        data_ptr->vertexId  = data.vertexId;

        beginInsertRows(QModelIndex(), dataStorage.length(), dataStorage.length());
        dataStorage.append(data_ptr);
        endInsertRows();
    }
}

void GeoFenceVerticesDataModel::readFromFile(QDomElement root)
{
    removeRows(0,rowCount());

    GeoFenceVerticesData * data=NULL;
    QDomNode node = root.firstChild();
    QDomElement modelSection = node.toElement();

    // Iterate over list until "vertices" is found.
    while(!node.isNull() && modelSection.tagName() != "vertices") {
        node=node.nextSibling();
        modelSection = node.toElement();
    }

    if(modelSection.tagName() == "vertices") {
        node = node.firstChild();
        while (!node.isNull()) {
            modelSection = node.toElement();
            QDomElement e = modelSection.toElement();
            if (e.tagName() == "vertex") {
                QDomNode fieldNode = e.firstChild();
                data = new GeoFenceVerticesData;
                while (!fieldNode.isNull()) {
                    QDomElement field = fieldNode.toElement();
                    if (field.tagName() == "field") {
                        if(field.attribute("name") == "latitude")
                            data->latitude = field.attribute("value").toDouble();
                        else if(field.attribute("name") == "longitude")
                            data->longitude = field.attribute("value").toDouble();
                        else if(field.attribute("name") == "altitude")
                            data->altitude = field.attribute("value").toDouble();
                        else if(field.attribute("name") == "vertexId")
                            data->vertexId=field.attribute("value").toInt();
                    }
                    fieldNode=fieldNode.nextSibling();
                }

                beginInsertRows(QModelIndex(), dataStorage.length(), dataStorage.length());
                dataStorage.append(data);
                endInsertRows();
            }

            node = node.nextSibling();
        }
    }
}

bool GeoFenceVerticesDataModel::setColumnByIndex(GeoFenceVerticesData  *row, const int index, const QVariant value)
{
    bool retVal = false;
    switch(index)
    {
    case GEO_LATITUDE:
        row->latitude = value.toDouble();
        retVal = true;
        break;
    case GEO_LONGITUDE:
        row->longitude = value.toDouble();
        retVal = true;
        break;
    case GEO_ALTITUDE:
        row->altitude = value.toDouble();
        retVal = true;
        break;
    case GEO_VERTEX_ID:
        row->vertexId = value.toInt();
        retVal = true;
        break;
    }
    return retVal;
}
QVariant GeoFenceVerticesDataModel::getColumnByIndex(const GeoFenceVerticesData *row, const int index) const
{
    switch(index)
    {
    case GEO_LATITUDE:
        return row->latitude;
        break;
    case GEO_LONGITUDE:
        return row->longitude;
        break;
    case GEO_ALTITUDE:
        return row->altitude;
        break;
    case GEO_VERTEX_ID:
        return row->vertexId;
        break;
    }
}
