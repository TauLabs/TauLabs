/**
 ******************************************************************************
 * @file       flightdatamodel.cpp
 * @author     PhoenixPilot Project, http://github.com/PhoenixPilot Copyright (C) 2012.
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Path Planner Pluggin
 * @{
 * @brief Representation of a flight plan
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

#include "flightdatamodel.h"
#include <QFile>
#include <QDomDocument>
#include <QMessageBox>
#include <waypoint.h>

QMap<int,QString> FlightDataModel::modeNames = QMap<int, QString>();

//! Initialize an empty flight plan
FlightDataModel::FlightDataModel(QObject *parent):QAbstractTableModel(parent)
{
    // This could be auto populated from the waypoint object but nothing else in the
    // model depends on run time properties and we might want to exclude certain modes
    // being presented later (e.g. driving on a multirotor)
    modeNames.clear();
    modeNames.insert(Waypoint::MODE_CIRCLEPOSITIONLEFT, tr("Circle Position Left"));
    modeNames.insert(Waypoint::MODE_CIRCLEPOSITIONRIGHT, tr("Circle Position Right"));
    modeNames.insert(Waypoint::MODE_DRIVECIRCLELEFT, tr("Drive Circle Left"));
    modeNames.insert(Waypoint::MODE_DRIVECIRCLERIGHT, tr("Drive Circle Right"));
    modeNames.insert(Waypoint::MODE_DRIVEENDPOINT, tr("Drive Endpoint"));
    modeNames.insert(Waypoint::MODE_DRIVEVECTOR, tr("Drive Vector"));
    modeNames.insert(Waypoint::MODE_FLYCIRCLELEFT, tr("Fly Circle Left"));
    modeNames.insert(Waypoint::MODE_FLYCIRCLERIGHT, tr("Fly Circle Right"));
    modeNames.insert(Waypoint::MODE_FLYENDPOINT, tr("Fly Endpoint"));
    modeNames.insert(Waypoint::MODE_FLYVECTOR, tr("Fly Vector"));
    modeNames.insert(Waypoint::MODE_LAND, tr("Land"));
    modeNames.insert(Waypoint::MODE_STOP, tr("Stop"));
}

//! Return the number of waypoints
int FlightDataModel::rowCount(const QModelIndex &/*parent*/) const
{
    return dataStorage.length();
}

//! Return the number of fields in the model
int FlightDataModel::columnCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;
    return LASTCOLUMN-1;
}

/**
 * @brief FlightDataModel::data Fetch the data from the model
 * @param index Specifies the row and column to fetch
 * @param role Either use Qt::DisplayRole or Qt::EditRole
 * @return The data as a variant or QVariant::Invalid for a bad role
 */
QVariant FlightDataModel::data(const QModelIndex &index, int role) const
{
    if (role == Qt::DisplayRole||role==Qt::EditRole)
    {
        int rowNumber=index.row();
        int columnNumber=index.column();
        if(rowNumber>dataStorage.length()-1 || rowNumber<0)
            return QVariant::Invalid;
        pathPlanData * myRow=dataStorage.at(rowNumber);
        QVariant ret;
        if (index.column() == (int) FlightDataModel::MODE && role == Qt::DisplayRole) {
            int value = getColumnByIndex(myRow,columnNumber).toInt();
            ret = QVariant(modeNames.value(value));
        } else
            ret = getColumnByIndex(myRow,columnNumber);

        return ret;
    }
    else {
        return QVariant::Invalid;
    }
}

/**
 * @brief FlightDataModel::setColumnByIndex The data for a particular path plan entry
 * @param row Which waypoint representation to modify
 * @param index Which data type to modify (FlightDataModel::pathPlanDataEnum)
 * @param value The new value
 * @return True if succeeded, otherwise false
 */
bool FlightDataModel::setColumnByIndex(pathPlanData *row, const int index, const QVariant value)
{
    switch(index)
    {
    case WPDESCRITPTION:
        row->wpDescritption=value.toString();
        return true;
        break;
    case LATPOSITION:
        row->latPosition=value.toDouble();
        return true;
        break;
    case LNGPOSITION:
        row->lngPosition=value.toDouble();
        return true;
        break;
    case ALTITUDE:
        row->altitude=value.toDouble();
        return true;
        break;
    case VELOCITY:
        row->velocity=value.toFloat();
        return true;
        break;
    case MODE:
        row->mode=value.toInt();
        return true;
        break;
    case MODE_PARAMS:
        row->mode_params=value.toFloat();
        return true;
        break;
    default:
        return false;
    }
    return false;
}

/**
 * @brief FlightDataModel::getColumnByIndex Get the data from a particular column
 * @param row The pathPlanData structure to use
 * @param index Which column (FlightDataModel::pathPlanDataEnum)
 * @return The data
 */
QVariant FlightDataModel::getColumnByIndex(const pathPlanData *row,const int index) const
{
    switch(index)
    {
    case WPDESCRITPTION:
        return row->wpDescritption;
        break;
    case LATPOSITION:
        return row->latPosition;
        break;
    case LNGPOSITION:
        return row->lngPosition;
        break;
    case ALTITUDE:
        return row->altitude;
        break;
    case VELOCITY:
        return row->velocity;
        break;
    case MODE:
        return row->mode;
        break;
    case MODE_PARAMS:
        return row->mode_params;
        break;
    }
}

/**
 * @brief FlightDataModel::headerData Get the names of the columns
 * @param section
 * @param orientation
 * @param role
 * @return
 */
QVariant FlightDataModel::headerData(int section, Qt::Orientation orientation, int role) const
 {
     if (role == Qt::DisplayRole)
     {
         if(orientation==Qt::Vertical)
         {
             return QString::number(section+1);
         }
         else if (orientation == Qt::Horizontal) {
             switch (section)
             {
             case WPDESCRITPTION:
                 return QString("Description");
                 break;
             case LATPOSITION:
                 return QString("Latitude");
                 break;
             case LNGPOSITION:
                 return QString("Longitude");
                 break;
             case ALTITUDE:
                 return QString("Altitude");
                 break;
             case VELOCITY:
                 return QString("Velocity");
                 break;
             case MODE:
                 return QString("Mode");
                 break;
             case MODE_PARAMS:
                 return QString("Mode parameters");
                 break;
             default:
                 return QString();
                 break;
             }
         }
     }
     else
       return QAbstractTableModel::headerData(section, orientation, role);
}

/**
 * @brief FlightDataModel::setData Set the data at a given location
 * @param index Specifies both the row (waypoint) and column (field0
 * @param value The new value
 * @param role Used by the Qt MVC to determine what to do.  Should be Qt::EditRole
 * @return  True if setting data succeeded, otherwise false
 */
bool FlightDataModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (index.isValid() && role == Qt::EditRole)
    {
        int columnIndex = index.column();
        int rowIndex = index.row();

        pathPlanData *myRow = dataStorage.at(rowIndex);
        setColumnByIndex(myRow,columnIndex,value);
        emit dataChanged(index,index);

        return true;
    }
    return false;
}

/**
 * @brief FlightDataModel::flags Tell QT MVC which flags are supported for items
 * @return That the item is selectable, editable and enabled
 */
Qt::ItemFlags FlightDataModel::flags(const QModelIndex & /*index*/) const
{
    return Qt::ItemIsSelectable |  Qt::ItemIsEditable | Qt::ItemIsEnabled ;
}

/**
 * @brief FlightDataModel::insertRows Create a new waypoint
 * @param row The new waypoint id
 * @param count How many to add
 * @return
 */
bool FlightDataModel::insertRows(int row, int count, const QModelIndex &/*parent*/)
{
    pathPlanData * data;
    beginInsertRows(QModelIndex(),row,row+count-1);
    for(int x=0; x<count;++x)
    {
        // Initialize new internal representation
        data=new pathPlanData;
        data->latPosition=0;
        data->lngPosition=0;

        // If there is a previous waypoint, initialize some of the fields to that value
        if(rowCount() > 0)
        {
            data->altitude=this->data(this->index(rowCount()-1,ALTITUDE)).toDouble();
            data->velocity=this->data(this->index(rowCount()-1,VELOCITY)).toFloat();
            data->mode=this->data(this->index(rowCount()-1,MODE)).toInt();
            data->mode_params=this->data(this->index(rowCount()-1,MODE_PARAMS)).toFloat();
        } else {
            data->altitude=0;
            data->velocity=0;
            data->mode=1;
            data->mode_params=0;
        }
        dataStorage.insert(row,data);
    }

    endInsertRows();
}

/**
 * @brief FlightDataModel::removeRows Remove waypoints from the model
 * @param row The starting waypoint
 * @param count How many to remove
 * @return True if succeeded, otherwise false
 */
bool FlightDataModel::removeRows(int row, int count, const QModelIndex &/*parent*/)
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
}

/**
 * @brief FlightDataModel::writeToFile Write the waypoints to an xml file
 * @param fileName The filename to write to
 * @return
 */
bool FlightDataModel::writeToFile(QString fileName)
{

    QFile file(fileName);

    if (!file.open(QIODevice::WriteOnly)) {
        QMessageBox::information(NULL, tr("Unable to open file"), file.errorString());
        return false;
    }
    QDataStream out(&file);
    QDomDocument doc("PathPlan");
    QDomElement root = doc.createElement("waypoints");
    doc.appendChild(root);

    foreach(pathPlanData * obj,dataStorage)
    {

        QDomElement waypoint = doc.createElement("waypoint");
        waypoint.setAttribute("number",dataStorage.indexOf(obj));
        root.appendChild(waypoint);
        QDomElement field=doc.createElement("field");
        field.setAttribute("value",obj->wpDescritption);
        field.setAttribute("name","description");
        waypoint.appendChild(field);

        field=doc.createElement("field");
        field.setAttribute("value",obj->latPosition);
        field.setAttribute("name","latitude");
        waypoint.appendChild(field);

        field=doc.createElement("field");
        field.setAttribute("value",obj->lngPosition);
        field.setAttribute("name","longitude");
        waypoint.appendChild(field);

        field=doc.createElement("field");
        field.setAttribute("value",obj->altitude);
        field.setAttribute("name","altitude");
        waypoint.appendChild(field);

        field=doc.createElement("field");
        field.setAttribute("value",obj->velocity);
        field.setAttribute("name","velocity");
        waypoint.appendChild(field);

        field=doc.createElement("field");
        field.setAttribute("value",obj->mode);
        field.setAttribute("name","mode");
        waypoint.appendChild(field);

        field=doc.createElement("field");
        field.setAttribute("value",obj->mode_params);
        field.setAttribute("name","mode_params");
        waypoint.appendChild(field);
    }
    file.write(doc.toString().toAscii());
    file.close();
    return true;
}

/**
 * @brief FlightDataModel::readFromFile Read into the model from a flight plan xml file
 * @param fileName The filename to parse
 */
void FlightDataModel::readFromFile(QString fileName)
{
    removeRows(0,rowCount());
    QFile file(fileName);
    file.open(QIODevice::ReadOnly);
    QDomDocument doc("PathPlan");
    QByteArray array=file.readAll();
    QString error;
    if (!doc.setContent(array,&error)) {
        QMessageBox msgBox;
        msgBox.setText(tr("File Parsing Failed."));
        msgBox.setInformativeText(QString(tr("This file is not a correct XML file:%0")).arg(error));
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
        return;
    }
    file.close();

    QDomElement root = doc.documentElement();

    if (root.isNull() || (root.tagName() != "waypoints")) {
        QMessageBox msgBox;
        msgBox.setText(tr("Wrong file contents"));
        msgBox.setInformativeText(tr("This file does not contain correct UAVSettings"));
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
        return;
    }

    pathPlanData * data=NULL;
    QDomNode node = root.firstChild();
    while (!node.isNull()) {
        QDomElement e = node.toElement();
        if (e.tagName() == "waypoint") {
            QDomNode fieldNode=e.firstChild();
            data=new pathPlanData;
            while (!fieldNode.isNull()) {
                QDomElement field = fieldNode.toElement();
                if (field.tagName() == "field") {
                    if(field.attribute("name")=="altitude")
                        data->altitude=field.attribute("value").toDouble();
                    else if(field.attribute("name")=="description")
                        data->wpDescritption=field.attribute("value");
                    else if(field.attribute("name")=="latitude")
                        data->latPosition=field.attribute("value").toDouble();
                    else if(field.attribute("name")=="longitude")
                        data->lngPosition=field.attribute("value").toDouble();
                    else if(field.attribute("name")=="altitude")
                        data->altitude=field.attribute("value").toDouble();
                    else if(field.attribute("name")=="velocity")
                        data->velocity=field.attribute("value").toFloat();
                    else if(field.attribute("name")=="mode")
                        data->mode=field.attribute("value").toInt();
                    else if(field.attribute("name")=="mode_params")
                        data->mode_params=field.attribute("value").toFloat();
                }
                fieldNode=fieldNode.nextSibling();
            }
        beginInsertRows(QModelIndex(),dataStorage.length(),dataStorage.length());
        dataStorage.append(data);
        endInsertRows();
        }
        node=node.nextSibling();
    }
}

