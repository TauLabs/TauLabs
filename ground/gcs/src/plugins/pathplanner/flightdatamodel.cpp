/**
 ******************************************************************************
 * @file       flightdatamodel.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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
#include "utils/coordinateconversions.h"
#include "homelocation.h"
#include <QFile>
#include <QDomDocument>
#include <QMessageBox>
#include <waypoint.h>
#include "extensionsystem/pluginmanager.h"
#include "../plugins/uavobjects/uavobjectmanager.h"
#include "../plugins/uavobjects/uavobject.h"

QMap<int,QString> FlightDataModel::modeNames = QMap<int, QString>();

//! Initialize an empty flight plan
FlightDataModel::FlightDataModel(QObject *parent) : QAbstractTableModel(parent)
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
    return LASTCOLUMN;
}

/**
 * @brief FlightDataModel::data Fetch the data from the model
 * @param index Specifies the row and column to fetch
 * @param role Either use Qt::DisplayRole or Qt::EditRole
 * @return The data as a variant or QVariant::Invalid for a bad role
 */
QVariant FlightDataModel::data(const QModelIndex &index, int role) const
{
    if (role == Qt::DisplayRole || role==Qt::EditRole || role==Qt::UserRole)
    {

        if(!index.isValid() || index.row() > dataStorage.length()-1)
            return QVariant::Invalid;

        PathPlanData * row=dataStorage.at(index.row());

        // For the case of mode we want the model to normally return the string value
        // associated with that enum for display purposes.  However in the case of
        // Qt::UserRole this should fall through and return the numerical value of
        // the enum
        if (index.column() == (int) FlightDataModel::MODE && role == Qt::DisplayRole) {
            return modeNames.value(row->mode);
        }

        struct FlightDataModel::NED NED;
        switch(index.column())
        {
        case WPDESCRIPTION:
            return row->wpDescription;
        case LATPOSITION:
            return row->latPosition;
        case LNGPOSITION:
            return row->lngPosition;
        case ALTITUDE:
            return row->altitude;
        case NED_NORTH:
            NED = getNED(index.row());
            return NED.North;
        case NED_EAST:
            NED = getNED(index.row());
            return NED.East;
        case NED_DOWN:
            NED = getNED(index.row());
            return NED.Down;
        case VELOCITY:
            return row->velocity;
        case MODE:
            return row->mode;
        case MODE_PARAMS:
            return row->mode_params;
        case LOCKED:
            return row->locked;
        }
    }

    return QVariant::Invalid;
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
             case WPDESCRIPTION:
                 return QString("Description");
             case LATPOSITION:
                 return QString("Latitude");
             case LNGPOSITION:
                 return QString("Longitude");
             case ALTITUDE:
                 return QString("Altitude");
             case NED_NORTH:
                 return QString("Relative North");
             case NED_EAST:
                 return QString("Relative East");
             case NED_DOWN:
                 return QString("Relative Down");
             case VELOCITY:
                 return QString("Velocity");
             case MODE:
                 return QString("Mode");
             case MODE_PARAMS:
                 return QString("Mode parameters");
             case LOCKED:
                 return QString("Locked");
             default:
                 return QVariant::Invalid;
             }
         }
     }

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
        PathPlanData *row = dataStorage.at(index.row());

        // Do not allow changing any values except locked when the column is locked
        if (row->locked && index.column() != (int) FlightDataModel::LOCKED)
            return false;

        struct FlightDataModel::NED NED;
        QModelIndex otherIndex;
        switch(index.column())
        {
        case WPDESCRIPTION:
            row->wpDescription=value.toString();
            break;
        case LATPOSITION:
            row->latPosition=value.toDouble();
            // Indicate this also changed the north
            otherIndex = this->index(index.row(), FlightDataModel::NED_NORTH);
            emit dataChanged(otherIndex,otherIndex);
            break;
        case LNGPOSITION:
            row->lngPosition=value.toDouble();
            // Indicate this also changed the east
            otherIndex = this->index(index.row(), FlightDataModel::NED_EAST);
            emit dataChanged(otherIndex,otherIndex);
            break;
        case ALTITUDE:
            row->altitude=value.toDouble();
            // Indicate this also changed the NED down
            otherIndex = this->index(index.row(), FlightDataModel::NED_DOWN);
            emit dataChanged(otherIndex,otherIndex);
            break;
        case NED_NORTH:
            NED = getNED(index.row());
            NED.North = value.toDouble();
            setNED(index.row(), NED);
            // Indicate this also changed the latitude
            otherIndex = this->index(index.row(), FlightDataModel::LATPOSITION);
            emit dataChanged(otherIndex,otherIndex);
            break;
        case NED_EAST:
            NED = getNED(index.row());
            NED.East = value.toDouble();
            setNED(index.row(), NED);
            // Indicate this also changed the longitude
            otherIndex = this->index(index.row(), FlightDataModel::LNGPOSITION);
            emit dataChanged(otherIndex,otherIndex);
            break;
        case NED_DOWN:
            NED = getNED(index.row());
            NED.Down = value.toDouble();
            setNED(index.row(), NED);
            // Indicate this also changed the altitude
            otherIndex = this->index(index.row(), FlightDataModel::ALTITUDE);
            emit dataChanged(otherIndex,otherIndex);
            break;
        case VELOCITY:
            row->velocity=value.toFloat();
            break;
        case MODE:
            row->mode=value.toInt();
            break;
        case MODE_PARAMS:
            row->mode_params=value.toFloat();
            break;
        case LOCKED:
            row->locked = value.toBool();
            break;
        default:
            return false;
        }

        emit dataChanged(index,index);
        return true;
    }
    return false;
}

/**
 * @brief FlightDataModel::flags Tell QT MVC which flags are supported for items
 * @return That the item is selectable, editable and enabled
 */
Qt::ItemFlags FlightDataModel::flags(const QModelIndex & index) const
{
    // Locked is always editable
    if (index.column() == (int) FlightDataModel::LOCKED)
        return Qt::ItemIsSelectable |  Qt::ItemIsEditable | Qt::ItemIsEnabled;

    // Suppress editable flag if row is locked
    PathPlanData *row = dataStorage.at(index.row());
    if (row->locked)
        return Qt::ItemIsSelectable | Qt::ItemIsEnabled;

    return Qt::ItemIsSelectable |  Qt::ItemIsEditable | Qt::ItemIsEnabled;

}

/**
 * @brief FlightDataModel::insertRows Create a new waypoint
 * @param row The new waypoint id
 * @param count How many to add
 * @return
 */
bool FlightDataModel::insertRows(int row, int count, const QModelIndex &/*parent*/)
{
    PathPlanData * data;
    beginInsertRows(QModelIndex(),row,row+count-1);
    for(int x=0; x<count;++x)
    {
        // Initialize new internal representation
        data=new PathPlanData;
        data->latPosition=0;
        data->lngPosition=0;

        // If there is a previous waypoint, initialize some of the fields to that value
        if(rowCount() > 0)
        {
            PathPlanData * prevRow = dataStorage.at(rowCount()-1);
            data->altitude    = prevRow->altitude;
            data->velocity    = prevRow->velocity;
            data->mode        = prevRow->mode;
            data->mode_params = prevRow->mode_params;
            data->locked      = prevRow->locked;
        } else {
            data->altitude    = 0;
            data->velocity    = 0;
            data->mode        = Waypoint::MODE_FLYVECTOR;
            data->mode_params = 0;
            data->locked      = false;
        }
        dataStorage.insert(row,data);
    }

    endInsertRows();

    return true;
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

    return true;
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

    foreach(PathPlanData * obj,dataStorage)
    {

        QDomElement waypoint = doc.createElement("waypoint");
        waypoint.setAttribute("number",dataStorage.indexOf(obj));
        root.appendChild(waypoint);
        QDomElement field=doc.createElement("field");
        field.setAttribute("value",obj->wpDescription);
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

        field=doc.createElement("field");
        field.setAttribute("value",obj->locked);
        field.setAttribute("name","is_locked");
        waypoint.appendChild(field);
    }
    file.write(doc.toString().toLatin1());
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

    PathPlanData * data=NULL;
    QDomNode node = root.firstChild();
    while (!node.isNull()) {
        QDomElement e = node.toElement();
        if (e.tagName() == "waypoint") {
            QDomNode fieldNode=e.firstChild();
            data=new PathPlanData;
            while (!fieldNode.isNull()) {
                QDomElement field = fieldNode.toElement();
                if (field.tagName() == "field") {
                    if(field.attribute("name")=="altitude")
                        data->altitude=field.attribute("value").toDouble();
                    else if(field.attribute("name")=="description")
                        data->wpDescription=field.attribute("value");
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
                    else if(field.attribute("name")=="is_locked")
                        data->locked=field.attribute("value").toInt();
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


/**
 * @brief ModelUavoProxy::getHomeLocation Take care of scaling the home location UAVO to
 * degrees (lat lon) and meters altitude
 * @param [out] home A 3 element double array to store resul in
 * @return True if successful, false otherwise
 */
bool FlightDataModel::getHomeLocation(double *homeLLA) const
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager * objMngr = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objMngr);

    HomeLocation *home = HomeLocation::GetInstance(objMngr);
    if (home == NULL)
        return false;

    HomeLocation::DataFields homeLocation = home->getData();
    homeLLA[0] = homeLocation.Latitude / 1e7;
    homeLLA[1] = homeLocation.Longitude / 1e7;
    homeLLA[2] = homeLocation.Altitude;

    return true;
}

/**
 * @brief FlightDataModel::getNED Get hte NEW representation of a waypoint
 * @param row Which waypoint to get
 * @return The NED structure
 */
struct FlightDataModel::NED FlightDataModel::getNED(int index) const
{
    double f_NED[3];
    double homeLLA[3];
    PathPlanData * row = dataStorage.at(index);
    double LLA[3] = {row->latPosition, row->lngPosition, row->altitude};

    getHomeLocation(homeLLA);
    Utils::CoordinateConversions().LLA2NED_HomeLLA(LLA, homeLLA, f_NED);

    struct NED NED;
    NED.North = f_NED[0];
    NED.East = f_NED[1];
    NED.Down = f_NED[2];

    return NED;
}

/**
 * @brief FlightDataModel::setNED Set a waypoint by the NED representation
 * @param row Which waypoint to set
 * @param NED The NED structure
 * @return True if successful
 */
bool FlightDataModel::setNED(int index, struct FlightDataModel::NED NED)
{
    double homeLLA[3];
    double LLA[3];
    PathPlanData * row = dataStorage.at(index);
    double f_NED[3] = {NED.North, NED.East, NED.Down};

    getHomeLocation(homeLLA);
    Utils::CoordinateConversions().NED2LLA_HomeLLA(homeLLA, f_NED, LLA);

    row->latPosition = LLA[0];
    row->lngPosition = LLA[1];
    row->altitude = LLA[2];

    return true;
}

/**
 * @brief FlightDataModel::replaceData with data from a new model
 * @param newModel the new data to use
 * @return true if successful
 */
bool FlightDataModel::replaceData(FlightDataModel *newModel)
{
    // Delete existing data
    removeRows(0,rowCount());

    for (int i = 0; i < newModel->rowCount(); i++) {
        insertRow(i);
        for (int j = 0; j < newModel->columnCount(); j++) {
            // Use Qt::UserRole to make sure the mode is fetched numerically
            setData(index(i,j), newModel->data(newModel->index(i, j), Qt::UserRole));
        }
    }

    return true;
}
