/**
 ******************************************************************************
 * @file       modelmapproxy.cpp
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
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

#include "modelmapproxy.h"
#include "../pathplanner/waypointdialog.h"

ModelMapProxy::ModelMapProxy(QObject *parent,OPMapWidget *map, FlightDataModel *model,QItemSelectionModel * selectionModel):QObject(parent),myMap(map),model(model),selection(selectionModel)
{
    connect(model,SIGNAL(rowsInserted(const QModelIndex&,int,int)),this,SLOT(rowsInserted(const QModelIndex&,int,int)));
    connect(model,SIGNAL(rowsRemoved(const QModelIndex&,int,int)),this,SLOT(rowsRemoved(const QModelIndex&,int,int)));
    connect(selection,SIGNAL(currentRowChanged(QModelIndex,QModelIndex)),this,SLOT(currentRowChanged(QModelIndex,QModelIndex)));
    connect(model,SIGNAL(dataChanged(QModelIndex,QModelIndex)),this,SLOT(dataChanged(QModelIndex,QModelIndex)));
    connect(myMap,SIGNAL(selectedWPChanged(QList<WayPointItem*>)),this,SLOT(selectedWPChanged(QList<WayPointItem*>)));
    connect(myMap,SIGNAL(WPManualCoordChange(WayPointItem*)),this,SLOT(WPValuesChanged(WayPointItem*)));
    connect(myMap,SIGNAL(WPNumberChanged(int,int,WayPointItem*)),this,SLOT(WPValuesChanged(WayPointItem*)));
}

/**
 * @brief ModelMapProxy::WPValuesChanged The UI changed a waypoint, update the model
 * @param wp The handle to the changed waypoint
 */
void ModelMapProxy::WPValuesChanged(WayPointItem * wp)
{
    QModelIndex index;
    index=model->index(wp->Number(),FlightDataModel::LATPOSITION);
    if(!index.isValid())
        return;
    model->setData(index,wp->Coord().Lat(),Qt::EditRole);
    index=model->index(wp->Number(),FlightDataModel::LNGPOSITION);
    model->setData(index,wp->Coord().Lng(),Qt::EditRole);

    index=model->index(wp->Number(),FlightDataModel::ALTITUDE);
    model->setData(index,wp->Altitude(),Qt::EditRole);

    index=model->index(wp->Number(),FlightDataModel::DISRELATIVE);
    model->setData(index,wp->getRelativeCoord().distance,Qt::EditRole);
    index=model->index(wp->Number(),FlightDataModel::BEARELATIVE);
    model->setData(index,wp->getRelativeCoord().bearingToDegrees(),Qt::EditRole);
    index=model->index(wp->Number(),FlightDataModel::ALTITUDERELATIVE);
    model->setData(index,wp->getRelativeCoord().altitudeRelative,Qt::EditRole);
}

/**
 * @brief ModelMapProxy::currentRowChanged When a row is changed, highlight the waypoint
 * @param current The selected row
 * @param previous Unused
 */
void ModelMapProxy::currentRowChanged(QModelIndex current, QModelIndex previous)
{
    Q_UNUSED(previous);

    QList<WayPointItem*> list;
    WayPointItem * wp=findWayPointNumber(current.row());
    if(!wp)
        return;
    list.append(wp);
    myMap->setSelectedWP(list);
}

/**
 * @brief ModelMapProxy::selectedWPChanged When a list of waypoints are changed, select them in model
 * @param list The list of changed waypoints
 */
void ModelMapProxy::selectedWPChanged(QList<WayPointItem *> list)
{
    selection->clearSelection();
    foreach(WayPointItem * wp,list)
    {
        QModelIndex index=model->index(wp->Number(),0);
        selection->setCurrentIndex(index,QItemSelectionModel::Select | QItemSelectionModel::Rows);
    }
}

/**
 * @brief ModelMapProxy::overlayTranslate Map from path types types to Overlay types
 * @param type The map delegate type which is like a Waypoint::Mode
 * @return
 */
ModelMapProxy::overlayType ModelMapProxy::overlayTranslate(int type)
{
    switch(type)
    {
    case WaypointDataDelegate::MODE_FLYENDPOINT:
    case WaypointDataDelegate::MODE_FLYVECTOR:
    case WaypointDataDelegate::MODE_DRIVEENDPOINT:
    case WaypointDataDelegate::MODE_DRIVEVECTOR:
        return OVERLAY_LINE;
        break;
    case WaypointDataDelegate::MODE_FLYCIRCLERIGHT:
    case WaypointDataDelegate::MODE_DRIVECIRCLERIGHT:
        return OVERLAY_CURVE_RIGHT;
        break;
    case WaypointDataDelegate::MODE_FLYCIRCLELEFT:
    case WaypointDataDelegate::MODE_DRIVECIRCLELEFT:
        return OVERLAY_CURVE_LEFT;
        break;
    default:
        break;
    }
}

/**
 * @brief ModelMapProxy::createOverlay Create a graphical path component
 * @param from The starting location
 * @param to The ending location (for circles the radius) which is a HomeItem
 * @param type The type of path component
 * @param color
 */
void ModelMapProxy::createOverlay(WayPointItem *from, WayPointItem *to,
                                  ModelMapProxy::overlayType type, QColor color,
                                  double radius=0)
{
    if(from==NULL || to==NULL || from==to)
        return;
    switch(type)
    {
    case OVERLAY_LINE:
        myMap->WPLineCreate(from,to,color);
        break;
    case OVERLAY_CIRCLE_RIGHT:
        myMap->WPCircleCreate(to,from,true,color);
        break;
    case OVERLAY_CIRCLE_LEFT:
        myMap->WPCircleCreate(to,from,false,color);
        break;
    case OVERLAY_CURVE_RIGHT:
        myMap->WPCurveCreate(to,from,radius,true,color);
        break;
    case OVERLAY_CURVE_LEFT:
        myMap->WPCurveCreate(to,from,radius,false,color);
        break;
    default:
        break;

    }
}

/**
 * @brief ModelMapProxy::createOverlay Create a graphical path component
 * @param from The starting location
 * @param to The ending location (for circles the radius) which is a HomeItem
 * @param type The type of path component
 * @param color
 */
void ModelMapProxy::createOverlay(WayPointItem *from, HomeItem *to, ModelMapProxy::overlayType type,QColor color)
{
    if(from==NULL || to==NULL)
        return;
    switch(type)
    {
    case OVERLAY_LINE:
        myMap->WPLineCreate(to,from,color);
        break;
    case OVERLAY_CIRCLE_RIGHT:
        myMap->WPCircleCreate(to,from,true,color);
        break;
    case OVERLAY_CIRCLE_LEFT:
        myMap->WPCircleCreate(to,from,false,color);
        break;
    default:
        break;

    }
}

/**
 * @brief ModelMapProxy::refreshOverlays Update the information from the model and
 * redraw all the components
 */
void ModelMapProxy::refreshOverlays()
{
    myMap->deleteAllOverlays();
    if(model->rowCount()<1)
        return;
    WayPointItem *wp_current = NULL;
    WayPointItem *wp_next = NULL;
    overlayType wp_next_overlay;

    // Get first waypoint type before stepping through path
    wp_current = findWayPointNumber(0);
    overlayType wp_current_overlay = overlayTranslate(model->data(model->index(0,FlightDataModel::MODE)).toInt());
    createOverlay(wp_current,myMap->Home,wp_current_overlay,Qt::green);

    for(int x=0;x<model->rowCount();++x)
    {
        wp_current = findWayPointNumber(x);

        wp_next_overlay = overlayTranslate(model->data(model->index(x+1,FlightDataModel::MODE)).toInt());

        wp_next = findWayPointNumber(x+1);
        createOverlay(wp_current, wp_next, wp_next_overlay, Qt::green,
                      model->data(model->index(x+1,FlightDataModel::MODE_PARAMS)).toFloat());
    }
}

/**
 * @brief ModelMapProxy::findWayPointNumber Return the graphial icon for the requested waypoint
 * @param number The waypoint number
 * @return The pointer to the graphical item or NULL
 */
WayPointItem *ModelMapProxy::findWayPointNumber(int number)
{
    if(number<0)
        return NULL;
    return myMap->WPFind(number);
}

/**
 * @brief ModelMapProxy::rowsRemoved Called whenever a row is removed from the model
 * @param parent Unused
 * @param first The first row removed
 * @param last The last row removed
 */
void ModelMapProxy::rowsRemoved(const QModelIndex &parent, int first, int last)
{
    Q_UNUSED(parent);

    for(int x=last;x>first-1;x--)
    {
        myMap->WPDelete(x);
    }
    refreshOverlays();
}

/**
 * @brief ModelMapProxy::dataChanged Update the display whenever the model information changes
 * @param topLeft The first waypoint and column changed
 * @param bottomRight The last waypoint and column changed
 */
void ModelMapProxy::dataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight)
{
    Q_UNUSED(bottomRight);

    // Abort if no corresponding graphical item
    WayPointItem *item = findWayPointNumber(topLeft.row());
    if(!item)
        return;

    internals::PointLatLng latlng;
    distBearingAltitude distBearing;
    double altitude;
    bool relative;
    QModelIndex index;
    QString desc;

    for (int x = topLeft.row(); x <= bottomRight.row(); x++) {
        for (int column = topLeft.column(); column <= bottomRight.column(); column++) {
            // Action depends on which columns were modified
            switch(column)
            {
            case FlightDataModel::MODE:
                refreshOverlays();
                break;
            case FlightDataModel::WPDESCRITPTION:
                index=model->index(x,FlightDataModel::WPDESCRITPTION);
                desc=index.data(Qt::DisplayRole).toString();
                item->SetDescription(desc);
                break;
            case FlightDataModel::LATPOSITION:
                latlng=item->Coord();
                index=model->index(x,FlightDataModel::LATPOSITION);
                latlng.SetLat(index.data(Qt::DisplayRole).toDouble());
                item->SetCoord(latlng);
                break;
            case FlightDataModel::LNGPOSITION:
                latlng=item->Coord();
                index=model->index(x,FlightDataModel::LNGPOSITION);
                latlng.SetLng(index.data(Qt::DisplayRole).toDouble());
                item->SetCoord(latlng);
                break;
            case FlightDataModel::BEARELATIVE:
                distBearing=item->getRelativeCoord();
                index=model->index(x,FlightDataModel::BEARELATIVE);
                distBearing.setBearingFromDegrees(index.data(Qt::DisplayRole).toDouble());
                item->setRelativeCoord(distBearing);
                break;
            case FlightDataModel::DISRELATIVE:
                distBearing=item->getRelativeCoord();
                index=model->index(x,FlightDataModel::DISRELATIVE);
                distBearing.distance=index.data(Qt::DisplayRole).toDouble();
                item->setRelativeCoord(distBearing);
                break;
            case FlightDataModel::ALTITUDERELATIVE:
                distBearing=item->getRelativeCoord();
                index=model->index(x,FlightDataModel::ALTITUDERELATIVE);
                distBearing.altitudeRelative=index.data(Qt::DisplayRole).toFloat();
                item->setRelativeCoord(distBearing);
                break;
            case FlightDataModel::ISRELATIVE:
                index=model->index(x,FlightDataModel::ISRELATIVE);
                relative=index.data(Qt::DisplayRole).toBool();
                if(relative)
                    item->setWPType(mapcontrol::WayPointItem::relative);
                else
                    item->setWPType(mapcontrol::WayPointItem::absolute);
                break;
            case FlightDataModel::ALTITUDE:
                index=model->index(x,FlightDataModel::ALTITUDE);
                altitude=index.data(Qt::DisplayRole).toDouble();
                item->SetAltitude(altitude);
                break;
            case FlightDataModel::MODE_PARAMS:
                // Make sure to update radius of arcs
                refreshOverlays();
                break;
            case FlightDataModel::LOCKED:
                index=model->index(x,FlightDataModel::LOCKED);
                item->setFlag(QGraphicsItem::ItemIsMovable,!index.data(Qt::DisplayRole).toBool());
                break;
            }
        }
    }
}

/**
 * @brief ModelMapProxy::rowsInserted When rows are inserted in the model add the corresponding graphical items
 * @param parent Unused
 * @param first The first row to update
 * @param last The last row to update
 */
void ModelMapProxy::rowsInserted(const QModelIndex &parent, int first, int last)
{
    Q_UNUSED(parent);
    for(int x=first; x<last+1; x++)
    {
        QModelIndex index;
        WayPointItem * item;
        internals::PointLatLng latlng;
        distBearingAltitude distBearing;
        double altitude;
        bool relative;
        index=model->index(x,FlightDataModel::WPDESCRITPTION);
        QString desc=index.data(Qt::DisplayRole).toString();
        index=model->index(x,FlightDataModel::LATPOSITION);
        latlng.SetLat(index.data(Qt::DisplayRole).toDouble());
        index=model->index(x,FlightDataModel::LNGPOSITION);
        latlng.SetLng(index.data(Qt::DisplayRole).toDouble());
        index=model->index(x,FlightDataModel::DISRELATIVE);
        distBearing.distance=index.data(Qt::DisplayRole).toDouble();
        index=model->index(x,FlightDataModel::BEARELATIVE);
        distBearing.setBearingFromDegrees(index.data(Qt::DisplayRole).toDouble());
        index=model->index(x,FlightDataModel::ALTITUDERELATIVE);
        distBearing.altitudeRelative=index.data(Qt::DisplayRole).toFloat();
        index=model->index(x,FlightDataModel::ISRELATIVE);
        relative=index.data(Qt::DisplayRole).toBool();
        index=model->index(x,FlightDataModel::ALTITUDE);
        altitude=index.data(Qt::DisplayRole).toDouble();
        if(relative)
            item=myMap->WPInsert(distBearing,desc,x);
        else
            item=myMap->WPInsert(latlng,altitude,desc,x);
    }
    refreshOverlays();
}

/**
 * @brief ModelMapProxy::deleteWayPoint When a waypoint is deleted graphically, delete from the model
 * @param number The waypoint which was deleted
 */
void ModelMapProxy::deleteWayPoint(int number)
{
    model->removeRow(number,QModelIndex());
}

/**
 * @brief ModelMapProxy::createWayPoint When a waypoint is created graphically, insert into the end of the model
 * @param coord The coordinate the waypoint was created
 */
void ModelMapProxy::createWayPoint(internals::PointLatLng coord)
{
    model->insertRow(model->rowCount(),QModelIndex());
    QModelIndex index=model->index(model->rowCount()-1,FlightDataModel::LATPOSITION,QModelIndex());
    model->setData(index,coord.Lat(),Qt::EditRole);
    index=model->index(model->rowCount()-1,FlightDataModel::LNGPOSITION,QModelIndex());
    model->setData(index,coord.Lng(),Qt::EditRole);
}

/**
 * @brief ModelMapProxy::deleteAll When all the waypoints are deleted graphically, update the model
 */
void ModelMapProxy::deleteAll()
{
    model->removeRows(0,model->rowCount(),QModelIndex());
}
