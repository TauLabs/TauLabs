/**
******************************************************************************
*
* @file       mappointitem.h
* @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
* @brief      A graphicsItem representing a MapPointItem
* @see        The GNU Public License (GPL) Version 3
* @defgroup   OPMapWidget
* @{
*
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
#ifndef MAPPOINTITEM_H
#define MAPPOINTITEM_H

#include <QGraphicsItem>
#include <QLabel>
#include <QObject>
#include <QPainter>
#include <QPoint>

#include "../internals/pointlatlng.h"
#include "mapgraphicitem.h"
#include "physical_constants.h"

namespace mapcontrol
{

struct distBearingAltitude
{
    double distance;
    double bearing;
    distBearingAltitude() : distance(0.0), bearing(0.0) { }
};

/**
* @brief A QGraphicsItem representing a MapPointItem
*
* @class MapPointItem mappointitem.h "mappointitem.h"
*/
class MapPointItem: public QObject,public QGraphicsItem
{
    Q_OBJECT
    Q_INTERFACES(QGraphicsItem)
public:
    enum GraphicItemTypes {TYPE_WAYPOINTITEM = 1, TYPE_UAVITEM = 2, TYPE_HOMEITEM = 4, TYPE_GPSITEM = 6};

    /**
    * @brief Returns the MapPointItem description
    *
    * @return QString
    */
    QString Description(){return description;}

    /**
    * @brief Sets the MapPointItem description
    *
    * @param value
    */
    void SetDescription(QString const& value);

    /**
    * @brief Returns MapPointItem LatLng coordinate
    *
    */
    internals::PointLatLng Coord(){return coord;}

    /**
    * @brief  Sets MapPointItem LatLng coordinate
    *
    * @param value
    */
    virtual void SetCoord(internals::PointLatLng const& value);

    /**
    * @brief Returns the MapPointItem altitude
    *
    * @return int
    */
    float Altitude(){return altitude;}

    /**
    * @brief Sets the MapPointItem Altitude
    *
    * @param value
    */
    virtual void SetAltitude(const float &value);
    void setRelativeCoord(distBearingAltitude value);
    distBearingAltitude getRelativeCoord(){return relativeCoord;}

protected:
    MapGraphicItem* map;

    internals::PointLatLng coord; //coordinates of this MapPointItem
    float altitude;
    distBearingAltitude relativeCoord;
    QString description;

    double DistanceToPoint_2D(const internals::PointLatLng &coord);
    double DistanceToPoint_3D(const internals::PointLatLng &coord, const int &altitude);
private:

    QGraphicsSimpleTextItem* text;
    QGraphicsRectItem* textBG;
    QGraphicsSimpleTextItem* numberI;
    QGraphicsRectItem* numberIBG;
    QTransform transf;
    QString myCustomString;

public slots:
signals:
    void absolutePositionChanged(internals::PointLatLng coord, float altitude);
    void relativePositionChanged(QPointF point, MapPointItem* mappoint);
    void aboutToBeDeleted(MapPointItem *);
};
}
#endif // MAPPOINTITEM_H
