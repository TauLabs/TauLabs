/**
******************************************************************************
*
* @file       mapcircle.cpp
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
* @brief      A graphicsItem representing a circle connecting 2 map points
* @see        The GNU Public License (GPL) Version 3
* @defgroup   TLMapWidget
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
#include "mapcircle.h"
#include <math.h>
#include "homeitem.h"
#include "mappointitem.h"

namespace mapcontrol
{
MapCircle::MapCircle(MapPointItem *center, MapPointItem *radius, bool clockwise, MapGraphicItem *map, QColor color) :
    QGraphicsEllipseItem(map), my_center(center), my_radius(radius),
    my_map(map), myColor(color), myClockWise(clockwise)
{
    connect(center, SIGNAL(relativePositionChanged(QPointF, WayPointItem*)), this, SLOT(refreshLocations()));
    connect(radius, SIGNAL(relativePositionChanged(QPointF, WayPointItem*)), this, SLOT(refreshLocations()));
    connect(center, SIGNAL(aboutToBeDeleted(MapPointItem*)), this, SLOT(pointdeleted()));
    connect(radius, SIGNAL(aboutToBeDeleted(MapPointItem*)), this, SLOT(pointdeleted()));
    refreshLocations();
    connect(map,SIGNAL(childSetOpacity(qreal)),this,SLOT(setOpacitySlot(qreal)));
}

MapCircle::MapCircle(HomeItem *center, MapPointItem *radius, bool clockwise, MapGraphicItem *map, QColor color) :
    QGraphicsEllipseItem(map), my_center(center), my_radius(radius),
    my_map(map), myColor(color), myClockWise(clockwise)
{
    connect(center, SIGNAL(absolutePositionChanged(internals::PointLatLng, float)), this, SLOT(refreshLocations()));
    connect(radius, SIGNAL(relativePositionChanged(QPointF)), this, SLOT(refreshLocations()));
    connect(radius, SIGNAL(aboutToBeDeleted(MapPointItem*)), this, SLOT(pointdeleted()));
    refreshLocations();
    connect(map,SIGNAL(childSetOpacity(qreal)),this,SLOT(setOpacitySlot(qreal)));
}

int MapCircle::type() const
{
    // Enable the use of qgraphicsitem_cast with this item.
    return Type;
}

void MapCircle::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    QPointF p1;
    QPointF p2;
    p1=QPointF(line.p1().x(),line.p1().y()+line.length());
    p2=QPointF(line.p1().x(),line.p1().y()-line.length());
    QPen myPen = pen();
    myPen.setColor(myColor);
    qreal arrowSize = 10;
    painter->setPen(myPen);
    QBrush brush=painter->brush();
    painter->setBrush(myColor);
    double angle =0;
    if(!myClockWise)
        angle+=M_PI;

    QPointF arrowP1 = p1 + QPointF(sin(angle + M_PI / 3) * arrowSize,
                                   cos(angle + M_PI / 3) * arrowSize);
    QPointF arrowP2 = p1 + QPointF(sin(angle + M_PI - M_PI / 3) * arrowSize,
                                   cos(angle + M_PI - M_PI / 3) * arrowSize);

    QPointF arrowP21 = p2 + QPointF(sin(angle + M_PI + M_PI / 3) * arrowSize,
                                    cos(angle + M_PI + M_PI / 3) * arrowSize);
    QPointF arrowP22 = p2 + QPointF(sin(angle + M_PI + M_PI - M_PI / 3) * arrowSize,
                                    cos(angle + M_PI + M_PI - M_PI / 3) * arrowSize);

    arrowHead.clear();
    arrowHead << p1 << arrowP1 << arrowP2;
    painter->drawPolygon(arrowHead);
    arrowHead.clear();
    arrowHead << p2 << arrowP21 << arrowP22;
    painter->drawPolygon(arrowHead);
    painter->translate(-line.length(),-line.length());
    painter->setBrush(brush);
    painter->drawEllipse(this->rect());

}

void MapCircle::refreshLocations()
{
    line=QLineF(my_center->pos(),my_radius->pos());
    this->setRect(my_center->pos().x(),my_center->pos().y(),2*line.length(),2*line.length());
    this->update();
}

void MapCircle::pointdeleted()
{
    this->deleteLater();
}

void MapCircle::setOpacitySlot(qreal opacity)
{
    setOpacity(opacity);
}

}
