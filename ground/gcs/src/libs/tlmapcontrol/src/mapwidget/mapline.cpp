/**
******************************************************************************
*
* @file       mapline.cpp
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
* @brief      A graphicsItem representing a line connecting 2 map points
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
#include "mapline.h"
#include <math.h>
#include "homeitem.h"

namespace mapcontrol
{
MapLine::MapLine(MapPointItem *from, MapPointItem *to, MapGraphicItem *map, QColor color) :
    QGraphicsLineItem(map), source(from), destination(to), my_map(map), myColor(color)
{
    this->setLine(to->pos().x(),to->pos().y(),from->pos().x(),from->pos().y());
    connect(from, SIGNAL(relativePositionChanged(QPointF, MapPointItem*)), this, SLOT(refreshLocations()));
    connect(to, SIGNAL(relativePositionChanged(QPointF, MapPointItem*)), this, SLOT(refreshLocations()));
    connect(from, SIGNAL(aboutToBeDeleted(MapPointItem*)), this, SLOT(pointdeleted()));
    connect(to, SIGNAL(aboutToBeDeleted(MapPointItem*)), this, SLOT(pointdeleted()));
    if(myColor==Qt::green)
        this->setZValue(10);
    else if(myColor==Qt::yellow)
        this->setZValue(9);
    else if(myColor==Qt::red)
        this->setZValue(8);
    connect(map,SIGNAL(childSetOpacity(qreal)),this,SLOT(setOpacitySlot(qreal)));
}

MapLine::MapLine(HomeItem *from, MapPointItem *to, MapGraphicItem *map, QColor color) :
    QGraphicsLineItem(map), source(from), destination(to), my_map(map), myColor(color)
{
    this->setLine(to->pos().x(),to->pos().y(),from->pos().x(),from->pos().y());
    connect(from, SIGNAL(absolutePositionChanged(internals::PointLatLng, float)), this, SLOT(refreshLocations()));
    connect(to, SIGNAL(relativePositionChanged(QPointF, MapPointItem*)), this, SLOT(refreshLocations()));
    connect(to, SIGNAL(aboutToBeDeleted(MapPointItem*)), this, SLOT(pointdeleted()));
    if(myColor==Qt::green)
        this->setZValue(10);
    else if(myColor==Qt::yellow)
        this->setZValue(9);
    else if(myColor==Qt::red)
        this->setZValue(8);
    connect(map,SIGNAL(childSetOpacity(qreal)),this,SLOT(setOpacitySlot(qreal)));
}

int MapLine::type() const
{
    // Enable the use of qgraphicsitem_cast with this item.
    return Type;
}

QPainterPath MapLine::shape() const
{
    QPainterPath path = QGraphicsLineItem::shape();
    path.addPolygon(arrowHead);
    return path;
}

void MapLine::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    QPen myPen = pen();
    myPen.setColor(myColor);
    qreal arrowSize = 10;
    painter->setPen(myPen);
    painter->setBrush(myColor);

    // Prevent segfaults when length is zero
    double angle = (fabs(line().length()) < 1e-3) ? 0 : ::acos(line().dx() / line().length());

    if (line().dy() >= 0)
        angle = (M_PI * 2) - angle;

        QPointF arrowP1 = line().pointAt(0.5) + QPointF(sin(angle + M_PI / 3) * arrowSize,
                                        cos(angle + M_PI / 3) * arrowSize);
        QPointF arrowP2 = line().pointAt(0.5) + QPointF(sin(angle + M_PI - M_PI / 3) * arrowSize,
                                        cos(angle + M_PI - M_PI / 3) * arrowSize);
        arrowHead.clear();
        arrowHead << line().pointAt(0.5) << arrowP1 << arrowP2;

        painter->drawPolygon(arrowHead);
        if(myColor==Qt::red)
            myPen.setWidth(3);
        else if(myColor==Qt::yellow)
            myPen.setWidth(2);
        else if(myColor==Qt::green)
            myPen.setWidth(1);
        painter->setPen(myPen);
        painter->drawLine(line());

}

void MapLine::refreshLocations()
{
    this->setLine(destination->pos().x(),destination->pos().y(),source->pos().x(),source->pos().y());
}

void MapLine::pointdeleted()
{
    this->deleteLater();
}

void MapLine::setOpacitySlot(qreal opacity)
{
    setOpacity(opacity);
}

}
