/**
******************************************************************************
*
* @file       waypointcurve.cpp
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
* @brief      A graphicsItem representing a curve connecting 2 waypoints
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
#include "waypointcurve.h"
#include <math.h>
#include "homeitem.h"

namespace mapcontrol
{

/**
 * @brief WayPointCurve::WayPointCurve Create the curve
 * @param start The starting location of the curve (will redraw if moved)
 * @param dest The ending location of the curve (will redraw if moved)
 * @param radius Radius of the curve
 * @param clockwise Whether to curve clockwise or counter (when going from start to finish) as view from above
 * @param map Handle to the map object
 * @param color Color of the curve
 */
WayPointCurve::WayPointCurve(WayPointItem *start, WayPointItem *dest, double radius, bool clockwise, MapGraphicItem *map,QColor color) :
    MapArc(start, dest, 1.0/radius, clockwise, true/* FIXME: Should be curvature*/, map, color)
{
    connect(start,SIGNAL(aboutToBeDeleted(MapPointItem*)),this,SLOT(waypointdeleted()));
    connect(dest,SIGNAL(aboutToBeDeleted(MapPointItem*)),this,SLOT(waypointdeleted()));
    refreshLocations();
}

//! Return the type of the QGraphicsEllipseItem
int WayPointCurve::type() const
{
    // Enable the use of qgraphicsitem_cast with this item.
    return Type;
}

/**
 * @brief WayPointCurve::paint Draw the path arc
 * @param painter The painter for drawing
 */
void WayPointCurve::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    QPen myPen = pen();
    myPen.setColor(myColor);
    painter->setPen(myPen);
    painter->setBrush(myColor);

    qreal arrowSize = 10;
    QBrush brush=painter->brush();

    QPointF arrowP1 = midpoint + QPointF(sin(midpoint_angle + M_PI / 3) * arrowSize,
                                   cos(midpoint_angle + M_PI / 3) * arrowSize);
    QPointF arrowP2 = midpoint + QPointF(sin(midpoint_angle + M_PI - M_PI / 3) * arrowSize,
                                   cos(midpoint_angle + M_PI - M_PI / 3) * arrowSize);

    arrowHead.clear();
    arrowHead << midpoint << arrowP1 << arrowP2;
    painter->drawPolygon(arrowHead);
    painter->setBrush(brush);
    painter->drawArc(this->rect(), this->startAngle(), this->spanAngle());
}

void WayPointCurve::waypointdeleted()
{
    this->deleteLater();
}

}
