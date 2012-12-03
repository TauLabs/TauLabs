/**
******************************************************************************
*
* @file       waypointcurvele.cpp
* @author     PhoenixPilot Project, http://github.com/PhoenixPilot Copyright (C) 2012.
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
    m_start(start), m_dest(dest), m_radius(radius), m_clockwise(clockwise),
    my_map(map),QGraphicsEllipseItem(map),myColor(color)
{
    connect(start,SIGNAL(localPositionChanged(QPointF,WayPointItem*)),this,SLOT(refreshLocations()));
    connect(dest,SIGNAL(localPositionChanged(QPointF,WayPointItem*)),this,SLOT(refreshLocations()));
    connect(start,SIGNAL(aboutToBeDeleted(WayPointItem*)),this,SLOT(waypointdeleted()));
    connect(dest,SIGNAL(aboutToBeDeleted(WayPointItem*)),this,SLOT(waypointdeleted()));
    refreshLocations();
    connect(map,SIGNAL(childSetOpacity(qreal)),this,SLOT(setOpacitySlot(qreal)));

}

//! Return the type of the QGraphicsEllipseItem
int WayPointCurve::type() const
{
    // Enable the use of qgraphicsitem_cast with this item.
    return Type;
}

void WayPointCurve::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    QPen myPen = pen();
    myPen.setColor(myColor);
    painter->setPen(myPen);
    painter->setBrush(myColor);

    //QGraphicsEllipseItem::paint(painter, option, widget);
    painter->drawArc(this->rect(), this->startAngle(), this->spanAngle());

    /*
    qreal arrowSize = 10;
    QBrush brush=painter->brush();
    QPointF p1;
    QPointF p2;
    p1=QPointF(line.p1().x(),line.p1().y()+line.length());
    p2=QPointF(line.p1().x(),line.p1().y()-line.length());
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
    */
}

void WayPointCurve::refreshLocations()
{
    double m_n, m_e, p_n, p_e, d, center_x, center_y;

    // Center between start and end
    m_n = (m_start->pos().x() + m_dest->pos().x()) / 2;
    m_e = (m_start->pos().y() + m_dest->pos().y()) / 2;

    // Normal vector the line between start and end.
    if (!m_clockwise) {
        p_n = -(m_dest->pos().y() - m_start->pos().y());
        p_e = (m_dest->pos().x() - m_start->pos().x());
    } else {
        p_n = (m_dest->pos().y() - m_start->pos().y());
        p_e = -(m_dest->pos().x() - m_start->pos().x());
    }

    double radius_sign = (m_radius > 0) ? 1 : -1;
    double pixels2meters = my_map->Projection()->GetGroundResolution(my_map->ZoomTotal(), m_start->Coord().Lat());
    double radius = fabs(m_radius / pixels2meters);

    // Work out how far to go along the perpendicular bisector
    d = sqrt(radius * radius / (p_n * p_n + p_e * p_e) - 0.25f);

    if (fabs(p_n) < 1e-3 && fabs(p_e) < 1e-3) {
        center_x = m_n;
        center_y = m_e;
    } else {
        center_x = m_n + p_n * d * radius_sign;
        center_y = m_e + p_e * d * radius_sign;
    }

    // Store the center
    center.setX(center_x);
    center.setY(center_y);

    qDebug() << "Center: " << center_x << ", " << center_y;
    qDebug() << "Px / m: " << pixels2meters;
    qDebug() << "Radius " << m_radius << " m, " << radius;

    double startAngle = atan2(-(m_start->pos().y() - center_y), m_start->pos().x() - center_x);
    double endAngle = atan2(-(m_dest->pos().y() - center_y), m_dest->pos().x() - center_x);
    double span = endAngle - startAngle;
    if (!m_clockwise) {
        qDebug() << startAngle << " " << endAngle << " " << span;
        if (span > 0)
            span = span - 2 * M_PI;
    } else {
        qDebug() << startAngle << " " << endAngle << " " << span;
        if (span < 0)
            span = span + 2 * M_PI;
    }
    setRect(center_x - radius, center_y - radius, 2 * radius, 2 * radius);
    setStartAngle(startAngle * 180.0 / M_PI * 16.0);
    setSpanAngle(span * 180.0 / M_PI * 16.0);
    update();
}

void WayPointCurve::waypointdeleted()
{
    this->deleteLater();
}

void WayPointCurve::setOpacitySlot(qreal opacity)
{
    setOpacity(opacity);
}

}
