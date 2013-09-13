/**
******************************************************************************
*
* @file       maparc.cpp
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
* @brief      A graphicsItem representing an arc connecting 2 waypoints
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
#include "maparc.h"
#include <math.h>
#include "homeitem.h"

namespace mapcontrol
{

/**
 * @brief MapArc::MapArc Create the curve
 * @param start The starting location of the curve (will redraw if moved)
 * @param dest The ending location of the curve (will redraw if moved)
 * @param radius Radius of the curve
 * @param clockwise Whether to curve clockwise or counter (when going from start to finish) as view from above
 * @param map Handle to the map object
 * @param color Color of the curve
 */
MapArc::MapArc(MapPointItem *start, MapPointItem *dest, double curvature, bool clockwise, bool rank, MapGraphicItem *map,QColor color) :
    QGraphicsEllipseItem(map),
    my_map(map),
    myColor(color),
    m_start(start),
    m_dest(dest),
    m_curvature(curvature),
    m_clockwise(clockwise),
    m_rank(rank)
{
    connect(start, SIGNAL(relativePositionChanged(QPointF, MapPointItem*)), this, SLOT(refreshLocations()));
    connect(dest,  SIGNAL(relativePositionChanged(QPointF, MapPointItem*)), this, SLOT(refreshLocations()));
    connect(map, SIGNAL(childSetOpacity(qreal)),this,SLOT(setOpacitySlot(qreal)));
}


/**
 * @brief MapArc::refreshLocations Update the settings for the
 * arc when it is moved or the zoom changes
 */
void MapArc::refreshLocations()
{
    double m_n;
    double m_e;
    double p_n;
    double p_e;
    double d;
    double center_x;
    double center_y;

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

    double radius_sign = (m_curvature > 0) ? 1 : -1;
    double pixels2meters = my_map->Projection()->GetGroundResolution(my_map->ZoomTotal(), m_start->Coord().Lat());
    double radius = fabs((1.0/m_curvature) / pixels2meters);

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

    // Compute the midpoint along the arc for the arrow
    d = sqrt(radius * radius / (p_n * p_n + p_e * p_e));
    midpoint.setX(center_x - p_n * d);
    midpoint.setY(center_y - p_e * d);
    midpoint_angle = -atan2(m_dest->pos().y() - m_start->pos().y(), m_dest->pos().x() - m_start->pos().x());

    double startAngle = atan2(-(m_start->pos().y() - center_y), m_start->pos().x() - center_x);
    double endAngle = atan2(-(m_dest->pos().y() - center_y), m_dest->pos().x() - center_x);
    double span = endAngle - startAngle;
    if (!m_clockwise) {
        if (span > 0)
            span = span - 2 * M_PI;
    } else {
        if (span < 0)
            span = span + 2 * M_PI;
    }
    setRect(center_x - radius, center_y - radius, 2 * radius, 2 * radius);
    setStartAngle(startAngle * 180.0 / M_PI * 16.0);
    setSpanAngle(span * 180.0 / M_PI * 16.0);
    update();
}

void MapArc::endpointdeleted()
{
    this->deleteLater();
}

void MapArc::setOpacitySlot(qreal opacity)
{
    setOpacity(opacity);
}

}
