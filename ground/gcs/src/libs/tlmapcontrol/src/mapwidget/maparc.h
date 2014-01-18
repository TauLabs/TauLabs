/**
******************************************************************************
*
* @file       maparc.h
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
* @brief      A graphicsItem representing an arc connecting 2 points
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
#ifndef MAPARC_H
#define MAPARC_H

#include "mappointitem.h"

namespace mapcontrol
{

/**
 * @brief The MapArc class draws an arc between two graphics items of a given
 * radius and direction of curvature.  It will display a red straight line if the
 * radius is insufficient to connect the two waypoints
 */
class MapArc:public QObject, public QGraphicsEllipseItem
{
    Q_OBJECT
    Q_INTERFACES(QGraphicsItem)
public:
    enum GraphicItemTypes {TYPE_WAYPOINTCURVE = 9};
    enum ArcRank {ARC_RANK_MAJOR, ARC_RANK_MINOR};

    MapArc(MapPointItem *start, MapPointItem *dest,
                  double curvature, bool clockwise, bool rank,
                  MapGraphicItem * map, QColor color=Qt::green);
    void setColor(const QColor &color) { myColor = color; }

protected:
    //! Handle to the map this is shown on
    MapGraphicItem * my_map;
    QColor myColor;

    //! Start of the arc
    MapPointItem * m_start;

    //! End of the arc
    MapPointItem * m_dest;

    //! Curvature of the arc
    double m_curvature;

    //! Direction of curvature
    bool m_clockwise;

    //! Arc rank
    bool m_rank;

    //! Center coordinate
    QPointF center;

    //! Half way point
    QPointF midpoint;

    //! Angle of normal at midpoint
    double midpoint_angle;

public slots:
    //! Called if the endpoints move, in order to redraw the arc
    void refreshLocations();

    //! Called if the start or end point is destroyed
    void endpointdeleted();

    void setOpacitySlot(qreal opacity);
};
}

#endif // MAPARC_H
