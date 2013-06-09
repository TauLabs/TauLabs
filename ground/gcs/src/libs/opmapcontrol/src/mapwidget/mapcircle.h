/**
******************************************************************************
*
* @file       mapcircle.h
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
* @brief      A graphicsItem representing a circle connecting 2 map point
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
#ifndef MAPCIRCLE_H
#define MAPCIRCLE_H

#include "mappointitem.h"

namespace mapcontrol
{

class HomeItem;

class MapCircle: public QObject, public QGraphicsEllipseItem
{
    Q_OBJECT
    Q_INTERFACES(QGraphicsItem)
public:
    enum { Type = UserType + 9 };
    MapCircle(MapPointItem *center, MapPointItem *radius, bool clockwise, MapGraphicItem *map, QColor color=Qt::green);
    MapCircle(HomeItem *center, MapPointItem *radius, bool clockwise, MapGraphicItem *map, QColor color=Qt::green);
    int type() const;
    void setColor(const QColor &color)
        { myColor = color; }
private:
    QGraphicsItem *my_center;
    QGraphicsItem *my_radius;
    MapGraphicItem *my_map;
    QPolygonF arrowHead;
    QColor myColor;
    bool myClockWise;
    QLineF line;
protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
public slots:
    void refreshLocations();
    void pointdeleted();
    void setOpacitySlot(qreal opacity);
};
}

#endif // MAPCIRCLE_H
