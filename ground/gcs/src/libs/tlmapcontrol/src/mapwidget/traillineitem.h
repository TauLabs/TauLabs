/**
******************************************************************************
*
* @file       traillineitem.h
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
* @brief      A graphicsItem representing a WayPoint
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
#ifndef TAILLINEITEM_H
#define TAILLINEITEM_H

#include <QGraphicsItem>
#include <QPainter>
#include <QLabel>
#include "../internals/pointlatlng.h"
#include <QObject>
#include "mapgraphicitem.h"

namespace mapcontrol
{

    class TrailLineItem:public QObject,public QGraphicsLineItem
    {
        Q_OBJECT
        Q_INTERFACES(QGraphicsItem)
    public:
                enum { Type = UserType + 7 };
        TrailLineItem(internals::PointLatLng const& coord1,internals::PointLatLng const& coord2, QBrush color,MapGraphicItem * map);
        int type() const;
        internals::PointLatLng coord1;
        internals::PointLatLng coord2;
    private:
        QBrush m_brush;
        MapGraphicItem * m_map;
    public slots:
        void setLineSlot();
    signals:

    };
}

#endif // TAILLINEITEM_H
