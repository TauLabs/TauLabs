/**
******************************************************************************
*
* @file       trailitem.cpp
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
* @brief      A graphicsItem representing a trail point
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
#include "trailitem.h"
#include <QDateTime>
namespace mapcontrol
{
TrailItem::TrailItem(internals::PointLatLng const& coord,int const& altitude, QBrush color, MapGraphicItem *map):QGraphicsItem(map),coord(coord),m_brush(color),m_map(map)
    {
        QDateTime time=QDateTime::currentDateTime();
        QString coord_str = " " + QString::number(coord.Lat(), 'f', 6) + "   " + QString::number(coord.Lng(), 'f', 6);
        setToolTip(QString(tr("Position:")+"%1\n"+tr("Altitude:")+"%2\n"+tr("Time:")+"%3").arg(coord_str).arg(QString::number(altitude)).arg(time.toString()));
    }

    void TrailItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
    {
        Q_UNUSED(option);
        Q_UNUSED(widget);

        painter->setBrush(m_brush);
        painter->drawEllipse(-2,-2,4,4);
    }
    QRectF TrailItem::boundingRect()const
    {
        return QRectF(-2,-2,4,4);
    }


    int TrailItem::type()const
    {
        return Type;
    }

    void TrailItem::setPosSLOT()
    {
        setPos(m_map->FromLatLngToLocal(this->coord).X(),m_map->FromLatLngToLocal(this->coord).Y());
    }
}
