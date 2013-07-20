/**
******************************************************************************
*
* @file       mappoint.cpp
* @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
* @brief      A graphicsItem representing a map item
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
#include "mappointitem.h"

namespace mapcontrol
{
    void MapPointItem::SetAltitude(const float &value)
    {
        if(altitude==value)
            return;
        altitude=value;
        this->update();
    }

    void MapPointItem::setRelativeCoord(distBearingAltitude value)
    {
        relativeCoord=value;
        this->update();
    }

    void MapPointItem::SetCoord(const internals::PointLatLng &value)
    {
        if(coord == value)
            return;
        coord = value;
        distBearingAltitude back=relativeCoord;
        if(qAbs(back.bearing-relativeCoord.bearing)>0.01 || qAbs(back.distance-relativeCoord.distance)>0.1)
        {
            relativeCoord=back;
        }
        this->update();
    }
    void MapPointItem::SetDescription(const QString &value)
    {
        if(description==value)
            return;
        description=value;
        this->update();
    }


    /**
     * @brief MapPointItem::DistanceToPoint_2D Calculates distance from this point to second point
     * @param coord2 Coordinates, second point
     * @return
     */
    double MapPointItem::DistanceToPoint_2D(const internals::PointLatLng &coord2)
    {
       return internals::PureProjection::DistanceBetweenLatLng(coord, coord2);
    }


    /**
     * @brief MapPointItem::DistanceToPoint_3D Calculates distance from this point to second point
     * @param coord2 Coordinates, second point
     * @param altitude2 Altitude, second point
     * @return
     */
    double MapPointItem::DistanceToPoint_3D(const internals::PointLatLng &coord2, const int &altitude2)
    {
        return internals::PureProjection::DistanceBetweenLatLngAlt(coord, altitude, coord2, altitude2);
    }



}
