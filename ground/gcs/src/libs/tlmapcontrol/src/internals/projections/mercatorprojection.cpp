/**
******************************************************************************
*
* @file       mercatorprojection.cpp
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @brief      
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
#include "mercatorprojection.h"

 
namespace projections {
MercatorProjection::MercatorProjection():
    MinLatitude(-85.05112878), MaxLatitude(85.05112878),MinLongitude(-180),
    MaxLongitude(180), tileSize(256, 256)
{
}

Point MercatorProjection::FromLatLngToPixel(double lat, double lng, const int &zoom)
{
    Point ret;// = Point.Empty;

    lat = bound(lat, MinLatitude, MaxLatitude);
    lng = bound(lng, MinLongitude, MaxLongitude);

    double x = (lng + 180) / 360;
    double sinLatitude = sin(lat * M_PI / 180);
    double y = 0.5 - log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * M_PI);

    Size s = GetTileMatrixSizePixel(zoom);
    qint64 mapSizeX = s.Width();
    qint64 mapSizeY = s.Height();

    ret.SetX((qint64) round(bound(x * mapSizeX + 0.5, 0, mapSizeX - 1)));
    ret.SetY((qint64) round(bound(y * mapSizeY + 0.5, 0, mapSizeY - 1)));

    return ret;
}

/**
 * @brief MercatorProjection::FromPixelToLatLng Referenced from top-left of globe, so the lat-lon (0,0), i.e. the intersection of the equator and prime meridian, would be [1<<(zoom-1), 1<<(zoom-1)]
 * @param x Horizontal location in [pixels], referenced from left edge of global map
 * @param y Vertical location in [pixels], referenced from top edge of global map
 * @param zoom
 * @return Latitude and Longitude in [degrees]
 */
internals::PointLatLng MercatorProjection::FromPixelToLatLng(const qint64 &x,const qint64 &y,const int &zoom)
{
    internals::PointLatLng ret;// = internals::PointLatLng.Empty;

    Size s = GetTileMatrixSizePixel(zoom);
    double mapSizeX = s.Width();
    double mapSizeY = s.Height();

    //Calculate the percentage distance between top and bottom, and left and right
    double xx = (bound(x, 0, mapSizeX - 1) / mapSizeX) - 0.5;
    double yy = 0.5 - (bound(y, 0, mapSizeY - 1) / mapSizeY);

    ret.SetLat(90 - 360 * atan(exp(-yy * 2 * M_PI)) / M_PI);
    ret.SetLng(360 * xx);

    return ret;
}

Size MercatorProjection::TileSize() const
{
    return tileSize;
}
double MercatorProjection::Axis() const
{
    return 6378137;
}
double MercatorProjection::Flattening() const
{
    return (1.0 / 298.257223563);
}

/**
 * @brief MercatorProjection::GetTileMatrixMaxXY
 * @param zoom
 * @return
 */
Size MercatorProjection::GetTileMatrixMaxXY(const int &zoom)
{
    int xy = (1 << zoom);
    return  Size(xy - 1, xy - 1);
}

/**
 * @brief MercatorProjection::GetTileMatrixMinXY
 * @param zoom UNUSED
 * @return returns Size(0,0)
 */
Size MercatorProjection::GetTileMatrixMinXY(const int &zoom)
{
    Q_UNUSED(zoom);
    return Size(0, 0);
}
}
