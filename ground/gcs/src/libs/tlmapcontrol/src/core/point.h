/**
******************************************************************************
*
* @file       point.h
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
#ifndef OPOINT_H
#define OPOINT_H


#include <QString>

/**
 * For the most part, points are referenced in quadtiles. Due to integer overflow
 * issues, the integers must be 64-bits in order to allow for proper operation
 * of quadtile zoom levels greater than 22.
 */
namespace core {
    struct Size;
    struct Point
    {
        friend quint64 qHash(Point const& point);
        friend bool operator==(Point const& lhs,Point const& rhs);
        friend bool operator!=(Point const& lhs,Point const& rhs);
    public:

        Point();
        Point(qint64 x, qint64 y);
        Point(Size sz);
        bool IsEmpty(){return empty;}
        qint64 X()const{return this->x;}
        qint64 Y()const{return this->y;}
        void SetX(const qint64 &value){x=value;empty=false;}
        void SetY(const qint64 &value){y=value;empty=false;}
        QString ToString()const{return "{"+QString::number(x)+","+QString::number(y)+"}";}

        static Point Empty;
        void Offset(const qint64 &dx,const qint64 &dy)
        {
            x += dx;
            y += dy;
        }
        void Offset(Point p)
        {
            Offset(p.x, p.y);
        }

    private:
        qint64 x;
        qint64 y;
        bool empty;
    };
}
#endif // POINT_H
