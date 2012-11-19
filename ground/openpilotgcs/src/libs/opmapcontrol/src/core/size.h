/**
******************************************************************************
*
* @file       size.h
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
#ifndef SIZE_H
#define SIZE_H

#include "point.h"
#include <QString>
#include <QHash>

/**
 * For the most part, size is referenced in quadtiles. Due to integer overflow
 * issues, the integers must be 64-bits in order to allow for proper operation
 * of quadtile zoom levels greater than 22.
 */
namespace core {
    struct Size
    {
        //Size must be kept in 64-bit signed integer format, or else the data type overflows at mid-twenties zoom levels
        Size();
        Size(Point pt){width=pt.X(); height=pt.Y();}
        Size(qint64 Width, qint64 Height){width=Width; height=Height;}
        friend quint64 qHash(Size const& size);
        //  friend bool operator==(Size const& lhs,Size const& rhs);
        Size operator-(const Size &sz1){return Size(width-sz1.width,height-sz1.height);}
        Size operator+(const Size &sz1){return Size(sz1.width+width,sz1.height+height);}

        int GetHashCode(){return width^height;}
        quint64 qHash(Size const& /*rect*/){return width^height;}
        QString ToString(){return "With="+QString::number(width)+" ,Height="+QString::number(height);}
        qint64 Width()const {return width;}
        qint64 Height()const {return height;}
        void SetWidth(qint64 const& value){width=value;}
        void SetHeight(qint64 const& value){height=value;}
    private:
        qint64 width;
        qint64 height;
    };
}
#endif // SIZE_H
