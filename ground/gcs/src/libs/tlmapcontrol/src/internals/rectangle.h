/**
******************************************************************************
*
* @file       rectangle.h
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
#ifndef RECTANGLE_H
#define RECTANGLE_H
//#include <point.h>
#include "../core/size.h"
#include "math.h"
using namespace core;
namespace internals
{
struct Rectangle
{

    friend quint32 qHash(Rectangle const& rect);
    friend bool operator==(Rectangle const& lhs,Rectangle const& rhs);
public:
    static Rectangle Empty;
    static Rectangle FromLTRB(qint32 left, qint32 top, qint32 right, qint32 bottom);
    Rectangle(){x=0; y=0; width=0; height=0; }
    Rectangle(qint32 x, qint32 y, qint32 width, qint32 height)
    {
        this->x = x;
        this->y = y;
        this->width = width;
        this->height = height;
    }
    Rectangle(core::Point location, core::Size size)
    {
        this->x = location.X();
        this->y = location.Y();
        this->width = size.Width();
        this->height = size.Height();
    }
    core::Point GetLocation() {
        return core::Point(x, y);
    }
    void SetLocation(const core::Point &value)
    {
        x = value.X();
        y = value.Y();
    }
    qint32 X(){return x;}
    qint32 Y(){return y;}
    void SetX(const qint32 &value){x=value;}
    void SetY(const qint32 &value){y=value;}
    qint32 Width(){return width;}
    void SetWidth(const qint32 &value){width=value;}
    qint32 Height(){return height;}
    void SetHeight(const qint32 &value){height=value;}
    qint32 Left(){return x;}
    qint32 Top(){return y;}
    qint32 Right(){return x+width;}
    qint32 Bottom(){return y+height;}
    bool IsEmpty(){return (height==0 && width==0 && x==0 && y==0);}
    bool operator==(const Rectangle &cSource)
    {
        return (cSource.x == x && cSource.y == y && cSource.width == width && cSource.height == height);
    }


    bool operator!=(const Rectangle &cSource){return !(*this==cSource);}
    bool Contains(const qint32 &x, const qint32 &y)
    {
        return this->x<=x && x<this->x+this->width && this->y<=y && y<this->y+this->height;
    }
    bool Contains(const core::Point &pt)
    {
        return Contains(pt.X(),pt.Y());
    }
    bool Contains(const Rectangle &rect)
    {
        return (this->x <= rect.x) &&
                    ((rect.x + rect.width) <= (this->x + this->width)) &&
                    (this->y <= rect.y) &&
                    ((rect.y + rect.height) <= (this->y + this->height));
    }


    void Inflate(const qint32 &width,const qint32 &height)
          {
             this->x -= width;
             this->y -= height;
             this->width += 2*width;
             this->height += 2*height;
          }
    void Inflate(Size &size)
          {

             Inflate(size.Width(), size.Height());
          }
    static Rectangle Inflate(Rectangle rect, qint32 x, qint32 y);

    void Intersect(const Rectangle &rect)
          {
        Rectangle result = Rectangle::Intersect(rect, *this);

             this->x = result.X();
             this->y = result.Y();
             this->width = result.Width();
             this->height = result.Height();
          }
    static Rectangle Intersect(Rectangle a, Rectangle b);
    bool IntersectsWith(const Rectangle &rect)
         {
            return (rect.x < this->x + this->width) &&
               (this->x < (rect.x + rect.width)) &&
               (rect.y < this->y + this->height) &&
               (this->y < rect.y + rect.height);
         }
    static Rectangle Union(const Rectangle &a,const Rectangle &b);
    void Offset(const core::Point &pos)
    {
        Offset(pos.X(), pos.Y());
    }

    void Offset(const qint32 &x,const qint32 &y)
    {
        this->x += x;
        this->y += y;
    }
    QString ToString()
          {
        return "{X=" + QString::number(x) + ",Y=" + QString::number(y) +
                ",Width=" + QString::number(width) +
                ",Height=" +QString::number(height) +"}";
          }
private:
    qint32 x;
    qint32 y;
    qint32 width;
    qint32 height;
};
}
#endif // RECTANGLE_H
