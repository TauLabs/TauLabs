/**
******************************************************************************
*
* @file       mapgraphicitem.h
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
* @brief      The main graphicsItem used on the widget, contains the map and map logic
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
#ifndef MAPGRAPHICITEM_H
#define MAPGRAPHICITEM_H

#include <QGraphicsItem>
#include "../internals/core.h"
#include "../core/diagnostics.h"
#include "configuration.h"
#include <QtGui>
#include <QTransform>
#include <QWidget>
#include <QBrush>
#include <QFont>
#include <QObject>

namespace mapcontrol
{
    class WayPointItem;
    class TLMapWidget;
    /**
    * @brief The main graphicsItem used on the widget, contains the map and map logic
    *
    * @class MapGraphicItem mapgraphicitem.h "mapgraphicitem.h"
    */
    class MapGraphicItem:public QObject,public QGraphicsItem
    {
        friend class mapcontrol::TLMapWidget;
        Q_OBJECT
        Q_INTERFACES(QGraphicsItem)
    public:


        /**
        * @brief Contructer
        *
        * @param core
        * @param configuration the configuration to be used
        * @return
        */
        MapGraphicItem(internals::Core *core,Configuration *configuration);
        QRectF boundingRect() const;
        void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                   QWidget *widget);

        QSize sizeHint()const;
        /**
        * @brief Convertes LatLong coordinates to local item coordinates
        *
        * @param point LatLong point to be converted
        * @return core::Point Local item point
        */
        core::Point FromLatLngToLocal(internals::PointLatLng const& point);
        /**
        * @brief Converts from local item coordinates to LatLong point
        *
        * @param x x local coordinate
        * @param y y local coordinate
        * @return internals::PointLatLng LatLng coordinate
        */
        internals::PointLatLng FromLocalToLatLng(qint64 x, qint64 y);
        /**
        * @brief Returns true if map is being dragged
        *
        * @return
        */
        bool IsDragging()const{return core->IsDragging();}

        QImage lastimage;
        core::Point lastimagepoint;
        void paintImage(QPainter* painter);
        void ConstructLastImage(int const& zoomdiff);
        internals::PureProjection* Projection()const{return core->Projection();}
        double Zoom();
        double ZoomDigi();
        double ZoomTotal();
        void setOverlayOpacity(qreal value);
    protected:
        void mouseMoveEvent ( QGraphicsSceneMouseEvent * event );
        void mousePressEvent ( QGraphicsSceneMouseEvent * event );
        void wheelEvent ( QGraphicsSceneWheelEvent * event );
        void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
        bool IsMouseOverMarker()const{return isMouseOverMarker;}
        void keyPressEvent ( QKeyEvent * event );
        void keyReleaseEvent ( QKeyEvent * event );

        /**
        * @brief Returns current map zoom
        *
        * @return int Current map zoom
        */
        int ZoomStep()const;
        /**
        * @brief Sets map zoom
        *
        * @param value zoom value
        */
        void SetZoomStep(qint32 const& value);

        /**
        * @brief Ask Stacey
        *
        * @param value
        */
        void SetShowDragons(bool const& value);
    private:
        bool showDragons;
        bool SetZoomToFitRect(internals::RectLatLng const& rect);
        internals::Core *core;
        Configuration *config;
        bool showTileGridLines;
        qreal MapRenderTransform;
        void DrawMap2D(QPainter *painter);
        /**
        * @brief Maximum possible zoom
        *
        * @var maxZoom
        */
        int maxZoom;
        /**
        * @brief Minimum possible zoom
        *
        * @var minZoom
        */
        int minZoom;
        internals::RectLatLng selectedArea;
        internals::PointLatLng selectionStart;
        internals::PointLatLng selectionEnd;
        double zoomReal;
        double zoomDigi;
        QRectF maprect;
        bool isSelected;
        bool isMouseOverMarker;
        QPixmap dragons;
        void SetIsMouseOverMarker(bool const& value){isMouseOverMarker = value;}

        qreal rotation;
        /**
        * @brief Creates a rectangle that represents the "view" of the cuurent map, to compensate
        *       rotation
        *
        * @param rect original rectangle
        * @param angle angle of rotation
        * @return QRectF
        */
        QRectF boundingBox(QRectF const& rect, qreal const& angle);
        /**
        * @brief Returns the maximum allowed zoom
        *
        * @return int
        */
        int MaxZoom()const{return core->MaxZoom();}
        /**
        * @brief Returns the minimum allowed zoom
        *
        * @return int
        */
        int MinZoom()const{return minZoom;}
        internals::MouseWheelZoomType::Types GetMouseWheelZoomType(){return core->GetMouseWheelZoomType();}
        void SetSelectedArea(internals::RectLatLng const& value){selectedArea = value;this->update();}
        internals::RectLatLng SelectedArea()const{return selectedArea;}
        internals::RectLatLng BoundsOfMap;
        void Offset(qint64 const& x, qint64 const& y);
        bool CanDragMap()const{return core->CanDragMap;}
        void SetCanDragMap(bool const& value){core->CanDragMap = value;}

        void SetZoom(double const& value);
        void mapRotate ( qreal angle );
        void start();
        void  ReloadMap(){core->ReloadMap();}
        GeoCoderStatusCode::Types SetCurrentPositionByKeywords(QString const& keys){return core->SetCurrentPositionByKeywords(keys);}
        MapType::Types GetMapType(){return core->GetMapType();}
        void SetMapType(MapType::Types const& value){core->SetMapType(value);}

    private slots:
        void Core_OnNeedInvalidation();
        void childPosRefresh();
    public slots:
        /**
        * @brief To be called when the scene size changes
        *
        * @param rect
        */
        void resize ( QRectF const &rect=QRectF() );
    signals:
        /**
        * @brief Fired when the current zoom is changed
        *
        * @param zoom
        */
        void wpdoubleclicked(WayPointItem * wp);
        void zoomChanged(double zoomtotal,double zoomreal,double zoomdigi);
        void childRefreshPosition();
        void childSetOpacity(qreal value);
    };
}
#endif // MAPGRAPHICITEM_H
