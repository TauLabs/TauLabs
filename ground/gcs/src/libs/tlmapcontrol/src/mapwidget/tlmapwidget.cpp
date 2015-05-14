/**
******************************************************************************
*
* @file       tlmapwidget.cpp
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
* @brief      The Map Widget, this is the part exposed to the user
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

#include "tlmapwidget.h"
#include <QtGui>
#include <QMetaObject>
#include "waypointitem.h"

namespace mapcontrol
{

    TLMapWidget::TLMapWidget(QWidget *parent, Configuration *config) : QGraphicsView(parent),
        configuration(config),UAV(0),GPS(0),Home(0),followmouse(true),
        compassRose(0),windCompass(0),showuav(false),showhome(false),
        diagTimer(0),diagGraphItem(0),showDiag(false),overlayOpacity(1)
    {
        setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        this->setScene(new QGraphicsScene(this));

        core=new internals::Core;
        map=new MapGraphicItem(core,config);

        scene()->addItem(map);
        Home=new HomeItem(map,this);
        Home->setParentItem(map);
        Home->setZValue(-1);

        setStyleSheet("QToolTip {font-size:8pt; color:blue;opacity: 223; padding:2px; border-width:2px; border-style:solid; border-color: rgb(170, 170, 127);border-radius:4px }");

        connect(map,SIGNAL(zoomChanged(double,double,double)),this,SIGNAL(zoomChanged(double,double,double)));
        connect(map->core,SIGNAL(OnCurrentPositionChanged(internals::PointLatLng)),this,SIGNAL(OnCurrentPositionChanged(internals::PointLatLng)));
        connect(map->core,SIGNAL(OnEmptyTileError(int,core::Point)),this,SIGNAL(OnEmptyTileError(int,core::Point)));
        connect(map->core,SIGNAL(OnMapDrag()),this,SIGNAL(OnMapDrag()));
        connect(map->core,SIGNAL(OnMapTypeChanged(MapType::Types)),this,SIGNAL(OnMapTypeChanged(MapType::Types)));
        connect(map->core,SIGNAL(OnMapZoomChanged()),this,SIGNAL(OnMapZoomChanged()));
        connect(map->core,SIGNAL(OnTileLoadComplete()),this,SIGNAL(OnTileLoadComplete()));
        connect(map->core,SIGNAL(OnTileLoadStart()),this,SIGNAL(OnTileLoadStart()));
        connect(map->core,SIGNAL(OnTilesStillToLoad(int)),this,SIGNAL(OnTilesStillToLoad(int)));
        connect(map,SIGNAL(wpdoubleclicked(WayPointItem*)),this,SIGNAL(OnWayPointDoubleClicked(WayPointItem*)));
        connect(scene(),SIGNAL(selectionChanged()),this,SLOT(OnSelectionChanged()));
        SetShowDiagnostics(showDiag);
        this->setMouseTracking(followmouse);
        SetShowCompassRose(true);
        SetShowWindCompass(false);
        QPixmapCache::setCacheLimit(64*1024);

        this->adjustSize();
    }

    void TLMapWidget::SetShowDiagnostics(bool const& value)
    {
        showDiag=value;
        if(!showDiag)
        {
            if(diagGraphItem!=0)
            {
                delete diagGraphItem;
                diagGraphItem=0;
            }
            if(diagTimer!=0)
            {
                delete diagTimer;
                diagTimer=0;
            }

            if(GPS!=0)
            {
                delete GPS;
                GPS=0;
            }
        }
        else
        {
            diagTimer=new QTimer();
            connect(diagTimer,SIGNAL(timeout()),this,SLOT(diagRefresh()));
            diagTimer->start(500);
            if(GPS==0)
            {
                GPS=new GPSItem(map,this);
                GPS->setParentItem(map);
                setOverlayOpacity(overlayOpacity);
            }
        }

    }

    void TLMapWidget::SetUavPic(QString UAVPic)
    {
        if(UAV!=0)
            UAV->SetUavPic(UAVPic);
        if(GPS!=0)
            GPS->SetUavPic(UAVPic);
    }

    MapLine * TLMapWidget::WPLineCreate(WayPointItem *from, WayPointItem *to,QColor color)
    {
        if(!from|!to)
            return NULL;
        MapLine* ret= new MapLine(from,to,map,color);
        ret->setOpacity(overlayOpacity);
        return ret;
    }

    MapLine * TLMapWidget::WPLineCreate(HomeItem *from, WayPointItem *to,QColor color)
    {
        if(!from|!to)
            return NULL;
        MapLine* ret= new MapLine(from,to,map,color);
        ret->setOpacity(overlayOpacity);
        return ret;
    }

    /**
     * @brief OPMapWidget::WPCurveCreate Create a curve from one waypoint to another with specified radius
     * @param start The starting waypoint
     * @param dest The ending waypoint
     * @param radius The radius to use connecting the two
     * @param clockwise The curvature direction from above
     * @param color The color of the path
     * @return The waypoint curve object
     */
    WayPointCurve * TLMapWidget::WPCurveCreate(WayPointItem *start, WayPointItem *dest, double radius, bool clockwise, QColor color)
    {
        if (!start || !dest)
            return NULL;
        WayPointCurve *ret = new WayPointCurve(start, dest, radius, clockwise, map, color);
        ret->setOpacity(overlayOpacity);
        return ret;
    }

    MapCircle *TLMapWidget::WPCircleCreate(WayPointItem *center, WayPointItem *radius, bool clockwise, QColor color)
    {
        if(!center|!radius)
            return NULL;
        MapCircle* ret= new MapCircle(center,radius,clockwise,map,color);
        ret->setOpacity(overlayOpacity);
        return ret;
    }

    MapCircle *TLMapWidget::WPCircleCreate(HomeItem *center, WayPointItem *radius, bool clockwise,QColor color)
    {
        if(!center|!radius)
            return NULL;
        MapCircle* ret= new MapCircle(center,radius,clockwise,map,color);
        ret->setOpacity(overlayOpacity);
        return ret;
    }

    void TLMapWidget::SetShowUAV(const bool &value)
    {
        if(value && UAV==0)
        {
            UAV=new UAVItem(map,this);
            UAV->setParentItem(map);
            connect(this,SIGNAL(UAVLeftSafetyBouble(internals::PointLatLng)),UAV,SIGNAL(UAVLeftSafetyBouble(internals::PointLatLng)));
            connect(this,SIGNAL(UAVReachedWayPoint(int,WayPointItem*)),UAV,SIGNAL(UAVReachedWayPoint(int,WayPointItem*)));
            UAV->setOpacity(overlayOpacity);
        }
        else if(!value)
        {
            if(UAV!=0)
            {
                delete UAV;
                UAV=NULL;
            }

        }
    }

    void TLMapWidget::SetShowHome(const bool &value)
    {
            Home->setVisible(value);
    }

    void TLMapWidget::resizeEvent(QResizeEvent *event)
    {
        if (scene())
            scene()->setSceneRect(
                    QRect(QPoint(0, 0), event->size()));
        QGraphicsView::resizeEvent(event);

        if(compassRose)
            compassRose->setScale(0.1+0.05*(qreal)(event->size().width())/1000*(qreal)(event->size().height())/600);
        if(windCompass) {
            windCompass->setPos(70 - windCompass->boundingRect().width()/2, this->size().height() - 70 - windCompass->boundingRect().height()/2);
            windspeedTxt->setPos(73 - windCompass->boundingRect().width()/2 * windCompass->scale(), this->size().height() - windCompass->boundingRect().height()/2 * windCompass->scale() - 30);
        }

    }

    QSize TLMapWidget::sizeHint() const
    {
        return map->sizeHint();
    }

    void TLMapWidget::showEvent(QShowEvent *event)
    {
        connect(scene(),SIGNAL(sceneRectChanged(QRectF)),map,SLOT(resize(QRectF)));
        map->start();
        QGraphicsView::showEvent(event);
    }

    TLMapWidget::~TLMapWidget()
    {
        if(UAV)
            delete UAV;
        if(Home)
            delete Home;
        if(map)
            delete map;
        if(core)
            delete core;
        if(configuration)
            delete configuration;
        foreach(QGraphicsItem* i,this->items())
        {
            if(i)
                delete i;
        }
    }

    void TLMapWidget::closeEvent(QCloseEvent *event)
    {
        core->OnMapClose();
        event->accept();
    }

    void TLMapWidget::SetUseOpenGL(const bool &value)
    {
        useOpenGL=value;
        if (useOpenGL)
            setViewport(new QGLWidget(QGLFormat(QGL::SampleBuffers), this));
        else
            setViewport(new QWidget());
        update();
    }

    internals::PointLatLng TLMapWidget::currentMousePosition()
    {
        return currentmouseposition;
    }

    void TLMapWidget::mouseMoveEvent(QMouseEvent *event)
    {
        QGraphicsView::mouseMoveEvent(event);
        QPointF p=event->pos();
        p=map->mapFromParent(p);
        currentmouseposition=map->FromLocalToLatLng(p.x(),p.y());
    }

    ////////////////WAYPOINT////////////////////////
    WayPointItem* TLMapWidget::WPCreate()
    {
        WayPointItem* item=new WayPointItem(this->CurrentPosition(),0,map);
        ConnectWP(item);
        item->setParentItem(map);
        int position=item->Number();
        emit WPCreated(position,item);
        return item;
    }

    WayPointItem* TLMapWidget::magicWPCreate()
    {
        WayPointItem* item=new WayPointItem(map,true);
        item->SetShowNumber(false);
        item->setParentItem(map);
        return item;
    }

    void TLMapWidget::WPCreate(WayPointItem* item)
    {
        ConnectWP(item);
        item->setParentItem(map);
        int position=item->Number();
        emit WPCreated(position,item);
        setOverlayOpacity(overlayOpacity);
    }

    WayPointItem* TLMapWidget::WPCreate(internals::PointLatLng const& coord,int const& altitude)
    {
        WayPointItem* item=new WayPointItem(coord,altitude,map);
        ConnectWP(item);
        item->setParentItem(map);
        int position=item->Number();
        emit WPCreated(position,item);
        setOverlayOpacity(overlayOpacity);
        return item;
    }

    WayPointItem* TLMapWidget::WPCreate(internals::PointLatLng const& coord,int const& altitude, QString const& description)
    {
        WayPointItem* item=new WayPointItem(coord,altitude,description,map);
        ConnectWP(item);
        item->setParentItem(map);
        int position=item->Number();
        emit WPCreated(position,item);
        setOverlayOpacity(overlayOpacity);
        return item;
    }

    WayPointItem* TLMapWidget::WPCreate(const distBearingAltitude &relativeCoord, const QString &description)
    {
        WayPointItem* item=new WayPointItem(relativeCoord,description,map);
        ConnectWP(item);
        item->setParentItem(map);
        int position=item->Number();
        emit WPCreated(position,item);
        setOverlayOpacity(overlayOpacity);
        return item;
    }

    WayPointItem* TLMapWidget::WPInsert(const int &position)
    {
        WayPointItem* item=new WayPointItem(this->CurrentPosition(),0,map);
        item->SetNumber(position);
        ConnectWP(item);
        item->setParentItem(map);
        emit WPInserted(position,item);
        setOverlayOpacity(overlayOpacity);
        return item;
    }

    void TLMapWidget::WPInsert(WayPointItem* item,const int &position)
    {
        item->SetNumber(position);
        ConnectWP(item);
        item->setParentItem(map);
        emit WPInserted(position,item);
        setOverlayOpacity(overlayOpacity);
    }

    WayPointItem* TLMapWidget::WPInsert(internals::PointLatLng const& coord,int const& altitude,const int &position)
    {
        WayPointItem* item=new WayPointItem(coord,altitude,map);
        item->SetNumber(position);
        ConnectWP(item);
        item->setParentItem(map);
        emit WPInserted(position,item);
        setOverlayOpacity(overlayOpacity);
        return item;
    }

    WayPointItem* TLMapWidget::WPInsert(internals::PointLatLng const& coord,int const& altitude, QString const& description,const int &position)
    {
        internals::PointLatLng mcoord;
        bool reloc=false;
        if(coord==internals::PointLatLng(0,0))
        {
            mcoord=CurrentPosition();
            reloc=true;
        }
        else
            mcoord=coord;
        WayPointItem* item=new WayPointItem(mcoord,altitude,description,map);
        item->SetNumber(position);
        ConnectWP(item);
        item->setParentItem(map);
        emit WPInserted(position,item);
        if(reloc)
            emit WPValuesChanged(item);
        setOverlayOpacity(overlayOpacity);
        return item;
    }

    WayPointItem* TLMapWidget::WPInsert(distBearingAltitude const& relative, QString const& description,const int &position)
    {
        WayPointItem* item=new WayPointItem(relative,description,map);
        item->SetNumber(position);
        ConnectWP(item);
        item->setParentItem(map);
        emit WPInserted(position,item);
        setOverlayOpacity(overlayOpacity);
        return item;
    }

    void TLMapWidget::WPDelete(WayPointItem *item)
    {
        emit WPDeleted(item->Number(),item);
        delete item;
    }

    void TLMapWidget::WPDelete(int number)
    {
        foreach(QGraphicsItem* i,map->childItems())
        {
            WayPointItem* w=qgraphicsitem_cast<WayPointItem*>(i);
            if(w)
            {
                if(w->Number()==number)
                {
                    emit WPDeleted(w->Number(),w);
                    delete w;
                    return;
                }
            }
        }
    }

    WayPointItem * TLMapWidget::WPFind(int number)
    {
        foreach(QGraphicsItem* i,map->childItems())
        {
            WayPointItem* w=qgraphicsitem_cast<WayPointItem*>(i);
            if(w)
            {
                if(w->Number()==number)
                {
                    return w;
                }
            }
        }
        return NULL;
    }

    void TLMapWidget::WPSetVisibleAll(bool value)
    {
        foreach(QGraphicsItem* i,map->childItems())
        {
            WayPointItem* w=qgraphicsitem_cast<WayPointItem*>(i);
            if(w)
            {
                if(w->Number()!=-1)
                    w->setVisible(value);
            }
        }
    }

    void TLMapWidget::WPDeleteAll()
    {
        foreach(QGraphicsItem* i,map->childItems())
        {
            WayPointItem* w=qgraphicsitem_cast<WayPointItem*>(i);
            if(w)
            {
                if(w->Number()!=-1)
                {
                    emit WPDeleted(w->Number(),w);
                    delete w;
                }
            }
        }
    }

    bool TLMapWidget::WPPresent()
    {
        foreach(QGraphicsItem* i,map->childItems())
        {
            WayPointItem* w=qgraphicsitem_cast<WayPointItem*>(i);
            if(w)
            {
                if(w->Number()!=-1)
                {
                    return true;
                }
            }
        }
        return false;
    }

    void TLMapWidget::deleteAllOverlays()
    {
        foreach(QGraphicsItem* i,map->childItems())
        {
            MapLine* w=qgraphicsitem_cast<MapLine*>(i);
            if(w)
                w->deleteLater();
            else
            {
                MapCircle* ww=qgraphicsitem_cast<MapCircle*>(i);
                if(ww)
                    ww->deleteLater();
            }
        }
    }

    QList<WayPointItem*> TLMapWidget::WPSelected()
    {
        QList<WayPointItem*> list;
        foreach(QGraphicsItem* i,scene()->selectedItems())
        {
            WayPointItem* w=qgraphicsitem_cast<WayPointItem*>(i);
            if(w)
                list.append(w);
        }
        return list;
    }

    void TLMapWidget::WPRenumber(WayPointItem *item, const int &newnumber)
    {
        item->SetNumber(newnumber);
    }

    void TLMapWidget::ConnectWP(WayPointItem *item)
    {
        connect(item,SIGNAL(WPNumberChanged(int,int,WayPointItem*)),this,SIGNAL(WPNumberChanged(int,int,WayPointItem*)),Qt::DirectConnection);
        connect(item,SIGNAL(WPValuesChanged(WayPointItem*)),this,SIGNAL(WPValuesChanged(WayPointItem*)),Qt::DirectConnection);
        connect(item,SIGNAL(manualCoordChange(WayPointItem*)),this,SIGNAL(WPManualCoordChange(WayPointItem*)),Qt::DirectConnection);
        connect(this,SIGNAL(WPInserted(int,WayPointItem*)),item,SLOT(WPInserted(int,WayPointItem*)),Qt::DirectConnection);
        connect(this,SIGNAL(WPNumberChanged(int,int,WayPointItem*)),item,SLOT(WPRenumbered(int,int,WayPointItem*)),Qt::DirectConnection);
        connect(this,SIGNAL(WPDeleted(int,WayPointItem*)),item,SLOT(WPDeleted(int,WayPointItem*)),Qt::DirectConnection);
    }

    void TLMapWidget::diagRefresh()
    {
        if(showDiag)
        {
            if(diagGraphItem==0)
            {
                diagGraphItem=new QGraphicsTextItem();
                scene()->addItem(diagGraphItem);
                diagGraphItem->setPos(10,100);
                diagGraphItem->setZValue(3);
                diagGraphItem->setFlag(QGraphicsItem::ItemIsMovable,true);
                diagGraphItem->setDefaultTextColor(Qt::yellow);
            }
            diagGraphItem->setPlainText(core->GetDiagnostics().toString());
        }
        else
            if(diagGraphItem!=0)
            {
            delete diagGraphItem;
            diagGraphItem=0;
        }
    }

    //////////////////////////////////////////////
    /**
     * @brief OPMapWidget::SetShowCompassRose Shows the compass rose on the map.
     * @param value If true the compass rose is enabled. If false it is disabled.
     */
    void TLMapWidget::SetShowCompassRose(const bool &value)
    {
        if(value && !compassRose)
        {
            compassRose=new QGraphicsSvgItem(QString::fromUtf8(":/markers/images/compass.svg"));
            compassRose->setScale(0.1+0.05*(qreal)(this->size().width())/1000*(qreal)(this->size().height())/600);
            compassRose->setFlag(QGraphicsItem::ItemIsMovable,false);
            compassRose->setFlag(QGraphicsItem::ItemIsSelectable,false);
            scene()->addItem(compassRose);
            compassRose->setTransformOriginPoint(compassRose->boundingRect().width()/2,compassRose->boundingRect().height()/2);
            compassRose->setPos(55-compassRose->boundingRect().width()/2,55-compassRose->boundingRect().height()/2);
            compassRose->setZValue(3);
            compassRose->setOpacity(0.7);
        }
        if(!value && compassRose)
        {
            delete compassRose;
            compassRose=0;
        }
    }

    /**
     * @brief OPMapWidget::SetShowWindCompass Shows the compass rose on the map.
     * @param value If true the compass is enabled. If false it is disabled.
     */
    void TLMapWidget::SetShowWindCompass(const bool &value)
    {
        if (value && !windCompass) {
            windCompass=new QGraphicsSvgItem(QString::fromUtf8(":/markers/images/wind_compass.svg"));
            windCompass->setScale(120/windCompass->boundingRect().width()); // A constant 120 pixels large
            windCompass->setFlag(QGraphicsItem::ItemIsMovable, false);
            windCompass->setFlag(QGraphicsItem::ItemIsSelectable, false);
            windCompass->setTransformOriginPoint(windCompass->boundingRect().width()/2, windCompass->boundingRect().height()/2);
            windCompass->setZValue(compassRose->zValue() + 1);
            windCompass->setOpacity(0.70);
            scene()->addItem(windCompass);

            // Add text
            windspeedTxt = new QGraphicsTextItem();
            windspeedTxt->setDefaultTextColor(QColor("Black"));
            windspeedTxt->setZValue(compassRose->zValue() + 2);

            scene()->addItem(windspeedTxt);

            // Reset and position
            double dummyWind[3] = {0,0,0};
            setWindVelocity(dummyWind);
            windCompass->setPos(70 - windCompass->boundingRect().width()/2, this->size().height() - 70 - windCompass->boundingRect().height()/2);
            windspeedTxt->setPos(73 - windCompass->boundingRect().width()/2 * windCompass->scale(), this->size().height() - windCompass->boundingRect().height()/2 * windCompass->scale() - 30);
        }

        if (!value && windCompass) {
            delete windspeedTxt;
            delete windCompass;
            windCompass=0;
        }
    }

    void TLMapWidget::setWindVelocity(double windVelocity_NED[3])
    {
        double windAngle_D = atan2(windVelocity_NED[1], windVelocity_NED[0]) * RAD2DEG;
        if (windAngle_D < 0) // Wrap to [0,360)
            windAngle_D = windAngle_D + 360;

        if (windspeedTxt != NULL)
            windspeedTxt->setPlainText(QString("%1%2 @ %3m/s\nsink: %4m/s").arg(windAngle_D, 3, 'f', 0, QChar(0x30)).arg(QChar(0x00B0)).arg(sqrt(pow(windVelocity_NED[0], 2) + pow(windVelocity_NED[1], 2)), 3, 'f', 1).arg(windVelocity_NED[2], 0, 'f', 1)); // 0x00B0 is unicode for the degree symbol.

        windCompass->setRotation(windAngle_D);
    }


    void TLMapWidget::setOverlayOpacity(qreal value)
    {
        map->setOverlayOpacity(value);
        overlayOpacity=value;
    }

    void TLMapWidget::SetRotate(qreal const& value)
    {
        map->mapRotate(value);
        if(compassRose && (compassRose->rotation() != value)) {
            compassRose->setRotation(value);
        }
        if(windCompass && (windCompass->rotation() != value)) {
            windCompass->setRotation(value);
        }
    }

    void TLMapWidget::RipMap()
    {
        new MapRipper(core,map->SelectedArea());
    }

    void TLMapWidget::setSelectedWP(QList<WayPointItem * >list)
    {
        this->scene()->clearSelection();
        foreach(WayPointItem * wp,list)
        {
            wp->setSelected(true);
        }
    }

    void TLMapWidget::OnSelectionChanged()
    {
        QList<QGraphicsItem*> list;
        QList<WayPointItem*> wplist;
        list=this->scene()->selectedItems();
        foreach(QGraphicsItem* item,list)
        {
            WayPointItem * wp=qgraphicsitem_cast<WayPointItem*>(item);
            if(wp)
                wplist.append(wp);
        }
        if(wplist.length()>0)
            emit selectedWPChanged(wplist);
    }
}
