/**
******************************************************************************
*
* @file       core.cpp
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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
#include "core.h"

#ifdef DEBUG_CORE
qlonglong internals::Core::debugcounter=0;
#endif

using namespace projections;

namespace internals {
    Core::Core():started(false),MouseWheelZooming(false),currentPosition(0,0),currentPositionPixel(0,0),LastLocationInBounds(-1,-1),sizeOfMapArea(0,0)
            ,minOfTiles(0,0),maxOfTiles(0,0),zoom(0),isDragging(false),TooltipTextPadding(10,10),mapType(MapType::None),loaderLimit(5),maxzoom(21),runningThreads(0)
    {
        mousewheelzoomtype=MouseWheelZoomType::MousePositionAndCenter;
        SetProjection(new MercatorProjection());
        this->setAutoDelete(false);
        ProcessLoadTaskCallback.setMaxThreadCount(10);
        renderOffset=Point(0,0);
        dragPoint=Point(0,0);
        CanDragMap=true;
        tilesToload=0;
        TLMaps::Instance();
    }
    Core::~Core()
    {
        ProcessLoadTaskCallback.waitForDone();
    }

    void Core::run()
    {
        MrunningThreads.lock();
        ++runningThreads;
        MrunningThreads.unlock();
#ifdef DEBUG_CORE
        qlonglong debug;
        Mdebug.lock();
        debug=++debugcounter;
        Mdebug.unlock();
        qDebug()<<"core:run"<<" ID="<<debug;
#endif //DEBUG_CORE
        bool last = false;

        LoadTask task;

        MtileLoadQueue.lock();
        {
            if(tileLoadQueue.count() > 0)
            {
                task = tileLoadQueue.dequeue();
                {

                    last = (tileLoadQueue.count() == 0);
#ifdef DEBUG_CORE
                    qDebug()<<"TileLoadQueue: " << tileLoadQueue.count()<<" Point:"<<task.Pos.ToString()<<" ID="<<debug;;
#endif //DEBUG_CORE
                }
            }
        }
        MtileLoadQueue.unlock();

        if(task.HasValue())
            if(loaderLimit.tryAcquire(1,TLMaps::Instance()->Timeout))
            {
            MtileToload.lock();
            --tilesToload;
            MtileToload.unlock();
#ifdef DEBUG_CORE
            qDebug()<<"loadLimit semaphore aquired "<<loaderLimit.available()<<" ID="<<debug<<" TASK="<<task.Pos.ToString()<<" "<<task.Zoom;
#endif //DEBUG_CORE

            {

#ifdef DEBUG_CORE
                qDebug()<<"task as value, begining get"<<" ID="<<debug;;
#endif //DEBUG_CORE
                {
                    Tile* m = Matrix.TileAt(task.Pos);

                    if(m==0 || m->Overlays.count() == 0)
                    {
#ifdef DEBUG_CORE
                        qDebug()<<"Fill empty TileMatrix: " + task.ToString()<<" ID="<<debug;;
#endif //DEBUG_CORE

                        Tile* t = new Tile(task.Zoom, task.Pos);
                        QVector<MapType::Types> layers= TLMaps::Instance()->GetAllLayersOfType(GetMapType());

                        foreach(MapType::Types tl,layers)
                        {
                            int retry = 0;

                            do
                            {
                                QByteArray tileImage;

                                // tile number inversion(BottomLeft -> TopLeft) for pergo maps
                                if(tl == MapType::PergoTurkeyMap)
                                {
                                    tileImage = TLMaps::Instance()->GetImageFromServer(tl, Point(task.Pos.X(), maxOfTiles.Height() - task.Pos.Y()), task.Zoom);
                                }
                                else if(tl == MapType::UserImage)
                                {
                                    tileImage = TLMaps::Instance()->GetImageFromFile(tl, task.Pos, task.Zoom, userImageHorizontalScale, userImageVerticalScale, userImageLocation, Projection());
                                }
                                else // ok
                                {
#ifdef DEBUG_CORE
                                    qDebug()<<"start getting image"<<" ID="<<debug;
#endif //DEBUG_CORE
                                    tileImage = TLMaps::Instance()->GetImageFromServer(tl, task.Pos, task.Zoom);
#ifdef DEBUG_CORE
                                    qDebug()<<"Core::run:gotimage size:"<<tileImage.count()<<" ID="<<debug;
#endif //DEBUG_CORE
                                }

                                if(tileImage.length()!=0)
                                {
                                    Moverlays.lock();
                                    {
                                        t->Overlays.append(tileImage);
#ifdef DEBUG_CORE
                                        qDebug()<<"Core::run append tileImage:"<<tileImage.length()<<" to tile:"<<t->GetPos().ToString()<<" now has "<<t->Overlays.count()<<" overlays"<<" ID="<<debug;
#endif //DEBUG_CORE

                                    }
                                    Moverlays.unlock();

                                    break;
                                }
                                else if(TLMaps::Instance()->RetryLoadTile > 0)
                                {
#ifdef DEBUG_CORE
                                    qDebug()<<"ProcessLoadTask: " << task.ToString()<< " -> empty tile, retry " << retry<<" ID="<<debug;;
#endif //DEBUG_CORE
                                    {
                                        QWaitCondition wait;
                                        QMutex m;
                                        m.lock();
                                        wait.wait(&m,500);
                                    }
                                }
                            }
                            while((++retry < TLMaps::Instance()->RetryLoadTile) && (tl == MapType::UserImage));
                        }

                        if(t->Overlays.count() > 0)
                        {
                            Matrix.SetTileAt(task.Pos,t);
                            emit OnNeedInvalidation();

#ifdef DEBUG_CORE
                            qDebug()<<"Core::run add tile "<<t->GetPos().ToString()<<" to matrix index "<<task.Pos.ToString()<<" ID="<<debug;
                            qDebug()<<"Core::run matrix index "<<task.Pos.ToString()<<" as tile with "<<Matrix.TileAt(task.Pos)->Overlays.count()<<" ID="<<debug;
#endif //DEBUG_CORE
                        }
                        else
                        {
                            // emit OnTilesStillToLoad(tilesToload);

                            delete t;
                            t = 0;
                            emit OnNeedInvalidation();
                        }

                        // layers = null;
                    }
                }


                {
                    // last buddy cleans stuff ;}
                    if(last)
                    {
                        TLMaps::Instance()->kiberCacheLock.lockForWrite();
                        TLMaps::Instance()->TilesInMemory.RemoveMemoryOverload();
                        TLMaps::Instance()->kiberCacheLock.unlock();

                        MtileDrawingList.lock();
                        {
                            Matrix.ClearPointsNotIn(tileDrawingList);
                        }
                        MtileDrawingList.unlock();


                        emit OnTileLoadComplete();


                        emit OnNeedInvalidation();

                    }
                }



            }
#ifdef DEBUG_CORE
            qDebug()<<"loaderLimit release:"+loaderLimit.available()<<" ID="<<debug;
#endif
            emit OnTilesStillToLoad(tilesToload<0? 0:tilesToload);
            loaderLimit.release();
        }
        MrunningThreads.lock();
        --runningThreads;
        MrunningThreads.unlock();
    }
    diagnostics Core::GetDiagnostics()
    {
        MrunningThreads.lock();
        diag=TLMaps::Instance()->GetDiagnostics();
        diag.runningThreads=runningThreads;
        MrunningThreads.unlock();
        return diag;
    }

    void Core::SetZoom(const int &value)
    {
        if (!isDragging)
        {

            zoom=value;
            minOfTiles=Projection()->GetTileMatrixMinXY(value);
            maxOfTiles=Projection()->GetTileMatrixMaxXY(value);
            currentPositionPixel=Projection()->FromLatLngToPixel(currentPosition, value);
            if(started)
            {
                MtileLoadQueue.lock();
                tileLoadQueue.clear();
                MtileLoadQueue.unlock();
                MtileToload.lock();
                tilesToload=0;
                MtileToload.unlock();
                Matrix.Clear();
                GoToCurrentPositionOnZoom();
                UpdateBounds();
                keepInBounds();
                emit OnMapDrag();
                emit OnMapZoomChanged();
                emit OnNeedInvalidation();
            }
        }
    }

    void Core::SetCurrentPosition(const PointLatLng &value)
    {
        if(!IsDragging())
        {
            currentPosition = value;
            SetCurrentPositionGPixel(Projection()->FromLatLngToPixel(value, Zoom()));

            if(started)
            {
                GoToCurrentPosition();
                emit OnCurrentPositionChanged(currentPosition);
            }
        }
        else
        {
            currentPosition = value;
            SetCurrentPositionGPixel(Projection()->FromLatLngToPixel(value, Zoom()));

            if(started)
            {
                emit OnCurrentPositionChanged(currentPosition);
            }
        }
    }

    void Core::SetUserImageHorizontalScale(double hScale)
    {
        userImageHorizontalScale=hScale;
    }
    void Core::SetUserImageVerticalScale(double vScale)
    {
        userImageVerticalScale=vScale;
    }
    void Core::SetUserImageLocation(QString mapLocation)
    {
        userImageLocation=mapLocation;
    }

    void Core::SetMapType(const MapType::Types &value)
    {

        if(value != GetMapType())
        {
            mapType = value;

            switch(value)
            {


            case MapType::ArcGIS_Map:
            case MapType::ArcGIS_Satellite:
            case MapType::ArcGIS_ShadedRelief:
            case MapType::ArcGIS_Terrain:
                {
                    if(Projection()->Type()!="PlateCarreeProjection")
                    {
                        SetProjection(new PlateCarreeProjection());
                    }
                    maxzoom=13;
                }
                break;
            case MapType::ArcGIS_MapsLT_Map_Hybrid:
            case MapType::ArcGIS_MapsLT_Map_Labels:
            case MapType::ArcGIS_MapsLT_Map:
            case MapType::ArcGIS_MapsLT_OrtoFoto:
                {
                    if(Projection()->Type()!="LKS94Projection")
                    {
                        SetProjection(new LKS94Projection());
                    }
                    maxzoom=11;
                }
                break;

            case MapType::PergoTurkeyMap:
                {
                    if(Projection()->Type()!="PlateCarreeProjectionPergo")
                    {
                        SetProjection(new PlateCarreeProjectionPergo());
                    }
                    maxzoom=17;
                }
                break;

            case MapType::YandexMapRu:
                {
                    if(Projection()->Type()!="MercatorProjectionYandex")
                    {
                        SetProjection(new MercatorProjectionYandex());
                    }
                    maxzoom=13;
                }
                break;

            case MapType::UserImage:
                {
                    if(Projection()->Type()!="MercatorProjection")
                    {
                        SetProjection(new MercatorProjection());
                    }
                    maxzoom=32;
                }
                break;
            default:
                {
                    if(Projection()->Type()!="MercatorProjection")
                    {
                        SetProjection(new MercatorProjection());
                    }
                    maxzoom=21;
                }
                break;
            }

            //Calculate the number of bits required to hold the tile size
            quint8 numBits=1;
            while( (1 << (numBits-1)) < projection->TileSize().Width()){
                numBits++;
            }
			
            //Ensure that no matter what the zoom can never exceed the number of bits required to display it
            if (numBits + (quint8)maxzoom > sizeof(((core::Point *) 0)->X())*8 - 1){ //Remove one because of the sign bit.
                maxzoom = sizeof(((core::Point *) 0)->X())*8 - 1 - numBits;
            }
			
            minOfTiles = Projection()->GetTileMatrixMinXY(Zoom());
            maxOfTiles = Projection()->GetTileMatrixMaxXY(Zoom());
            SetCurrentPositionGPixel(Projection()->FromLatLngToPixel(CurrentPosition(), Zoom()));

            if(started)
            {
                CancelAsyncTasks();
                OnMapSizeChanged(Width, Height);
                GoToCurrentPosition();
                ReloadMap();
                GoToCurrentPosition();
                emit OnMapTypeChanged(value);

            }
        }

    }
    void Core::StartSystem()
    {
        if(!started)
        {
            started = true;

            ReloadMap();
            GoToCurrentPosition();
        }
    }

    void Core::UpdateCenterTileXYLocation()
    {
        PointLatLng center = FromLocalToLatLng(Width/2, Height/2);
        Point centerPixel = Projection()->FromLatLngToPixel(center, Zoom());
        centerTileXYLocation = Projection()->FromPixelToTileXY(centerPixel);
    }

    void Core::OnMapSizeChanged(int const& width, int const& height)
    {
        Width = width;
        Height = height;

        sizeOfMapArea.SetWidth(1 + (Width/Projection()->TileSize().Width())/2);
        sizeOfMapArea.SetHeight(1 + (Height/Projection()->TileSize().Height())/2);

        UpdateCenterTileXYLocation();

        if(started)
        {
            UpdateBounds();

            emit OnCurrentPositionChanged(currentPosition);
        }
    }
    void Core::OnMapClose()
    {
        CancelAsyncTasks();
    }

    QList<UrlFactory::geoCodingStruct> Core::GetAddressesFromCoordinates(PointLatLng coord,GeoCoderStatusCode::Types &status)
    {
        return TLMaps::Instance()->GetPlacemarkFromGeocoder(coord,status,LanguageType().toShortString(TLMaps::Instance()->GetLanguage()));
    }

    QList<UrlFactory::geoCodingStruct> Core::GetCoordinatesFromAddress(QString const &address,GeoCoderStatusCode::Types &status)
    {
        return TLMaps::Instance()->GetLatLngFromGeodecoder(address,status,LanguageType().toShortString(TLMaps::Instance()->GetLanguage()));
    }

    double Core::GetElevationFromCoordinates(PointLatLng coord,GeoCoderStatusCode::Types &status)
    {
        return TLMaps::Instance()->GetElevationFromCoordinate(coord,status);
    }

    GeoCoderStatusCode::Types Core::SetCurrentPositionByKeywords(QString const& keys)
    {
        GeoCoderStatusCode::Types status = GeoCoderStatusCode::UNKNOWN_ERROR;
        QList <UrlFactory::geoCodingStruct> ret=TLMaps::Instance()->GetLatLngFromGeodecoder(keys, status,LanguageType().toShortString(TLMaps::Instance()->GetLanguage()));
        if((ret.length() > 0) && (status == GeoCoderStatusCode::OK))
        {
            PointLatLng pos = ret.at(0).coordinates;
            SetCurrentPosition(pos);
        }
        return status;
    }

    RectLatLng Core::CurrentViewArea()
    {
        PointLatLng p = Projection()->FromPixelToLatLng(-renderOffset.X(), -renderOffset.Y(), Zoom());
        double rlng = Projection()->FromPixelToLatLng(-renderOffset.X() + Width, -renderOffset.Y(), Zoom()).Lng();
        double blat = Projection()->FromPixelToLatLng(-renderOffset.X(), -renderOffset.Y() + Height, Zoom()).Lat();
        return RectLatLng::FromLTRB(p.Lng(), p.Lat(), rlng, blat);

    }
    PointLatLng Core::FromLocalToLatLng(qint64 const& x, qint64 const& y)
    {
        return Projection()->FromPixelToLatLng(Point(x - renderOffset.X(), y - renderOffset.Y()), Zoom());
    }


    Point Core::FromLatLngToLocal(PointLatLng const& latlng)
    {
        Point pLocal = Projection()->FromLatLngToPixel(latlng, Zoom());
        pLocal.Offset(renderOffset);
        return pLocal;
    }
    int Core::GetMaxZoomToFitRect(RectLatLng const& rect)
    {
        int zoom = 0;

        for(int i = 1; i <= MaxZoom(); i++)
        {
            Point p1 = Projection()->FromLatLngToPixel(rect.LocationTopLeft(), i);
            Point p2 = Projection()->FromLatLngToPixel(rect.Bottom(), rect.Right(), i);

            if(((p2.X() - p1.X()) <= Width+10) && (p2.Y() - p1.Y()) <= Height+10)
            {
                zoom = i;
            }
            else
            {
                break;
            }
        }

        return zoom;
    }
    void Core::BeginDrag(Point const& pt)
    {
        dragPoint.SetX(pt.X() - renderOffset.X());
        dragPoint.SetY(pt.Y() - renderOffset.Y());
        isDragging = true;
    }
    void Core::EndDrag()
    {
        isDragging = false;
        emit OnNeedInvalidation();

    }
    void Core::ReloadMap()
    {
        if(started)
        {
#ifdef DEBUG_CORE
            qDebug()<<"------------------";
#endif //DEBUG_CORE

            MtileLoadQueue.lock();
            {
                tileLoadQueue.clear();
            }
            MtileLoadQueue.unlock();
            MtileToload.lock();
            tilesToload=0;
            MtileToload.unlock();
            Matrix.Clear();

            emit OnNeedInvalidation();

        }
    }
    void Core::GoToCurrentPosition()
    {
        // reset stuff
        renderOffset = Point::Empty;
        centerTileXYLocationLast = Point::Empty;
        dragPoint = Point::Empty;

        // goto location
        Drag(Point(-(GetcurrentPositionGPixel().X() - Width/2), -(GetcurrentPositionGPixel().Y() - Height/2)));
    }
    void Core::GoToCurrentPositionOnZoom()
    {
        // reset stuff
        renderOffset = Point::Empty;
        centerTileXYLocationLast = Point::Empty;
        dragPoint = Point::Empty;

        // goto location and centering
        if(MouseWheelZooming)
        {
            if(mousewheelzoomtype != MouseWheelZoomType::MousePositionWithoutCenter)
            {
                Point pt = Point(-(GetcurrentPositionGPixel().X() - Width/2), -(GetcurrentPositionGPixel().Y() - Height/2));
                renderOffset.SetX(pt.X() - dragPoint.X());
                renderOffset.SetY(pt.Y() - dragPoint.Y());
            }
            else // without centering
            {
                renderOffset.SetX(-GetcurrentPositionGPixel().X() - dragPoint.X());
                renderOffset.SetY(-GetcurrentPositionGPixel().Y() - dragPoint.Y());
                renderOffset.Offset(mouseLastZoom);
            }
        }
        else // use current map center
        {
            mouseLastZoom = Point::Empty;

            Point pt = Point(-(GetcurrentPositionGPixel().X() - Width/2), -(GetcurrentPositionGPixel().Y() - Height/2));
            renderOffset.SetX(pt.X() - dragPoint.X());
            renderOffset.SetY(pt.Y() - dragPoint.Y());

        }

        UpdateCenterTileXYLocation();
    }
    void Core::DragOffset(Point const& offset)
    {
        renderOffset.Offset(offset);

        UpdateCenterTileXYLocation();

        if(centerTileXYLocation != centerTileXYLocationLast)
        {
            centerTileXYLocationLast = centerTileXYLocation;
            UpdateBounds();
        }

        {
            LastLocationInBounds = CurrentPosition();
            SetCurrentPosition (FromLocalToLatLng((qint64) Width/2, (qint64) Height/2));
        }

        emit OnNeedInvalidation();
        emit OnMapDrag();
    }
    void Core::Drag(Point const& pt)
    {
        renderOffset.SetX(pt.X() - dragPoint.X());
        renderOffset.SetY(pt.Y() - dragPoint.Y());
        keepInBounds();
        UpdateCenterTileXYLocation();

        if(centerTileXYLocation != centerTileXYLocationLast)
        {
            centerTileXYLocationLast = centerTileXYLocation;
            UpdateBounds();
        }

        if(IsDragging())
        {
            LastLocationInBounds = CurrentPosition();
            SetCurrentPosition(FromLocalToLatLng((qint64) Width/2, (qint64) Height/2));
        }

        emit OnNeedInvalidation();


        emit OnMapDrag();
    }
    void Core::CancelAsyncTasks()
    {
        if(started)
        {
            ProcessLoadTaskCallback.waitForDone();
            MtileLoadQueue.lock();
            {
                tileLoadQueue.clear();
                //tilesToload=0;
            }
            MtileLoadQueue.unlock();
            MtileToload.lock();
            tilesToload=0;
            MtileToload.unlock();
            //  ProcessLoadTaskCallback.waitForDone();
        }
    }
    void Core::UpdateBounds()
    {
        MtileDrawingList.lock();
        {
            FindTilesAround(tileDrawingList);

#ifdef DEBUG_CORE
            qDebug()<<"OnTileLoadStart: " << tileDrawingList.count() << " tiles to load at zoom " << Zoom() << ", time: " << QDateTime::currentDateTime().date();
#endif //DEBUG_CORE

            emit OnTileLoadStart();


            foreach(Point p,tileDrawingList)
            {
                LoadTask task = LoadTask(p, Zoom());
                {
                    MtileLoadQueue.lock();
                    {
                        if(!tileLoadQueue.contains(task))
                        {
                            MtileToload.lock();
                            ++tilesToload;
                            MtileToload.unlock();
                            tileLoadQueue.enqueue(task);
#ifdef DEBUG_CORE
                            qDebug()<<"Core::UpdateBounds new Task"<<task.Pos.ToString();
#endif //DEBUG_CORE
                            ProcessLoadTaskCallback.start(this);
                        }
                    }
                    MtileLoadQueue.unlock();
                }

            }
        }
        MtileDrawingList.unlock();
        UpdateGroundResolution();
    }
    void Core::FindTilesAround(QList<Point> &list)
    {
        list.clear();;
        for(int i = -sizeOfMapArea.Width(); i <= sizeOfMapArea.Width(); i++)
        {
            for(int j = -sizeOfMapArea.Height(); j <= sizeOfMapArea.Height(); j++)
            {
                Point p = centerTileXYLocation;
                p.SetX(p.X() + i);
                p.SetY(p.Y() + j);

                //if(p.X < minOfTiles.Width)
                //{
                //   p.X += (maxOfTiles.Width + 1);
                //}

                //if(p.X > maxOfTiles.Width)
                //{
                //   p.X -= (maxOfTiles.Width + 1);
                //}

                if(p.X() >= minOfTiles.Width() && p.Y() >= minOfTiles.Height() && p.X() <= maxOfTiles.Width() && p.Y() <= maxOfTiles.Height())
                {
                    if(!list.contains(p))
                    {
                        list.append(p);
                    }
                }
            }
        }


    }
    void Core::UpdateGroundResolution()
    {
        double rez = Projection()->GetGroundResolution(Zoom(), CurrentPosition().Lat());
        pxRes100m =   (int) (100.0 / rez); // 100 meters
        pxRes1000m =  (int) (1000.0 / rez); // 1km
        pxRes10km =   (int) (10000.0 / rez); // 10km
        pxRes100km =  (int) (100000.0 / rez); // 100km
        pxRes1000km = (int) (1000000.0 / rez); // 1000km
        pxRes5000km = (int) (5000000.0 / rez); // 5000km
    }
    /**
     * @brief Core::keepInBounds Saturate renderOffest. The lower bound is (0,0), and the upper bound is ???
     */
    void Core::keepInBounds()
    {
        if(renderOffset.X()>0)
            renderOffset.SetX(0);
        if(renderOffset.Y()>0)
            renderOffset.SetY(0);

        qint64 maxDragY=GetCurrentRegion().Height()-GetTileRect().Height()*(maxOfTiles.Height()-minOfTiles.Height()+1);
        qint64 maxDragX=GetCurrentRegion().Width()-GetTileRect().Width()*(maxOfTiles.Width()-minOfTiles.Width()+1);

        if(maxDragY>renderOffset.Y())
            renderOffset.SetY(maxDragY);
        if(maxDragX>renderOffset.X())
            renderOffset.SetX(maxDragX);

    }
}
