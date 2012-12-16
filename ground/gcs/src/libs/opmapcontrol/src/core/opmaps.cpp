/**
******************************************************************************
*
* @file       OPMaps.cpp
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
#include "opmaps.h"
#include "extensionsystem/pluginmanager.h"
#include "utils/coordinateconversions.h"
#include "QDebug"
#include "QPainter"

//#define DEBUG_Q_TILES

using namespace projections;

namespace core {
    OPMaps* OPMaps::m_pInstance=0;

    OPMaps* OPMaps::Instance()
    {
        if(!m_pInstance)
            m_pInstance=new OPMaps;
        return m_pInstance;
    }
    OPMaps::OPMaps():
        RetryLoadTile(2),useMemoryCache(true),lastZoom(0),quadCoordBottom(0),quadCoordRight(0)
    {
        accessmode=AccessMode::ServerAndCache;
        Language=LanguageType::PortuguesePortugal;
        LanguageStr=LanguageType().toShortString(Language);
        Cache::Instance();

    }


    OPMaps::~OPMaps()
    {
        TileDBcacheQueue.wait();
    }

    /**
     * @brief OPMaps::GetImageFromFile
     * @param type Type of map (Google Satellite, Bing, ARCGIS...)
     * @param pos Quadtile to be drawn
     * @param zoom Quadtile zoom level
     * @param hScale Scaling constant in the horizontal axis, in [m/px]
     * @param vScale Scaling constant in the vertical axis, in [m/px]
     * @param userImageFileName
     * @param projection Projection type for the map
     * @return
     */
    QByteArray OPMaps::GetImageFromFile(const MapType::Types &type,const core::Point &pos,const int &zoom, double hScale, double vScale, QString userImageFileName, internals::PureProjection *projection)
    {


#ifdef DEBUG_TIMINGS
        QTime time;
        time.restart();
#endif
#ifdef DEBUG_GMAPS
        qDebug()<<"Entered GetImageFrom";
#endif //DEBUG_GMAPS
        QByteArray ret;

        if(useMemoryCache)
        {
#ifdef DEBUG_GMAPS
            qDebug()<<"Try Tile from memory:Size="<<TilesInMemory.MemoryCacheSize();
#endif //DEBUG_GMAPS
            ret=GetTileFromMemoryCache(RawTile(type,pos,zoom));
            if(!ret.isEmpty())
            {
                errorvars.lock();
                ++diag.tilesFromMem;
                errorvars.unlock();
            }

        }
        if(ret.isEmpty())
        {
#ifdef DEBUG_GMAPS
            qDebug()<<"Tile not in memory";
#endif //DEBUG_GMAPS

            //Attempt to read tile from cache
            if(accessmode != (AccessMode::ServerOnly) && type != MapType::UserImage) //Don't use cache if the user supplies a file. This is because
            {
#ifdef DEBUG_GMAPS
                qDebug()<<"Try tile from DataBase";
#endif //DEBUG_GMAPS
                ret=Cache::Instance()->ImageCache.GetImageFromCache(type,pos,zoom);
                if(!ret.isEmpty())
                {
                    errorvars.lock();
                    ++diag.tilesFromDB;
                    errorvars.unlock();
#ifdef DEBUG_GMAPS
                    qDebug()<<"Tile found in Database";
#endif //DEBUG_GMAPS
                    if(useMemoryCache)
                    {
#ifdef DEBUG_GMAPS
                        qDebug()<<"Add Tile to memory";
#endif //DEBUG_GMAPS
                        AddTileToMemoryCache(RawTile(type,pos,zoom),ret);
                    }
                    return ret;
                }
            }

            //Attempt to read file from original source
            if(accessmode!=AccessMode::CacheOnly)
            {
                //If it's a local user file...
                if (type == MapType::UserImage)
                    {
                        //Load the image
                        static QImage imMap(userImageFileName);

                        //Get image width and height, in [m]
                        static double widthPx=imMap.width();
                        static double heightPx=imMap.height();
                        static double width=widthPx*hScale;
                        static double height=heightPx*vScale;
                        static double cornerLLA[3];
                        static bool once=true;

                        //TODO: Only do this once, not every time
                        if(once){
                            once=false;

                            double homeLLA[3]={0,0,0};
                            double cornerNED[3]={-height, width, 0};
                            Utils::CoordinateConversions().NED2LLA_HomeLLA(homeLLA, cornerNED, cornerLLA);
                        }

                        //Get the tile width
                        int tileSize= projection->TileSize().Width();

                        //Only update is the zoom level has changed
                        if(lastZoom != zoom){
                            lastZoom=zoom;

                            //Generate image scaled for this zoom level
                            imScaled=imMap.scaledToWidth(width / projection->GetGroundResolution(zoom, 0));

                            //Find the quadtile that contains the opposite corner of the image
                            double top=90, bottom=-90, left=-180, right=180;
                            double lat=cornerLLA[0];
                            double lon=cornerLLA[1];
                            quadCoordRight=0;
                            quadCoordBottom=0;
                            for (int i=0; i<zoom; i++)
                            {
                                quadCoordRight <<=1;
                                quadCoordBottom<<=1;
                                if ((left+right)/2<lon	)
                                {
                                    quadCoordRight|=1;
                                    left=(right+left)/2;
                                }
                                else
                                {
                                    right=(right+left)/2;
                                }

                                if ((top+bottom)/2<lat)
                                {
                                    bottom=(top+bottom)/2;
                                }
                                else
                                {
                                    quadCoordBottom |= 1;
                                    top=(top+bottom)/2;
                                }
                            }

                            // Determine the smallest quadtile that contains all the image, as determined by the location of the opposite corner.
                            leastCommonZoom=zoom-1;
                            while((((1<<leastCommonZoom) & quadCoordBottom) == ((1<<leastCommonZoom) & quadCoordRight)) && ((1<<leastCommonZoom) & quadCoordRight) && leastCommonZoom >0 )

                            {
                                leastCommonZoom--;
                            }


                            while((((1<<leastCommonZoom) & quadCoordBottom) == ((1<<leastCommonZoom) & quadCoordRight)) && !((1<<leastCommonZoom) & quadCoordRight) && leastCommonZoom >0 )
                            {
                                leastCommonZoom--;
                            }

                            qDebug() << "LCZ: " << leastCommonZoom;

                        }

                        // Only write output files for the smallest quadtile the contains the image, as determined by the opposite corner
                        if((pos.X() >> leastCommonZoom+1) == (quadCoordRight >> leastCommonZoom+1) && (pos.Y() >> leastCommonZoom+1) == (quadCoordBottom >> leastCommonZoom+1))
                        {

                            QImage retImage=imScaled.copy((pos.X() & (1<<leastCommonZoom+1)-1) * tileSize, (pos.Y() & (1<<leastCommonZoom+1)-1) * tileSize, tileSize, tileSize);

#ifdef DEBUG_Q_TILES
                            //For a silly reason of making sure that everything is properly drawn, display the quadtile element on each tile
                            retImage=retImage.convertToFormat(QImage::Format_ARGB32);
                            QPainter painter(&retImage);
                            painter.setFont(QFont("Chicago", 7)); // The font size
                            painter.setPen(QColor(233, 10, 150));
                            painter.drawText(20, 40, QString::number(pos.X() , 2));
                            painter.drawText(20, 80, QString::number(pos.Y() , 2));
                            painter.drawText(20, 120, QString::number(quadCoordRight , 2));
                            painter.drawText(20, 160, QString::number(quadCoordBottom , 2));
#endif

                            QBuffer buffer(&ret);
                            buffer.open(QIODevice::WriteOnly);
                            retImage.save(&buffer, "PNG"); // writes image into ba in PNG format
                        }
                        else{ //Nothing here, fill it in with black tiles.
                            QImage retImage(tileSize,tileSize, QImage::Format_ARGB32);
                            retImage.fill(Qt::black);

#ifdef DEBUG_Q_TILES
                            //For a silly reason of making sure that everything is properly drawn, display the quadtile element on each tile
                            QPainter painter(&retImage);
                            painter.setFont(QFont("Chicago", 16)); // The font size
                            painter.setPen(QColor(10, 233, 150));
                            painter.drawText(20, 40, QString::number(pos.X() , 2));
                            painter.drawText(20, 80, QString::number(pos.Y() , 2));
                            painter.drawText(20, 120, QString::number(quadCoordRight , 2));
                            painter.drawText(20, 160, QString::number(quadCoordBottom , 2));
#endif
                            QBuffer buffer(&ret);
                            buffer.open(QIODevice::WriteOnly);
                            retImage.save(&buffer, "PNG"); // writes image into ba in PNG format

                        }
                    }

                errorvars.unlock();

                //Save tile to cache
                if (useMemoryCache)
                {
#ifdef DEBUG_GMAPS
                    qDebug()<<"Add Tile to memory cache";
#endif //DEBUG_GMAPS
                    AddTileToMemoryCache(RawTile(type,pos,zoom),ret);
                }

                //Save tile to database
                if(accessmode!=AccessMode::ServerOnly)
                {
#ifdef DEBUG_GMAPS
                    qDebug()<<"Add tile to DataBase";
#endif //DEBUG_GMAPS
                    CacheItemQueue * item=new CacheItemQueue(type,pos,ret,zoom);
                    TileDBcacheQueue.EnqueueCacheTask(item);
                }


            }
        }
#ifdef DEBUG_GMAPS
        qDebug()<<"Entered GetImageFrom";
#endif //DEBUG_GMAPS
        return ret;
    }


    /**
     * @brief OPMaps::GetImageFromServer
     * @param type Type of map (Google Satellite, Bing, ARCGIS...)
     * @param pos Quadtile to be drawn
     * @param zoom Quadtile zoom level
     * @return
     */
    QByteArray OPMaps::GetImageFromServer(const MapType::Types &type,const Point &pos,const int &zoom)
    {

#ifdef DEBUG_TIMINGS
        QTime time;
        time.restart();
#endif
#ifdef DEBUG_GMAPS
        qDebug()<<"Entered GetImageFrom";
#endif //DEBUG_GMAPS
        QByteArray ret;

        if(useMemoryCache)
        {
#ifdef DEBUG_GMAPS
            qDebug()<<"Try Tile from memory:Size="<<TilesInMemory.MemoryCacheSize();
#endif //DEBUG_GMAPS
            ret=GetTileFromMemoryCache(RawTile(type,pos,zoom));
            if(!ret.isEmpty())
            {
                errorvars.lock();
                ++diag.tilesFromMem;
                errorvars.unlock();
            }

        }
        if(ret.isEmpty())
        {
#ifdef DEBUG_GMAPS
            qDebug()<<"Tile not in memory";
#endif //DEBUG_GMAPS

            //Attempt to read tile from cache
            if(accessmode != (AccessMode::ServerOnly) && type != MapType::UserImage) //Don't use cache if the user supplies a file. This is because
            {
#ifdef DEBUG_GMAPS
                qDebug()<<"Try tile from DataBase";
#endif //DEBUG_GMAPS
                ret=Cache::Instance()->ImageCache.GetImageFromCache(type,pos,zoom);
                if(!ret.isEmpty())
                {
                    errorvars.lock();
                    ++diag.tilesFromDB;
                    errorvars.unlock();
#ifdef DEBUG_GMAPS
                    qDebug()<<"Tile found in Database";
#endif //DEBUG_GMAPS
                    if(useMemoryCache)
                    {
#ifdef DEBUG_GMAPS
                        qDebug()<<"Add Tile to memory";
#endif //DEBUG_GMAPS
                        AddTileToMemoryCache(RawTile(type,pos,zoom),ret);
                    }
                    return ret;
                }
            }

            //Attempt to read file from original source
            if(accessmode!=AccessMode::CacheOnly)
            {
                { //Otherwise, we're getting the tiles from the internet
                    QEventLoop q;
                    QNetworkReply *reply;
                    QNetworkRequest qheader;
                    QNetworkAccessManager network;
                    QTimer tT;
                    tT.setSingleShot(true);
                    connect(&network, SIGNAL(finished(QNetworkReply*)),
                            &q, SLOT(quit()));
                    connect(&tT, SIGNAL(timeout()), &q, SLOT(quit()));
                    network.setProxy(Proxy);
    #ifdef DEBUG_GMAPS
                    qDebug()<<"Try Tile from the Internet";
    #endif //DEBUG_GMAPS
    #ifdef DEBUG_TIMINGS
                    qDebug()<<"opmaps before make image url"<<time.elapsed();
    #endif
                    QString url=MakeImageUrl(type,pos,zoom,LanguageStr);
    #ifdef DEBUG_TIMINGS
                    qDebug()<<"opmaps after make image url"<<time.elapsed();
    #endif		//url	"http://vec02.maps.yandex.ru/tiles?l=map&v=2.10.2&x=7&y=5&z=3"	string
                    //"http://map3.pergo.com.tr/tile/02/000/000/007/000/000/002.png"
                    qheader.setUrl(QUrl(url));
                    qheader.setRawHeader("User-Agent",UserAgent);
                    qheader.setRawHeader("Accept","*/*");
                    switch(type)
                    {
                    case MapType::GoogleMap:
                    case MapType::GoogleSatellite:
                    case MapType::GoogleLabels:
                    case MapType::GoogleTerrain:
                    case MapType::GoogleHybrid:
                        {
                            qheader.setRawHeader("Referrer", "http://maps.google.com/");
                        }
                        break;

                    case MapType::GoogleMapChina:
                    case MapType::GoogleSatelliteChina:
                    case MapType::GoogleLabelsChina:
                    case MapType::GoogleTerrainChina:
                    case MapType::GoogleHybridChina:
                        {
                            qheader.setRawHeader("Referrer", "http://ditu.google.cn/");
                        }
                        break;

                    case MapType::BingHybrid:
                    case MapType::BingMap:
                    case MapType::BingSatellite:
                        {
                            qheader.setRawHeader("Referrer", "http://www.bing.com/maps/");
                        }
                        break;

                    case MapType::YahooHybrid:
                    case MapType::YahooLabels:
                    case MapType::YahooMap:
                    case MapType::YahooSatellite:
                        {
                            qheader.setRawHeader("Referrer", "http://maps.yahoo.com/");
                        }
                        break;

                    case MapType::ArcGIS_MapsLT_Map_Labels:
                    case MapType::ArcGIS_MapsLT_Map:
                    case MapType::ArcGIS_MapsLT_OrtoFoto:
                    case MapType::ArcGIS_MapsLT_Map_Hybrid:
                        {
                            qheader.setRawHeader("Referrer", "http://www.maps.lt/map_beta/");
                        }
                        break;

                    case MapType::OpenStreetMapSurfer:
                    case MapType::OpenStreetMapSurferTerrain:
                        {
                            qheader.setRawHeader("Referrer", "http://www.mapsurfer.net/");
                        }
                        break;

                    case MapType::OpenStreetMap:
                    case MapType::OpenStreetOsm:
                        {
                            qheader.setRawHeader("Referrer", "http://www.openstreetmap.org/");
                        }
                        break;

                    case MapType::YandexMapRu:
                        {
                            qheader.setRawHeader("Referrer", "http://maps.yandex.ru/");
                        }
                        break;
                    default:
                        break;
                    }

                    qDebug() << "qheader: " << qheader.url();

                    reply=network.get(qheader);
                    tT.start(Timeout);
                    q.exec();

                    if(!tT.isActive()){
                        errorvars.lock();
                        ++diag.timeouts;
                        errorvars.unlock();
                        return ret;
                    }
                    tT.stop();
                    if( (reply->error()!=QNetworkReply::NoError))
                    {
                        errorvars.lock();
                        ++diag.networkerrors;
                        errorvars.unlock();
                        reply->deleteLater();
                        return ret;
                    }
                    ret=reply->readAll();
                    reply->deleteLater();//TODO can't this be global??
                    if(ret.isEmpty())
                    {
    #ifdef DEBUG_GMAPS
                        qDebug()<<"Invalid Tile";
    #endif //DEBUG_GMAPS
                        errorvars.lock();
                        ++diag.emptytiles;
                        errorvars.unlock();
                        return ret;
                    }
    #ifdef DEBUG_GMAPS
                    qDebug()<<"Received Tile from the Internet";
    #endif //DEBUG_GMAPS
                    errorvars.lock();
                    ++diag.tilesFromNet;
                }

                errorvars.unlock();

                //Save tile to cache
                if (useMemoryCache)
                {
#ifdef DEBUG_GMAPS
                    qDebug()<<"Add Tile to memory cache";
#endif //DEBUG_GMAPS
                    AddTileToMemoryCache(RawTile(type,pos,zoom),ret);
                }

                //Save tile to database
                if(accessmode!=AccessMode::ServerOnly)
                {
#ifdef DEBUG_GMAPS
                    qDebug()<<"Add tile to DataBase";
#endif //DEBUG_GMAPS
                    CacheItemQueue * item=new CacheItemQueue(type,pos,ret,zoom);
                    TileDBcacheQueue.EnqueueCacheTask(item);
                }


            }
        }
#ifdef DEBUG_GMAPS
        qDebug()<<"Entered GetImageFrom";
#endif //DEBUG_GMAPS
        return ret;
    }

    bool OPMaps::ExportToGMDB(const QString &file)
    {
        return Cache::Instance()->ImageCache.ExportMapDataToDB(Cache::Instance()->ImageCache.GtileCache()+QDir::separator()+"Data.qmdb",file);
    }
    bool OPMaps::ImportFromGMDB(const QString &file)
    {
        return Cache::Instance()->ImageCache.ExportMapDataToDB(file,Cache::Instance()->ImageCache.GtileCache()+QDir::separator()+"Data.qmdb");
    }

    diagnostics OPMaps::GetDiagnostics()
    {
        diagnostics i;
        errorvars.lock();
        i=diag;
        errorvars.unlock();
        return i;
    }
}

