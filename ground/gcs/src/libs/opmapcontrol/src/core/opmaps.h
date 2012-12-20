/**
******************************************************************************
*
* @file       OPMaps.h
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
#ifndef OPMaps_H
#define OPMaps_H


#include "debugheader.h"
#include "memorycache.h"
#include "rawtile.h"
#include "cache.h"
#include "accessmode.h"
#include "languagetype.h"
#include "cacheitemqueue.h"
#include "tilecachequeue.h"
#include "pureimagecache.h"
#include "alllayersoftype.h"
#include "urlfactory.h"
#include "diagnostics.h"

#include "../internals/pureprojection.h"
#include "../internals/projections/lks94projection.h"
#include "../internals/projections/mercatorprojection.h"
#include "../internals/projections/mercatorprojectionyandex.h"
#include "../internals/projections/platecarreeprojection.h"
#include "../internals/projections/platecarreeprojectionpergo.h"

namespace core {
    class OPMaps: public MemoryCache,public AllLayersOfType,public UrlFactory
    {


    public:

        ~OPMaps();

        static OPMaps* Instance();
        bool ImportFromGMDB(const QString &file);
        bool ExportToGMDB(const QString &file);
        /// <summary>
        /// timeout for map connections
        /// </summary>


        QByteArray GetImageFromServer(const MapType::Types &type,const core::Point &pos,const int &zoom);
        QByteArray GetImageFromFile(const MapType::Types &type,const core::Point &pos,const int &zoom, double hScale, double vScale, QString userImageFileName, internals::PureProjection *projection);
        bool UseMemoryCache(){return useMemoryCache;}//TODO
        void setUseMemoryCache(const bool& value){useMemoryCache=value;}
        void setLanguage(const LanguageType::Types& language){Language=language;}//TODO
        LanguageType::Types GetLanguage(){return Language;}//TODO
        AccessMode::Types GetAccessMode()const{return accessmode;}
        void setAccessMode(const AccessMode::Types& mode){accessmode=mode;}
        int RetryLoadTile;
        diagnostics GetDiagnostics();

    private:
        bool useMemoryCache;
        LanguageType::Types Language;
        AccessMode::Types accessmode;
        //  PureImageCache ImageCacheLocal;//TODO Criar acesso Get Set
        TileCacheQueue TileDBcacheQueue;
        OPMaps();
        OPMaps(OPMaps const&){}
        OPMaps& operator=(OPMaps const&){ return *this; }
        static OPMaps* m_pInstance;
        diagnostics diag;
        QMutex errorvars;
        quint8 lastZoom;
        int quadCoordRight;
        int quadCoordBottom;
        QImage imScaled;
        int leastCommonZoom;
    protected:
        // MemoryCache TilesInMemory;



    };

}
#endif // OPMaps_H
