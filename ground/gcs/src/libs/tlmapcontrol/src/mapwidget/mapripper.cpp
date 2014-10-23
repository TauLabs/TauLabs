/**
******************************************************************************
*
* @file       mapripper.cpp
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
* @brief      A class that allows ripping of a selection of the map
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
#include "mapripper.h"
namespace mapcontrol
{

MapRipper::MapRipper(internals::Core * core, const internals::RectLatLng & rect):sleep(100),cancel(false),progressForm(0),core(core),yesToAll(false)
    {
        if(!rect.IsEmpty())
        {
            type=core->GetMapType();
            progressForm=new MapRipForm;
            connect(progressForm,SIGNAL(cancelRequest()),this,SLOT(stopFetching()));
            area=rect;
            zoom=core->Zoom();
            maxzoom=core->MaxZoom();
            points=core->Projection()->GetAreaTileList(area,zoom,0);
            this->start();
            cancel=false;
            progressForm->show();
            connect(this,SIGNAL(percentageChanged(int)),progressForm,SLOT(SetPercentage(int)));
            connect(this,SIGNAL(numberOfTilesChanged(int,int)),progressForm,SLOT(SetNumberOfTiles(int,int)));
            connect(this,SIGNAL(providerChanged(QString,int)),progressForm,SLOT(SetProvider(QString,int)));
            connect(this,SIGNAL(finished()),this,SLOT(finish()));
            emit numberOfTilesChanged(0,0);
        }
        else
#ifdef Q_OS_DARWIN
            QMessageBox::information(new QWidget(),"No valid selection","This pre-caches map data.\n\nPlease first select the area of the map to rip with <COMMAND>+Left mouse click");
#else
            QMessageBox::information(new QWidget(),"No valid selection","This pre-caches map data.\n\nPlease first select the area of the map to rip with <CTRL>+Left mouse click");
#endif
    }
void MapRipper::finish()
{
    if(zoom<maxzoom && !cancel)
    {
        ++zoom;
        int ret;
        if(!yesToAll)
        {
            QMessageBox msgBox;
            msgBox.setText(QString("Continue Ripping at zoom level %1?").arg(zoom));
            // msgBox.setInformativeText("Do you want to save your changes?");
            msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No | QMessageBox::YesAll);
            msgBox.setDefaultButton(QMessageBox::Yes);
            ret = msgBox.exec();
        }
        else
            ret=QMessageBox::Yes;
        if(ret==QMessageBox::Yes)
        {
            points.clear();
            points=core->Projection()->GetAreaTileList(area,zoom,0);
            this->start();
        }
        else if(ret==QMessageBox::YesAll)
        {
            yesToAll=true;
            points.clear();
            points=core->Projection()->GetAreaTileList(area,zoom,0);
            this->start();
        }
        else
        {
            progressForm->close();
            delete progressForm;
            this->deleteLater();
        }
    }
    else
    {
        yesToAll=false;
        progressForm->close();
        delete progressForm;
        this->deleteLater();
    }
}


    void MapRipper::run()
    {
        int countOk = 0;
        bool goodtile=false;
        //  Stuff.Shuffle<Point>(ref list);
        QVector<core::MapType::Types> types = TLMaps::Instance()->GetAllLayersOfType(type);
        int all=points.count();
        for(int i = 0; i < all; i++)
        {
            emit numberOfTilesChanged(all,i+1);
            if(cancel)
                break;

            core::Point p = points[i];
            {
                //qDebug()<<"offline fetching:"<<p.ToString();
                foreach(core::MapType::Types type,types)
                {
                    emit providerChanged(core::MapType::StrByType(type),zoom);
                    QByteArray img = TLMaps::Instance()->GetImageFromServer(type, p, zoom);
                    if(img.length()!=0)
                    {
                        goodtile=true;
                        img=NULL;
                    }
                    else
                        goodtile=false;
                }
                if(goodtile)
                {
                    countOk++;
                }
                else
                {
                    i--;
                    QThread::msleep(1000);
                    continue;
                }
            }
            emit percentageChanged((int) ((i+1)*100/all));//, i+1);
            // worker.ReportProgress((int) ((i+1)*100/all), i+1);

            QThread::msleep(sleep);
        }
    }

    void MapRipper::stopFetching()
    {
        QMutexLocker locker(&mutex);
        cancel=true;
    }
}
