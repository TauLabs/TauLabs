/**
 ******************************************************************************
 *
 * @file       plotdata3d.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ScopePlugin Scope Gadget Plugin
 * @{
 * @brief The scope Gadget, graphically plots the states of UAVObjects
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

#ifndef PLOTDATA3D_H
#define PLOTDATA3D_H

#include "plotdata.h"

#include <QTimer>
#include <QTime>
#include <QVector>


/**
 * @brief The Plot3dData class Base class that keeps the data for each curve in the plot.
 */
class Plot3dData : public PlotData
{
    Q_OBJECT

public:
    Plot3dData(QString uavObject, QString uavField);
    ~Plot3dData();

    QVector<double>* zData;
    QVector<double>* zDataHistory;
    QVector<double>* timeDataHistory;

    void setZMinimum(double val){zMinimum=val;}
    void setZMaximum(double val){zMaximum=val;}

    double getZMinimum(){return zMinimum;}
    double getZMaximum(){return zMaximum;}

    virtual void setUpdatedFlagToTrue(){dataUpdated = true;}
    virtual bool readAndResetUpdatedFlag(){bool tmp = dataUpdated; dataUpdated = false; return tmp;}

protected:
    double zMinimum;
    double zMaximum;

private:
    bool dataUpdated;
};



#endif // PLOTDATA3D_H
