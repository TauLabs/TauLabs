/**
 ******************************************************************************
 *
 * @file       plotdata2d.h
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

#ifndef PLOTDATA2D_H
#define PLOTDATA2D_H

#include "plotdata.h"

#include <QTimer>
#include <QTime>
#include <QVector>


/**
 * @brief The Plot2dData class Base class that keeps the data for each curve in the plot.
 */
class Plot2dData : public PlotData
{
    Q_OBJECT

public:
    Plot2dData(QString uavObject, QString uavField);
    ~Plot2dData();

    QVector<double>* yDataHistory; //Used for scatterplots

    virtual void setUpdatedFlagToTrue(){dataUpdated = true;}
    virtual bool readAndResetUpdatedFlag(){bool tmp = dataUpdated; dataUpdated = false; return tmp;}

private:
    bool dataUpdated;
};

#endif // PLOTDATA2D_H
