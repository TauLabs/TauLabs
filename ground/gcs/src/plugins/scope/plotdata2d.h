/**
 ******************************************************************************
 *
 * @file       plotdata2d.h
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
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
#include "uavobject.h"

#include "qwt/src/qwt.h"
#include "qwt/src/qwt_color_map.h"
#include "qwt/src/qwt_plot.h"
#include "qwt/src/qwt_plot_curve.h"
#include "qwt/src/qwt_scale_draw.h"
#include "qwt/src/qwt_scale_widget.h"

#include <QTimer>
#include <QTime>
#include <QVector>


/**
 * @brief The Plot2dType enum Defines the different type of plots.
 */
enum Plot2dType {
    NO2DPLOT, //Signifies that there is no 2D plot configured
    SCATTERPLOT2D,
    HISTOGRAM,
    POLARPLOT
};


/**
 * @brief The Scatterplot2dType enum Defines the different type of plots.
 */
enum Scatterplot2dType {
    SERIES2D,
    TIMESERIES2D
};


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

    virtual bool append(UAVObject* obj) = 0;
    virtual Plot2dType plotType() = 0;
    virtual void removeStaleData() = 0;
    virtual void setUpdatedFlagToTrue(){dataUpdated = true;}
    virtual bool readAndResetUpdatedFlag(){bool tmp = dataUpdated; dataUpdated = false; return tmp;}

private:
    bool dataUpdated;

signals:
};

#endif // PLOTDATA2D_H
