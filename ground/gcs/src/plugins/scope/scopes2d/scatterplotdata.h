/**
 ******************************************************************************
 *
 * @file       scatterplotdata.h
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

#ifndef SCATTERPLOTDATA_H
#define SCATTERPLOTDATA_H

#include "scopes2d/plotdata2d.h"
#include "uavobject.h"
#include "qwt/src/qwt_plot_curve.h"

#include <QTimer>
#include <QTime>
#include <QVector>


/**
 * @brief The Scatterplot2dData class Base class that keeps the data for each curve in the plot.
 */
class ScatterplotData : public Plot2dData
{
    Q_OBJECT
public:
    ScatterplotData(QString uavObject, QString uavField):
        Plot2dData(uavObject, uavField){curve = 0;}
    ~ScatterplotData(){}

    virtual void clearPlots(PlotData *);

    void setCurve(QwtPlotCurve *val){curve = val;}

protected:
    QwtPlotCurve* curve;
};



/**
 * @brief The SeriesPlotData class The sequential plot have a fixed size
 * buffer of data. All the curves in one plot have the same size buffer.
 */
class SeriesPlotData : public ScatterplotData
{
    Q_OBJECT
public:
    SeriesPlotData(QString uavObject, QString uavField)
            : ScatterplotData(uavObject, uavField) {}
    ~SeriesPlotData() {}

    /*!
      \brief Append new data to the plot
      */
    bool append(UAVObject* obj);


    /*!
      \brief Removes the old data from the buffer
      */
    virtual void removeStaleData(){}
    virtual void plotNewData(PlotData *, ScopeConfig *, ScopeGadgetWidget *);
};


/**
 * @brief The TimeSeriesPlotData class The chrono plot has a variable sized buffer of data,
 * where the data is for a specified time period.
 */
class TimeSeriesPlotData : public ScatterplotData
{
    Q_OBJECT
public:
    TimeSeriesPlotData(QString uavObject, QString uavField)
            : ScatterplotData(uavObject, uavField) {
        scalePower = 1;
    }
    ~TimeSeriesPlotData() {
    }

    bool append(UAVObject* obj);

    virtual void removeStaleData();
    virtual void plotNewData(PlotData *, ScopeConfig *, ScopeGadgetWidget *);

private slots:
    void removeStaleDataTimeout();
};

#endif // SCATTERPLOTDATA_H
