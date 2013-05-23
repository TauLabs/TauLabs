/**
 ******************************************************************************
 *
 * @file       histogramplotdata.h
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

#ifndef HISTOGRAMDATA_H
#define HISTOGRAMDATA_H

#include "scopes2d/plotdata2d.h"
#include "uavobject.h"

#include "qwt/src/qwt_plot_histogram.h"


#include <QTimer>
#include <QTime>
#include <QVector>


/**
 * @brief The HistogramData class The histogram plot has a variable sized buffer of data,
 *  where the data is for a specified histogram data set.
 */
class HistogramData : public Plot2dData
{
    Q_OBJECT
public:
    HistogramData(QString uavObject, QString uavField, double binWidth, uint numberOfBins);
    ~HistogramData() {}

    bool append(UAVObject* obj);

    virtual void removeStaleData(){}
    virtual void plotNewData(PlotData *, ScopeConfig *, ScopeGadgetWidget *);
    virtual void clearPlots(PlotData *);

    QwtIntervalSeriesData *getIntervalSeriesData(){return intervalSeriesData;}
    void setHistogram(QwtPlotHistogram *val){histogram = val;}

private:
    QwtPlotHistogram *histogram;
    QVector<QwtIntervalSample> *histogramBins; //Used for histograms
    QVector<QwtInterval> *histogramInterval;
    QwtIntervalSeriesData *intervalSeriesData;

    double binWidth;
    uint numberOfBins;

private slots:

};

#endif // HISTOGRAMDATA_H
