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
#include "qwt/src/qwt_matrix_raster_data.h"
#include "qwt/src/qwt_plot.h"
#include "qwt/src/qwt_plot_curve.h"
#include "qwt/src/qwt_plot_histogram.h"
#include "qwt/src/qwt_plot_spectrogram.h"
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
//    void dataChanged();
};


///**
// * @brief The HistogramData class The histogram plot has a variable sized buffer of data,
// *  where the data is for a specified histogram data set.
// */
//class HistogramData : public Plot2dData
//{
//    Q_OBJECT
//public:
//    HistogramData(QString uavObject, QString uavField, double binWidth, uint numberOfBins) :
//        Plot2dData(uavObject, uavField),
//        histogram(0),
//        histogramBins(0),
//        histogramInterval(0),
//        intervalSeriesData(0)
//    {
//        this->binWidth = binWidth;
//        this->numberOfBins = numberOfBins;
//        scalePower = 1;
//    }

//    ~HistogramData() {
//    }

//    bool append(UAVObject* obj);

//    /*!
//      \brief The type of plot
//      */
//    virtual Plot2dType plotType() {
//        return HISTOGRAM;
//    }

//    virtual void removeStaleData(){}

//    QwtPlotHistogram* histogram;
//    QVector<QwtIntervalSample> *histogramBins; //Used for histograms
//    QVector<QwtInterval> *histogramInterval;
//    QwtIntervalSeriesData *intervalSeriesData;

//private:
//    double binWidth;
//    uint numberOfBins;

//private slots:

//};

//====================================================
// BELOW THIS LINE ARE SPECIFIC SCATTER PLOTS. THEY
// ARE FOLDED INTO THE Scatterplot CLASS
//====================================================

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
      \brief The type of plot
      */
    virtual Plot2dType plotType() {
        return SCATTERPLOT2D;
    }

    /*!
      \brief The type of scatterplot
      */
    virtual Scatterplot2dType scatterplotType() {
        return SERIES2D;
    }

    /*!
      \brief Removes the old data from the buffer
      */
    virtual void removeStaleData(){}
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

    /*!
      \brief The type of plot
      */
    virtual Plot2dType plotType() {
        return SCATTERPLOT2D;
    }

    /*!
      \brief The type of scatterplot
      */
    virtual Scatterplot2dType scatterplotType() {
        return TIMESERIES2D;
    }

    virtual void removeStaleData();

private:

private slots:
    void removeStaleDataTimeout();
};


#endif // PLOTDATA2D_H
