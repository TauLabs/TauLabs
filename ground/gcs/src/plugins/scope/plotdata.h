/**
 ******************************************************************************
 *
 * @file       plotdata.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
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

#ifndef PLOTDATA_H
#define PLOTDATA_H

#include "uavobject.h"

#include "qwt/src/qwt.h"
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
 * @brief The Plot3dType enum Defines the different type of plots.
 */
enum PlotDimensions {
    Plot2d,
    Plot3d
};


/**
 * @brief The Plot2dType enum Defines the different type of plots.
 */
enum Plot2dType {
    No2dPlot, //Signifies that there is no 2D plot configured
    Scatterplot2d,
    Histogram,
    PolarPlot
};


/**
 * @brief The Plot3dType enum Defines the different type of plots.
 */
enum Plot3dType {
    No3dPlot, //Signifies that there is no 3D plot configured
    Scatterplot3d,
    Spectrogram
};


/**
 * @brief The Scatterplot2dType enum Defines the different type of plots.
 */
enum Scatterplot2dType {
    Series2d,
    TimeSeries2d
};


/**
 * @brief The Scatterplot3dType enum Defines the different type of plots.
 */
enum Scatterplot3dType {
    TimeSeries3d
};


/**
 * @brief The SpectrogramType enum Defines the different type of spectrogram plots.
 */
enum SpectrogramType {
    VibrationTest,
    Custom
};


/**
 * @brief The Plot2dData class Base class that keeps the data for each curve in the plot.
 */
class Plot2dData : public QObject
{
    Q_OBJECT

public:
    Plot2dData(QString uavObject, QString uavField);
    ~Plot2dData();

    QString uavObjectName;
    QString uavFieldName;
    QString uavSubFieldName;
    bool haveSubField;
    int scalePower; //This is the power to which each value must be raised
    int meanSamples;
    double meanSum;
    QString mathFunction;
    double correctionSum;
    int correctionCount;
    QwtPlotCurve* curve;
    QwtPlotHistogram* histogram;
    QVector<double>* xData;        //Used for scatterplots
    QVector<double>* yData;        //Used for scatterplots
    QVector<double>* yDataHistory; //Used for scatterplots
    QVector<QwtIntervalSample> *histogramBins; //Used for histograms
    QVector<QwtInterval> *histogramInterval;
    QwtIntervalSeriesData *intervalSeriesData;

    void setYMinimum(double val){yMinimum=val;}
    void setYMaximum(double val){yMaximum=val;}
    void setXWindowSize(double val){m_xWindowSize=val;}

    double getYMinimum(){return yMinimum;}
    double getYMaximum(){return yMaximum;}
    double getXWindowSize(){return m_xWindowSize;}

    virtual bool append(UAVObject* obj) = 0;
    virtual Plot2dType plotType() = 0;
    virtual void removeStaleData() = 0;


    void updatePlotCurveData();

protected:
//    double valueAsDouble(UAVObject* obj, UAVObjectField* field);

private:
    double yMinimum;
    double yMaximum;
    double m_xWindowSize;


signals:
    void dataChanged();
};


/**
 * @brief The Plot3dData class Base class that keeps the data for each curve in the plot.
 */
class Plot3dData : public QObject
{
    Q_OBJECT

public:
    Plot3dData(QString uavObject, QString uavField);
    ~Plot3dData();

    QString uavObjectName;
    QString uavFieldName;
    QString uavSubFieldName;
    bool haveSubField;
    int scalePower; //This is the power to which each value must be raised
    int meanSamples;
    double meanSum;
    QString mathFunction;
    double correctionSum;
    int correctionCount;
    QwtPlotCurve* curve;
    QwtPlotSpectrogram *spectrogram;
    QVector<double>* xData;
    QVector<double>* yData;
    QVector<double>* zData;
    QVector<double>* zDataHistory;
    QVector<double>* timeDataHistory;

    SpectrogramType spectrogramType;

    QwtMatrixRasterData *rasterData;

    void setXMinimum(double val){xMinimum=val;}
    void setXMaximum(double val){xMaximum=val;}
    void setYMinimum(double val){yMinimum=val;}
    void setYMaximum(double val){yMaximum=val;}
    void setZMinimum(double val){zMinimum=val;}
    void setZMaximum(double val){zMaximum=val;}

    double getXMinimum(){return xMinimum;}
    double getXMaximum(){return xMaximum;}
    double getYMinimum(){return yMinimum;}
    double getYMaximum(){return yMaximum;}
    double getZMinimum(){return zMinimum;}
    double getZMaximum(){return zMaximum;}

    virtual bool append(UAVObject* obj) = 0;
    virtual Plot3dType plotType() = 0;
    virtual void removeStaleData() = 0;

    void updatePlotCurveData();

protected:
//    double valueAsDouble(UAVObject* obj, UAVObjectField* field);

private:
    double xMinimum;
    double xMaximum;
    double yMinimum;
    double yMaximum;
    double zMinimum;
    double zMaximum;


signals:
    void dataChanged();
};


/**
 * @brief The Scatterplot2dData class Base class that keeps the data for each curve in the plot.
 */
class ScatterplotData : public Plot2dData
{
    Q_OBJECT
public:
    ScatterplotData(QString uavObject, QString uavField):
        Plot2dData(uavObject, uavField){}
    ~ScatterplotData(){}
};



/**
 * @brief The HistogramData class The histogram plot has a variable sized buffer of data,
 *  where the data is for a specified histogram data set.
 */
class HistogramData : public Plot2dData
{
    Q_OBJECT
public:
    HistogramData(QString uavObject, QString uavField) :
        Plot2dData(uavObject, uavField)
    {
        scalePower = 1;
    }

    ~HistogramData() {
    }

    bool append(UAVObject* obj);

    /*!
      \brief The type of plot
      */
    virtual Plot2dType plotType() {
        return Histogram;
    }

    virtual void removeStaleData(){}

private:

private slots:

};


/**
 * @brief The SpectrogramData class The spectrogram plot has a fixed size
 * data buffer. All the curves in one plot have the same size buffer.
 */
class SpectrogramData : public Plot3dData
{
    Q_OBJECT
public:
    SpectrogramData(QString uavObject, QString uavField, double samplingFrequency, unsigned int windowWidth, double timeHorizon)
            : Plot3dData(uavObject, uavField)
    {
        this->samplingFrequency = samplingFrequency;
        this->timeHorizon = timeHorizon;
        this->windowWidth = windowWidth;
    }
    ~SpectrogramData() {}

    /*!
      \brief Append new data to the plot
      */
    bool append(UAVObject* obj);

    /*!
      \brief The type of plot
      */
    virtual Plot3dType plotType() {
        return Spectrogram;
    }

    /*!
      \brief Removes the old data from the buffer
      */
    virtual void removeStaleData(){}

    double samplingFrequency;
    double timeHorizon;
    unsigned int windowWidth;
};


//====================================================
// BELOW THIS LINE ARE SPECIFIC SCATTER PLOTS. THEY
// SHOULD BE FOLDED INTO THE Scatterplot CLASS
//====================================================

/**
 * @brief The SeriesPlotData class The sequential plot have a fixed size
 * buffer of data. All the curves in one plot have the same size buffer.
 */
class SeriesPlotData : public Plot2dData
{
    Q_OBJECT
public:
    SeriesPlotData(QString uavObject, QString uavField)
            : Plot2dData(uavObject, uavField) {}
    ~SeriesPlotData() {}

    /*!
      \brief Append new data to the plot
      */
    bool append(UAVObject* obj);

    /*!
      \brief The type of plot
      */
    virtual Plot2dType plotType() {
        return Scatterplot2d;
    }

    /*!
      \brief The type of scatterplot
      */
    virtual Scatterplot2dType scatterplotType() {
        return Series2d;
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
class TimeSeriesPlotData : public Plot2dData
{
    Q_OBJECT
public:
    TimeSeriesPlotData(QString uavObject, QString uavField)
            : Plot2dData(uavObject, uavField) {
        scalePower = 1;
    }
    ~TimeSeriesPlotData() {
    }

    bool append(UAVObject* obj);

    /*!
      \brief The type of plot
      */
    virtual Plot2dType plotType() {
        return Scatterplot2d;
    }

    /*!
      \brief The type of scatterplot
      */
    virtual Scatterplot2dType scatterplotType() {
        return TimeSeries2d;
    }

    virtual void removeStaleData();

private:

private slots:
    void removeStaleDataTimeout();
};

#endif // PLOTDATA_H
