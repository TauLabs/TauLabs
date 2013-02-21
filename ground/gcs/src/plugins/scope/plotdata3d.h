/**
 ******************************************************************************
 *
 * @file       plotdata3d.h
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

#ifndef PLOTDATA3D_H
#define PLOTDATA3D_H

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
 * @brief The Plot3dType enum Defines the different type of plots.
 */
enum Plot3dType {
    NO3DPLOT,
    SCATTERPLOT3D,
    SPECTROGRAM
};

///**
// * @brief The Scatterplot3dType enum Defines the different type of plots.
// */
//enum Scatterplot3dType {
//    TIMESERIES3D
//};


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

    virtual bool append(UAVObject* obj) = 0;
    virtual Plot3dType plotType() = 0;
    virtual void removeStaleData() = 0;
    virtual void setUpdatedFlagToTrue(){dataUpdated = true;}
    virtual bool readAndResetUpdatedFlag(){bool tmp = dataUpdated; dataUpdated = false; return tmp;}

    QwtPlotCurve* curve; //TODO: This shouldn't be here, it should be in a ScatterplotData class

protected:
    double zMinimum;
    double zMaximum;

private:
    bool dataUpdated;

signals:
//    void dataChanged();
};



#endif // PLOTDATA3D_H
