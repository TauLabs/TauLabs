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
enum PlotDimensions {
    PLOT2D,
    PLOT3D
};


class PlotData : public QObject
{
    Q_OBJECT
public:
    void setXMinimum(double val){xMinimum=val;}
    void setXMaximum(double val){xMaximum=val;}
    void setYMinimum(double val){yMinimum = val;}
    void setYMaximum(double val){yMaximum = val;}
    void setXWindowSize(double val){m_xWindowSize=val;}

    double getXMinimum(){return xMinimum;}
    double getXMaximum(){return xMaximum;}
    double getYMinimum(){return yMinimum;}
    double getYMaximum(){return yMaximum;}
    double getXWindowSize(){return m_xWindowSize;}


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

    QVector<double>* xData;        //Used for scatterplots
    QVector<double>* yData;        //Used for scatterplots

    QwtScaleWidget *rightAxis;

protected:
    double xMinimum;
    double xMaximum;
    double yMinimum;
    double yMaximum;

    double m_xWindowSize;
private:

};

/**
 * @brief The ColorMap class Defines a program-wide colormap
 */
class ColorMap: public QwtLinearColorMap
{
public:
    ColorMap():
        QwtLinearColorMap( Qt::darkCyan, Qt::red )
    {
        // Values given normalized to 1.
        addColorStop( 0.1, Qt::cyan );
        addColorStop( 0.6, Qt::green );
        addColorStop( 0.95, Qt::yellow );
    }
};


#endif // PLOTDATA_H
