/**
 ******************************************************************************
 *
 * @file       spectrogramplotdata.h
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

#ifndef SPECTROGRAMDATA_H
#define SPECTROGRAMDATA_H

#include "scopes3d/plotdata3d.h"
#include "uavobject.h"
#include "qwt/src/qwt_plot_spectrogram.h"
#include "qwt/src/qwt_matrix_raster_data.h"

#include <QTimer>
#include <QTime>
#include <QVector>


/**
 * @brief The SpectrogramData class The spectrogram plot has a fixed size
 * data buffer. All the curves in one plot have the same size buffer.
 */
class SpectrogramData : public Plot3dData
{
    Q_OBJECT
public:
    SpectrogramData(QString uavObject, QString uavField, double samplingFrequency, unsigned int windowWidth, double timeHorizon);
    ~SpectrogramData() {}

    /*!
      \brief Append new data to the plot
      */
    bool append(UAVObject* obj);

    /*!
      \brief Removes the old data from the buffer
      */
    virtual void removeStaleData(){}

    /*!
     * \brief readAndResetAutoscaleFlag reads the flag value and resets it
     * \return
     */
    double readAndResetAutoscaleValue(){double tmpVal = autoscaleValueUpdated; autoscaleValueUpdated = 0; return tmpVal;}

    virtual void plotNewData(PlotData *, ScopeConfig *, ScopeGadgetWidget *);
    virtual void clearPlots(PlotData *);
    virtual void setXMaximum(double val);
    virtual void setYMaximum(double val);
    virtual void setZMaximum(double val);

    QwtMatrixRasterData *getRasterData(){return rasterData;}
    void setSpectrogram(QwtPlotSpectrogram *val){spectrogram = val;}

private:
    void resetAxisRanges();

    QwtPlotSpectrogram *spectrogram;
    QwtMatrixRasterData *rasterData;

    double samplingFrequency;
    double timeHorizon;
    unsigned int windowWidth;
    double autoscaleValueUpdated;
};

#endif // SPECTROGRAMDATA_H
