/**
 ******************************************************************************
 *
 * @file       spectrogramdata.h
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

#ifndef SPECTROGRAMDATA_H
#define SPECTROGRAMDATA_H

#include "plotdata3d.h"
#include "uavobject.h"

#include "qwt/src/qwt.h"
#include "qwt/src/qwt_color_map.h"
#include "qwt/src/qwt_matrix_raster_data.h"
#include "qwt/src/qwt_plot.h"
#include "qwt/src/qwt_plot_spectrogram.h"
#include "qwt/src/qwt_scale_draw.h"
#include "qwt/src/qwt_scale_widget.h"

#include <QTimer>
#include <QTime>
#include <QVector>


/**
 * @brief The SpectrogramData class The histogram plot has a variable sized buffer of data,
 *  where the data is for a specified histogram data set.
 */
/**
 * @brief The SpectrogramData class The spectrogram plot has a fixed size
 * data buffer. All the curves in one plot have the same size buffer.
 */
class SpectrogramData : public Plot3dData
{
    Q_OBJECT
public:
    SpectrogramData(QString uavObject, QString uavField, double samplingFrequency, unsigned int windowWidth, double timeHorizon)
            : Plot3dData(uavObject, uavField),
              spectrogram(0),
              rasterData(0)
    {
        this->samplingFrequency = samplingFrequency;
        this->timeHorizon = timeHorizon;
        this->windowWidth = windowWidth;
        autoscaleValueUpdated = 0;
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
        return SPECTROGRAM;
    }

    /*!
      \brief Removes the old data from the buffer
      */
    virtual void removeStaleData(){}

    /*!
     * \brief readAndResetAutoscaleFlag reads the flag value and resets it
     * \return
     */
    double readAndResetAutoscaleValue(){double tmpVal = autoscaleValueUpdated; autoscaleValueUpdated = 0; return tmpVal;}

    double samplingFrequency;
    double timeHorizon;
    unsigned int windowWidth;

    double autoscaleValueUpdated;

    QwtPlotSpectrogram *spectrogram;
    SpectrogramType spectrogramType;

    QwtMatrixRasterData *rasterData;

};

#endif // SPECTROGRAMDATA_H
