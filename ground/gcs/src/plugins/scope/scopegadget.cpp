/**
 ******************************************************************************
 *
 * @file       scopegadget.cpp
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

#include "scopeplugin.h"
#include "scopegadget.h"
#include "scopegadgetconfiguration.h"
#include "scopegadgetwidget.h"

#include <QtGui/qcolor.h>

ScopeGadget::ScopeGadget(QString classId, ScopeGadgetWidget *widget, QWidget *parent) :
        IUAVGadget(classId, parent),
        m_widget(widget),
        configLoaded(false)
{

}

int sally=0;

void ScopeGadget::loadConfiguration(IUAVGadgetConfiguration* config)
{
    ScopeGadgetConfiguration *sgConfig = qobject_cast<ScopeGadgetConfiguration*>(config);
    ScopeGadgetWidget* widget = qobject_cast<ScopeGadgetWidget*>(m_widget);

    widget->setRefreshInterval(sgConfig->refreshInterval());

    if (sgConfig->getPlotDimensions() == Plot2d)
    {
        widget->setXWindowSize(sgConfig->dataSize());

        switch (sgConfig->getPlot2dType()){
        case Histogram:
        {
            widget->setupHistogramPlot();

            // Configured each data source
            foreach (Plot2dCurveConfiguration* plotCurveConfig,  sgConfig->plot2dCurveConfigs())
            {
                QString uavObjectName = plotCurveConfig->uavObjectName;
                QString uavFieldName = plotCurveConfig->uavFieldName;
                int scale = plotCurveConfig->yScalePower;
                int mean = plotCurveConfig->yMeanSamples;
                QString mathFunction = plotCurveConfig->mathFunction;
                QRgb color = plotCurveConfig->color;

                // TODO: It bothers me that I have to have the ScopeGadgetWidget widget in order to call getUavObjectFieldUnits()
                // Get and store the units
                QString units = widget->getUavObjectFieldUnits(uavObjectName, uavFieldName);
                sgConfig->setHistogramUnits(units);

                double binWidth = sgConfig->getHistogramConfiguration()->binWidth;

                // Create the Qwt histogram plot
                widget->addHistogram(
                        uavObjectName,
                            uavFieldName,
                            binWidth,
                            scale,
                            mean,
                            mathFunction,
                            QBrush(QColor(color))
                            );
            }
            break;
        }
        case Scatterplot2d:
        {
            switch (sgConfig->getScatterplot2dType())
            {
            case Series2d:
                widget->setupSeriesPlot();
                break;
            case TimeSeries2d:
                widget->setupTimeSeriesPlot();
                break;
            default:
                //We shouldn't be able to get here.
                Q_ASSERT(0);
            }


            // Configured each data source
            foreach (Plot2dCurveConfiguration* plotCurveConfig,  sgConfig->plot2dCurveConfigs())
            {
                QString uavObjectName = plotCurveConfig->uavObjectName;
                QString uavFieldName = plotCurveConfig->uavFieldName;
                int scale = plotCurveConfig->yScalePower;
                int mean = plotCurveConfig->yMeanSamples;
                QString mathFunction = plotCurveConfig->mathFunction;
                QRgb color = plotCurveConfig->color;

                // Create the Qwt curve plot
                widget->add2dCurvePlot(
                        uavObjectName,
                            uavFieldName,
                            scale,
                            mean,
                            mathFunction,
                            QPen(  QBrush(QColor(color),Qt::SolidPattern),
                               (qreal)1,
                               Qt::SolidLine,
                               Qt::SquareCap,
                               Qt::BevelJoin)
                            );
            }
            break;
        }
        default:
            //We should never get this far. Do something of interest
            Q_ASSERT(0);
            break;
        }
    }
    else if (sgConfig->getPlotDimensions() == Plot3d)
    {
        widget->setTimeHorizon(sgConfig->getTimeHorizon());

        if(sgConfig->getPlot3dType() == Spectrogram)
            widget->setupSpectrogramPlot();

        switch(sgConfig->getPlot3dType())
        {
        case Spectrogram:
        {
            QList<Plot3dCurveConfiguration*> plotSpectrogramConfigs = sgConfig->plot3dCurveConfigs();

            //There should be only one spectrogram per plot //TODO: Change this to handle multiple spectrograms
            if ( plotSpectrogramConfigs.length() != 1)
                break;
            Plot3dCurveConfiguration* plotSpectrogramConfig = plotSpectrogramConfigs.front();
            QString uavObjectName = plotSpectrogramConfig->uavObjectName;
            QString uavFieldName = plotSpectrogramConfig->uavFieldName;
            int scale = plotSpectrogramConfig->yScalePower;
            int mean = plotSpectrogramConfig->yMeanSamples;
            QString mathFunction = plotSpectrogramConfig->mathFunction;
            qDebug() << "Generating waterfall plot for UAVO " << uavObjectName;

            // TODO: It bothers me that I have to have the ScopeGadgetWidget widget in order to call getUavObjectFieldUnits()
            // Get and store the units
            QString units = widget->getUavObjectFieldUnits(uavObjectName, uavFieldName);
            sgConfig->setSpectrogramUnits(units);

            double timeHorizon       = sgConfig->getTimeHorizon();
            double samplingFrequency = sgConfig->getSpectrogramConfiguration()->samplingFrequency;
            int windowWidth          = sgConfig->getSpectrogramConfiguration()->windowWidth;

            // Create the Qwt waterfall plot
            widget->addWaterfallPlot(
                        uavObjectName,
                        uavFieldName,
                        scale,
                        mean,
                        mathFunction,
                        samplingFrequency,
                        windowWidth,
                        timeHorizon
                        );
            break;
        }
        case Scatterplot3d:
            foreach (Plot3dCurveConfiguration* plotSpectrogramConfig,  sgConfig->plot3dCurveConfigs()) {
                QString uavObjectName = plotSpectrogramConfig->uavObjectName;
                QString uavFieldName = plotSpectrogramConfig->uavFieldName;
                int scale = plotSpectrogramConfig->yScalePower;
                int mean = plotSpectrogramConfig->yMeanSamples;
                QString mathFunction = plotSpectrogramConfig->mathFunction;

                qDebug() << "Generating 3D time series plot for UAVO " << uavObjectName;

                // Do something here...

            }
            break;
        default:
            //We should never get this far. Do something of interest
            Q_ASSERT(0);
            break;
        }
    }
}

/**
  */
/**
 * @brief ScopeGadget::~ScopeGadget   Scope gadget destructor: should delete the
 * associated scope gadget widget too! <-- (Does it?[KDS])
 */
ScopeGadget::~ScopeGadget()
{
   delete m_widget;
}
