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
        scopeGadgetWidget(widget),
        configLoaded(false)
{

}

int sally=0;

void ScopeGadget::loadConfiguration(IUAVGadgetConfiguration* config)
{
    ScopeGadgetConfiguration *sgConfig = qobject_cast<ScopeGadgetConfiguration*>(config);

    scopeGadgetWidget->setRefreshInterval(sgConfig->refreshInterval());

    if (sgConfig->getPlotDimensions() == Plot2d)
    {
        scopeGadgetWidget->setXWindowSize(sgConfig->dataSize());

        switch (sgConfig->getPlot2dType()){
        case Histogram:
        {
            scopeGadgetWidget->setupHistogramPlot();

            // Configured each data source
            foreach (Plot2dCurveConfiguration* plotCurveConfig,  sgConfig->plot2dCurveConfigs())
            {
                QString uavObjectName = plotCurveConfig->uavObjectName;
                QString uavFieldName = plotCurveConfig->uavFieldName;
                int scale = plotCurveConfig->yScalePower;
                int mean = plotCurveConfig->yMeanSamples;
                QString mathFunction = plotCurveConfig->mathFunction;
                QRgb color = plotCurveConfig->color;

                // TODO: It bothers me that I have to have the ScopeGadgetWidget scopeGadgetWidget in order to call getUavObjectFieldUnits()
                // Get and store the units
                QString units = scopeGadgetWidget->getUavObjectFieldUnits(uavObjectName, uavFieldName);
                sgConfig->setHistogramUnits(units);

                double binWidth = sgConfig->getHistogramConfiguration()->binWidth;
                if (binWidth < 1e-3)
                    binWidth = 1e-3;

                uint numberOfBins = sgConfig->getHistogramConfiguration()->windowWidth;

                // Create the Qwt histogram plot
                scopeGadgetWidget->addHistogram(
                        uavObjectName,
                            uavFieldName,
                            binWidth,
                            numberOfBins,
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
                scopeGadgetWidget->setupSeriesPlot();
                break;
            case TimeSeries2d:
                scopeGadgetWidget->setupTimeSeriesPlot();
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
                scopeGadgetWidget->add2dCurvePlot(
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
        scopeGadgetWidget->setTimeHorizon(sgConfig->getTimeHorizon());

        if(sgConfig->getPlot3dType() == Spectrogram)
            scopeGadgetWidget->setupSpectrogramPlot();

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

            // TODO: It bothers me that I have to have the ScopeGadgetWidget scopeGadgetWidget in order to call getUavObjectFieldUnits()
            // Get and store the units
            QString units = scopeGadgetWidget->getUavObjectFieldUnits(uavObjectName, uavFieldName);
            sgConfig->setSpectrogramUnits(units);

            double timeHorizon       = sgConfig->getTimeHorizon();

            // Create the Qwt waterfall plot
            scopeGadgetWidget->addWaterfallPlot(
                        uavObjectName,
                        uavFieldName,
                        scale,
                        mean,
                        mathFunction,
                        timeHorizon,
                        sgConfig->getSpectrogramConfiguration()
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
 * @brief ScopeGadget::~ScopeGadget   Scope gadget destructor: deletes the
 * associated scope gadget widget too.
 */
ScopeGadget::~ScopeGadget()
{
   delete scopeGadgetWidget;
}
