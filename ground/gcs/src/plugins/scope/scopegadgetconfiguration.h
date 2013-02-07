/**
 ******************************************************************************
 *
 * @file       scopegadgetconfiguration.h
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

#ifndef SCOPEGADGETCONFIGURATION_H
#define SCOPEGADGETCONFIGURATION_H

#include "plotdata.h"
#include <coreplugin/iuavgadgetconfiguration.h>

#include <QVector>

using namespace Core;


// This struct holds the configuration for individual 2D data sources
struct Plot2dCurveConfiguration
{
    QString uavObjectName;
    QString uavFieldName;
    int yScalePower; //This is the power to which each value must be raised
    QRgb color;
    int yMeanSamples;
    QString mathFunction;
    double yMinimum;
    double yMaximum;
};

// This struct holds the configuration for individual 3D data sources
struct Plot3dCurveConfiguration
{
    QString uavObjectName;
    QString uavFieldName;
    int yScalePower; //This is the power to which each value must be raised
    QRgb color;
    int yMeanSamples;
    QString mathFunction;
    double yMinimum;
    double yMaximum;
};


// This struct holds the configuration for individual spectrogram data source
struct SpectrogramDataConfiguration
{
    double samplingFrequency;
    unsigned int windowWidth;
    QString yAxisUnits;
};


// This struct holds the configuration for individual histogram data source
struct HistogramDataConfiguration
{
    double binWidth;
    unsigned int windowWidth;
    QString xAxisUnits;
};


class ScopeGadgetConfiguration : public IUAVGadgetConfiguration
{
    Q_OBJECT
public:
    explicit ScopeGadgetConfiguration(QString classId, QSettings* qSettings = 0, QObject *parent = 0);

    ~ScopeGadgetConfiguration();

    //configuration setter functions
    void setScatterplot2dType(Scatterplot2dType value){m_scatterplot2dType = value;}
    void setPlot2dType(Plot2dType value){m_plot2dType = value;}
    void setPlot3dType(Plot3dType value){m_plot3dType = value;}
    void setPlotDimensions(PlotDimensions value){m_plotDimensions = value;}
//    void setMathFunctionType(int value){m_mathFunctionType = value;}
    void setDataSize(int value){m_dataSize = value;}
    void setTimeHorizon(double value){m_timeHorizon = value;}
    void setRefreshInterval(int value){m_refreshInterval = value;}
    void addPlot2dCurveConfig(Plot2dCurveConfiguration* value){m_Plot2dCurveConfigs.append(value);}
    void replacePlot2dCurveConfig(QList<Plot2dCurveConfiguration*> m_Plot2dCurveConfigs);
    void replacePlot3dCurveConfig(QList<Plot3dCurveConfiguration*> m_Plot3dCurveConfigs);
    void replaceSpectrogramConfig(SpectrogramDataConfiguration *val){if(m_SpectrogramConfig !=NULL) delete m_SpectrogramConfig; m_SpectrogramConfig = val;}
    void replaceHistogramConfig(HistogramDataConfiguration *val){if(m_HistogramConfig !=NULL) delete m_HistogramConfig; m_HistogramConfig = val;}
    void setSpectrogramUnits(QString units){m_SpectrogramConfig->yAxisUnits = units;}
    void setHistogramUnits(QString units){m_HistogramConfig->xAxisUnits = units;}

    //configurations getter functions
    Scatterplot2dType getScatterplot2dType(){return m_scatterplot2dType;}
    Plot2dType getPlot2dType(){return m_plot2dType;}
    Plot3dType getPlot3dType(){return m_plot3dType;}
    PlotDimensions getPlotDimensions(){return m_plotDimensions;}
//    int mathFunctionType(){return m_mathFunctionType;}
    int dataSize(){return m_dataSize;}
    double getTimeHorizon(){return m_timeHorizon;}
    int refreshInterval(){return m_refreshInterval;}
    QList<Plot2dCurveConfiguration*> plot2dCurveConfigs(){return m_Plot2dCurveConfigs;}
    QList<Plot3dCurveConfiguration*> plot3dCurveConfigs(){return m_Plot3dCurveConfigs;}
    SpectrogramDataConfiguration *getSpectrogramConfiguration() {return m_SpectrogramConfig;}
    HistogramDataConfiguration *getHistogramConfiguration() {return m_HistogramConfig;}

    void saveConfig(QSettings* settings) const; //THIS SEEMS TO BE UNUSED
    IUAVGadgetConfiguration *clone();

private:
    PlotDimensions m_plotDimensions;
    Plot2dType m_plot2dType; //The type of 2d plot
    Plot3dType m_plot3dType; //The type of 3d plot
    Scatterplot2dType m_scatterplot2dType;
    int m_dataSize; //The size of the data buffer to render in the curve plot
    double m_timeHorizon; //The time window
    int m_refreshInterval; //The interval to replot the curve widget. The data buffer is refresh as the data comes in.
    QList<Plot2dCurveConfiguration*> m_Plot2dCurveConfigs;
    QList<Plot3dCurveConfiguration*> m_Plot3dCurveConfigs;
    SpectrogramDataConfiguration *m_SpectrogramConfig;
    HistogramDataConfiguration *m_HistogramConfig;

    void clearPlot2dData();
    void clearPlot3dData();

};

#endif // SCOPEGADGETCONFIGURATION_H
