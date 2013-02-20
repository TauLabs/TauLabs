/**
 ******************************************************************************
 *
 * @file       spectrogramscope.cpp
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

#include "scopes3d/spectrogramscopeconfig.h"


SpectrogramScope::SpectrogramScope()
{
    m_refreshInterval = 50; //TODO: This should not be set here. Probably should come from a define somewhere.
    yAxisUnits = "";
    timeHorizon = 60;
    samplingFrequency = 100;
    windowWidth = 64;
    zMaximum = 120;
}


SpectrogramScope::SpectrogramScope(QSettings *qSettings) //TODO: Understand where to put m_refreshInterval default values
{
    timeHorizon = qSettings->value("timeHorizon").toDouble();
    samplingFrequency = qSettings->value("samplingFrequency").toDouble();
    windowWidth       = qSettings->value("windowWidth").toInt();
    zMaximum = qSettings->value("zMaximum").toDouble();

    int plot3dCurveCount = qSettings->value("dataSourceCount").toInt();

    for(int i = 0; i < plot3dCurveCount; i++){
        Plot3dCurveConfiguration *plotCurveConf = new Plot3dCurveConfiguration();

        qSettings->beginGroup(QString("spectrogramDataSource") + QString().number(i));

        plotCurveConf->uavObjectName = qSettings->value("uavObject").toString();
        plotCurveConf->uavFieldName  = qSettings->value("uavField").toString();
        plotCurveConf->color         = qSettings->value("color").value<QRgb>();
        plotCurveConf->yScalePower   = qSettings->value("yScalePower").toInt();
        plotCurveConf->mathFunction  = qSettings->value("mathFunction").toString();
        plotCurveConf->yMeanSamples  = qSettings->value("yMeanSamples").toInt();

        plotCurveConf->yMinimum = qSettings->value("yMinimum").toDouble();
        plotCurveConf->yMaximum = qSettings->value("yMaximum").toDouble();

        //Stop reading XML block
        qSettings->endGroup();

        m_spectrogramSourceConfigs.append(plotCurveConf);
    }
}


SpectrogramScope::~SpectrogramScope()
{

}

//TODO: This should probably be a constructor, too.
void SpectrogramScope::clone(ScopesGeneric *originalScope)
{
//    this->parent() = new SpectrogramScope();

    SpectrogramScope *originalHistogramScope = (SpectrogramScope*) originalScope;

    timeHorizon = originalHistogramScope->timeHorizon;

    int plotCurveCount = originalHistogramScope->m_spectrogramSourceConfigs.size();

    for (int i = 0; i < plotCurveCount; i++){
        Plot3dCurveConfiguration *currentPlotCurveConf = originalHistogramScope->m_spectrogramSourceConfigs.at(i);
        Plot3dCurveConfiguration *newSpectrogramConf     = new Plot3dCurveConfiguration();

        newSpectrogramConf->uavObjectName = currentPlotCurveConf->uavObjectName;
        newSpectrogramConf->uavFieldName  = currentPlotCurveConf->uavFieldName;
        newSpectrogramConf->color         = currentPlotCurveConf->color;
        newSpectrogramConf->yScalePower   = currentPlotCurveConf->yScalePower;
        newSpectrogramConf->yMeanSamples  = currentPlotCurveConf->yMeanSamples;
        newSpectrogramConf->mathFunction  = currentPlotCurveConf->mathFunction;

        newSpectrogramConf->yMinimum = currentPlotCurveConf->yMinimum;
        newSpectrogramConf->yMaximum = currentPlotCurveConf->yMaximum;

        m_spectrogramSourceConfigs.append(newSpectrogramConf);
    }
}


void SpectrogramScope::saveConfiguration(QSettings* qSettings)
{
    //Start writing new XML block
    qSettings->beginGroup(QString("plot3d"));

    int plot3dCurveCount = m_spectrogramSourceConfigs.size();

    qSettings->setValue("plot3dType", m_plot3dType);
    qSettings->setValue("dataSourceCount", plot3dCurveCount);

    qSettings->setValue("samplingFrequency", samplingFrequency);
    qSettings->setValue("timeHorizon", timeHorizon);
    qSettings->setValue("windowWidth", windowWidth);
    qSettings->setValue("zMaximum",  zMaximum);

    for(int i = 0; i < plot3dCurveCount; i++){
        Plot3dCurveConfiguration *plotCurveConf = m_spectrogramSourceConfigs.at(i);

        //Start new XML block
        qSettings->beginGroup(QString("spectrogramDataSource") + QString().number(i));

        qSettings->setValue("uavObject",  plotCurveConf->uavObjectName);
        qSettings->setValue("uavField",  plotCurveConf->uavFieldName);
        qSettings->setValue("colormap",  plotCurveConf->color);

        //Stop writing XML block
        qSettings->endGroup();
    }

    //Stop writing XML block
    qSettings->endGroup();
}


/**
 * @brief SpectrogramScope::replaceHistogramSource Replaces the list of histogram data sources
 * @param histogramSourceConfigs
 */
void SpectrogramScope::replaceSpectrogramDataSource(QList<Plot3dCurveConfiguration*> spectrogramSourceConfigs)
{
    m_spectrogramSourceConfigs.clear();
    m_spectrogramSourceConfigs.append(spectrogramSourceConfigs);
}


/**
 * @brief SpectrogramScope::loadConfiguration loads the plot configuration into the scope gadget widget
 * @param scopeGadgetWidget
 */
void SpectrogramScope::loadConfiguration(ScopeGadgetWidget **scopeGadgetWidget)
{
    (*scopeGadgetWidget)->setupSpectrogramPlot();
    (*scopeGadgetWidget)->setRefreshInterval(m_refreshInterval);

    //There should be only one spectrogram per plot //TODO: Change this to handle multiple spectrograms
    if ( m_spectrogramSourceConfigs.length() != 1)
        return;

    Plot3dCurveConfiguration* spectrogramSourceConfigs = m_spectrogramSourceConfigs.front();
    QString uavObjectName = spectrogramSourceConfigs->uavObjectName;
    QString uavFieldName = spectrogramSourceConfigs->uavFieldName;
    int scale = spectrogramSourceConfigs->yScalePower;
    int mean = spectrogramSourceConfigs->yMeanSamples;
    QString mathFunction = spectrogramSourceConfigs->mathFunction;

    // TODO: It bothers me that I have to have the ScopeGadgetWidget scopeGadgetWidget in order to call getUavObjectFieldUnits()
    // Get and store the units
    units = (*scopeGadgetWidget)->getUavObjectFieldUnits(uavObjectName, uavFieldName);

            // Create the Qwt waterfall plot
            (*scopeGadgetWidget)->addWaterfallPlot(
                        uavObjectName,
                        uavFieldName,
                        scale,
                        mean,
                        mathFunction,
                        timeHorizon,
                        samplingFrequency,
                        windowWidth,
                        zMaximum
                        );
}
