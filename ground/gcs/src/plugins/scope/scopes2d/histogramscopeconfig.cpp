/**
 ******************************************************************************
 *
 * @file       histogramscope.cpp
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

#include "scopes2d/histogramscopeconfig.h"


HistogramScope::HistogramScope()
{
    binWidth = 1;
    maxNumberOfBins = 1000;
    m_refreshInterval = 50; //TODO: This should not be set here. Probably should come from a define somewhere.
}


HistogramScope::HistogramScope(QSettings *qSettings) //TODO: Understand where to put m_refreshInterval default values
{
    binWidth    = qSettings->value("binWidth").toDouble();
    //Ensure binWidth is not too small
    if (binWidth < 1e-3)
        binWidth = 1e-3;

    maxNumberOfBins = qSettings->value("maxNumberOfBins").toInt();
    this->m_refreshInterval = m_refreshInterval;
    this->m_plotDimensions = m_plotDimensions;


    int dataSourceCount = qSettings->value("dataSourceCount").toInt();
    for(int i = 0; i < dataSourceCount; i++)
    {
        qSettings->beginGroup(QString("histogramDataSource") + QString().number(i));

        Plot2dCurveConfiguration *plotCurveConf = new Plot2dCurveConfiguration();

        plotCurveConf->uavObjectName = qSettings->value("uavObject").toString();
        plotCurveConf->uavFieldName  = qSettings->value("uavField").toString();
        plotCurveConf->color         = qSettings->value("color").value<QRgb>();
        plotCurveConf->yScalePower   = qSettings->value("yScalePower").toInt();
        plotCurveConf->mathFunction  = qSettings->value("mathFunction").toString();
        plotCurveConf->yMeanSamples  = qSettings->value("yMeanSamples").toInt();
        plotCurveConf->yMinimum      = qSettings->value("yMinimum").toDouble();
        plotCurveConf->yMaximum      = qSettings->value("yMaximum").toDouble();

        //Stop reading XML block
        qSettings->endGroup();

        m_HistogramSourceConfigs.append(plotCurveConf);

    }
}


HistogramScope::~HistogramScope()
{

}

//TODO: This should probably be a constructor, too.
void HistogramScope::clone(ScopesGeneric *originalScope)
{
//    this->parent() = new HistogramScope();

    HistogramScope *originalHistogramScope = (HistogramScope*) originalScope;

    binWidth = originalHistogramScope->binWidth;
    maxNumberOfBins = originalHistogramScope->maxNumberOfBins;
    m_refreshInterval = originalHistogramScope->m_refreshInterval;

    int histogramSourceCount = originalHistogramScope->m_HistogramSourceConfigs.size();

    for(int i = 0; i < histogramSourceCount; i++)
    {
        Plot2dCurveConfiguration *currentHistogramSourceConf = originalHistogramScope->m_HistogramSourceConfigs.at(i);
        Plot2dCurveConfiguration *newHistogramSourceConf     = new Plot2dCurveConfiguration();

        newHistogramSourceConf->uavObjectName = currentHistogramSourceConf->uavObjectName;
        newHistogramSourceConf->uavFieldName  = currentHistogramSourceConf->uavFieldName;
        newHistogramSourceConf->color         = currentHistogramSourceConf->color;
        newHistogramSourceConf->yScalePower   = currentHistogramSourceConf->yScalePower;
        newHistogramSourceConf->yMeanSamples  = currentHistogramSourceConf->yMeanSamples;
        newHistogramSourceConf->mathFunction  = currentHistogramSourceConf->mathFunction;
        newHistogramSourceConf->yMinimum = currentHistogramSourceConf->yMinimum;
        newHistogramSourceConf->yMaximum = currentHistogramSourceConf->yMaximum;

        m_HistogramSourceConfigs.append(newHistogramSourceConf);
    }
}

void HistogramScope::saveConfiguration(QSettings* qSettings)
{
    //Stop writing XML blocks
    qSettings->beginGroup(QString("plot2d"));
//    qSettings->beginGroup(QString("histogram"));

    qSettings->setValue("plot2dType", m_plot2dType);
    qSettings->setValue("binWidth", binWidth);
    qSettings->setValue("maxNumberOfBins", maxNumberOfBins);

    int dataSourceCount = m_HistogramSourceConfigs.size();
    qSettings->setValue("dataSourceCount", dataSourceCount);

    // For each curve source in the plot
    for(int i = 0; i < dataSourceCount; i++)
    {
        Plot2dCurveConfiguration *plotCurveConf = m_HistogramSourceConfigs.at(i); //TODO: Understand why this seems to be grabbing i-1
        qSettings->beginGroup(QString("histogramDataSource") + QString().number(i));

        qSettings->setValue("uavObject",  plotCurveConf->uavObjectName);
        qSettings->setValue("uavField",  plotCurveConf->uavFieldName);
        qSettings->setValue("color",  plotCurveConf->color);
        qSettings->setValue("mathFunction",  plotCurveConf->mathFunction);
        qSettings->setValue("yScalePower",  plotCurveConf->yScalePower);
        qSettings->setValue("yMeanSamples",  plotCurveConf->yMeanSamples);
        qSettings->setValue("yMinimum",  plotCurveConf->yMinimum);
        qSettings->setValue("yMaximum",  plotCurveConf->yMaximum);

        //Stop writing XML blocks
        qSettings->endGroup();
    }

    //Stop writing XML block
//    qSettings->endGroup();
    qSettings->endGroup();

}


/**
 * @brief HistogramScope::replaceHistogramSource Replaces the list of histogram data sources
 * @param histogramSourceConfigs
 */
void HistogramScope::replaceHistogramDataSource(QList<Plot2dCurveConfiguration*> histogramSourceConfigs)
{
    m_HistogramSourceConfigs.clear();
    m_HistogramSourceConfigs.append(histogramSourceConfigs);
}


/**
 * @brief HistogramScope::loadConfiguration loads the plot configuration into the scope gadget widget
 * @param scopeGadgetWidget
 */
void HistogramScope::loadConfiguration(ScopeGadgetWidget **scopeGadgetWidget)
{
    (*scopeGadgetWidget)->setupHistogramPlot();
    (*scopeGadgetWidget)->setRefreshInterval(m_refreshInterval);

    // Configured each data source
    foreach (Plot2dCurveConfiguration* histogramDataSourceConfig,  m_HistogramSourceConfigs)
    {
        QString uavObjectName = histogramDataSourceConfig->uavObjectName;
        QString uavFieldName = histogramDataSourceConfig->uavFieldName;
        int scale = histogramDataSourceConfig->yScalePower;
        int mean = histogramDataSourceConfig->yMeanSamples;
        QString mathFunction = histogramDataSourceConfig->mathFunction;
        QRgb color = histogramDataSourceConfig->color;

        // TODO: It bothers me that I have to have the ScopeGadgetWidget scopeGadgetWidget in order to call getUavObjectFieldUnits()
        // Get and store the units
        units = (*scopeGadgetWidget)->getUavObjectFieldUnits(uavObjectName, uavFieldName);

        // Create the Qwt histogram plot
        (*scopeGadgetWidget)->addHistogram(
                uavObjectName,
                    uavFieldName,
                    binWidth,
                    maxNumberOfBins,
                    scale,
                    mean,
                    mathFunction,
                    QBrush(QColor(color))
                    );
    }
}
