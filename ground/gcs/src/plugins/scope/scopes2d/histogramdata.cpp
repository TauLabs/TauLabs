/**
 ******************************************************************************
 *
 * @file       histogramdata.cpp
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

#include "scopes2d/histogramdata.h"

HistogramData::HistogramData(QString uavObject, QString uavField, double binWidth, uint numberOfBins) :
    Plot2dData(uavObject, uavField),
    histogram(0),
    histogramBins(0),
    histogramInterval(0),
    intervalSeriesData(0)
{
    this->binWidth = binWidth;
    this->numberOfBins = numberOfBins;
    scalePower = 1;
}


void HistogramData::saveConfiguration(/*QSettings* qSettings*/)
{
//    qSettings->setValue("binWidth", binWidth);
//    qSettings->setValue("windowWidth", numberOfBins);

//    // For each curve source in the plot
//    for(int i = 0; i < plot2dCurveCount; i++)
//    {
//        Plot2dCurveConfiguration *plotCurveConf = m_Plot2dCurveConfigs.at(i);
//        qSettings->beginGroup(QString("plot2dHistogram") + QString().number(i));

//        qSettings->setValue("uavObject",  plotCurveConf->uavObjectName);
//        qSettings->setValue("uavField",  plotCurveConf->uavFieldName);
//        qSettings->setValue("color",  plotCurveConf->color);
//        qSettings->setValue("mathFunction",  plotCurveConf->mathFunction);
//        qSettings->setValue("yScalePower",  plotCurveConf->yScalePower);
//        qSettings->setValue("yMeanSamples",  plotCurveConf->yMeanSamples);
//        qSettings->setValue("yMinimum",  plotCurveConf->yMinimum);
//        qSettings->setValue("yMaximum",  plotCurveConf->yMaximum);

//        //Stop writing XML block
//        qSettings->endGroup();
//    }

}
