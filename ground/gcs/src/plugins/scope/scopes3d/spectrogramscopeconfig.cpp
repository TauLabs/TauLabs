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


SpectrogramScope::SpectrogramScope(Ui::ScopeGadgetOptionsPage *options_page)
{
    bool parseOK = false;

    windowWidth = options_page->sbSpectrogramWidth->value();
    samplingFrequency = options_page->sbSpectrogramFrequency->value();
    timeHorizon = options_page->sbSpectrogramTimeHorizon->value();
    zMaximum = options_page->spnMaxSpectrogramZ->value();


    Plot3dCurveConfiguration* newPlotCurveConfigs = new Plot3dCurveConfiguration();
    newPlotCurveConfigs->uavObjectName = options_page->cmbUAVObjectsSpectrogram->currentText();
    newPlotCurveConfigs->uavFieldName  = options_page->cmbUavoFieldSpectrogram->currentText();
    newPlotCurveConfigs->yScalePower   = options_page->sbSpectrogramDataMultiplier->value();
    newPlotCurveConfigs->yMeanSamples  = options_page->spnMeanSamplesSpectrogram->value();
    newPlotCurveConfigs->mathFunction  = options_page->cmbMathFunctionSpectrogram->currentText();

    QVariant varColor = (int)QColor(options_page->btnColorSpectrogram->text()).rgb();
    int rgb = varColor.toInt(&parseOK);
    if(!parseOK)
        newPlotCurveConfigs->color = QColor(Qt::red).rgb();
    else
        newPlotCurveConfigs->color = (QRgb) rgb;

    m_spectrogramSourceConfigs.append(newPlotCurveConfigs);

}

SpectrogramScope::~SpectrogramScope()
{

}


ScopesGeneric* SpectrogramScope::cloneScope(ScopesGeneric *originalScope)
{
    SpectrogramScope *originalHistogramScope = (SpectrogramScope*) originalScope;
    SpectrogramScope *cloneObj = new SpectrogramScope();

    cloneObj->timeHorizon = originalHistogramScope->timeHorizon;

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

        cloneObj->m_spectrogramSourceConfigs.append(newSpectrogramConf);
    }

    return cloneObj;
}


void SpectrogramScope::saveConfiguration(QSettings* qSettings)
{
    //Start writing new XML block
    qSettings->beginGroup(QString("plot3d"));

    int plot3dCurveCount = m_spectrogramSourceConfigs.size();

    qSettings->setValue("plot3dType", SPECTROGRAM);
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


void SpectrogramScope::setGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page)
{
    //Set the tab widget to 3D
    options_page->tabWidget2d3d->setCurrentWidget(options_page->tabPlot3d);

    //Set the plot type
    options_page->cmb3dPlotType->setCurrentIndex(options_page->cmb3dPlotType->findData(getScopeType()));

    options_page->sbSpectrogramTimeHorizon->setValue(timeHorizon);
    options_page->sbSpectrogramFrequency->setValue(samplingFrequency);
    options_page->spnMaxSpectrogramZ->setValue(zMaximum);

    foreach (Plot3dCurveConfiguration* plot3dData,  m_spectrogramSourceConfigs) {
        int uavoIdx= options_page->cmbUAVObjectsSpectrogram->findText(plot3dData->uavObjectName);
        options_page->cmbUAVObjectsSpectrogram->setCurrentIndex(uavoIdx);
//        on_cmbUAVObjectsSpectrogram_currentIndexChanged(plot3dData->uavObjectName);
        options_page->sbSpectrogramWidth->setValue(windowWidth);

        int uavoFieldIdx= options_page->cmbUavoFieldSpectrogram->findText(plot3dData->uavFieldName);
        options_page->cmbUavoFieldSpectrogram->setCurrentIndex(uavoFieldIdx);
    }

}
