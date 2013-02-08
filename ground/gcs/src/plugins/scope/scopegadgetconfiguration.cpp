/**
 ******************************************************************************
 *
 * @file       scopegadgetconfiguration.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ScopePlugin Scope Gadget Plugin
 * @{
 * @brief The scope gadget configuration, sets up the configuration for one single scope.
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

#include "scopegadgetconfiguration.h"

ScopeGadgetConfiguration::ScopeGadgetConfiguration(QString classId, QSettings* qSettings, QObject *parent) :
        IUAVGadgetConfiguration(classId, parent),
        m_SpectrogramConfig(0),
        m_HistogramConfig(0)
{
    //Defaults for unconfigured scope
    m_plot2dType = No2dPlot;
    m_scatterplot2dType = TimeSeries2d;
    m_dataSize = 60;
    m_refreshInterval = 50;
    m_timeHorizon = 60;
    m_plotDimensions = Plot2d;

    //if a saved configuration exists load it
    if(qSettings != 0)
    {
        m_plotDimensions =  (PlotDimensions) qSettings->value("plotDimensions").toInt();
        m_timeHorizon = qSettings->value("timeHorizon").toDouble();

        if(m_plotDimensions == Plot2d)
        {
            //Start reading new XML block
            qSettings->beginGroup(QString("plot2d"));

            m_plot2dType = (Plot2dType) qSettings->value("plot2dType").toUInt();
            m_dataSize = qSettings->value("dataSize").toInt();
            m_scatterplot2dType = (Scatterplot2dType) qSettings->value("scatterplot2dType").toUInt();

            int plot2dCurveCount = qSettings->value("plot2dCurveCount").toInt();
            for(int i = 0; i < plot2dCurveCount; i++)
            {
                Plot2dCurveConfiguration *plotCurveConf = new Plot2dCurveConfiguration();

                if (m_plot2dType == Scatterplot2d){

                    //Start reading new XML block
                    qSettings->beginGroup(QString("plot2dScatterplot") + QString().number(i));

                    plotCurveConf->uavObjectName = qSettings->value("uavObject").toString();
                    plotCurveConf->uavFieldName  = qSettings->value("uavField").toString();
                    plotCurveConf->color         = qSettings->value("color").value<QRgb>();
                    plotCurveConf->yScalePower   = qSettings->value("yScalePower").toInt();
                    plotCurveConf->mathFunction  = qSettings->value("mathFunction").toString();
                    plotCurveConf->yMeanSamples  = qSettings->value("yMeanSamples").toInt();
                    plotCurveConf->yMinimum      = qSettings->value("yMinimum").toDouble();
                    plotCurveConf->yMaximum      = qSettings->value("yMaximum").toDouble();

                    //End XML block
                    qSettings->endGroup();
                }
                else if (m_plot2dType == Histogram){
                    m_HistogramConfig = new HistogramDataConfiguration();
                    m_HistogramConfig->binWidth    = qSettings->value("binWidth").toDouble();
                    m_HistogramConfig->windowWidth = qSettings->value("windowWidth").toInt();

                    qSettings->beginGroup(QString("plot2dHistogram") + QString().number(i));

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
                }
                else{
                    //We shouldn't be able to get this far
                    Q_ASSERT(0);
                }

                m_Plot2dCurveConfigs.append(plotCurveConf);
            }
            //Stop reading XML block
            qSettings->endGroup();
        }
        else if(m_plotDimensions == Plot3d){
            //Start reading new XML block
            qSettings->beginGroup(QString("plot3d"));

            int plot3dCurveCount = qSettings->value("plot3dCurveCount").toInt();
            m_plot3dType = (Plot3dType) qSettings->value("plot3dType").toInt(); //<--TODO: This requires that the enum values be defined at 0,1,...n


            if(m_plot3dType == Spectrogram){
                m_SpectrogramConfig = new SpectrogramDataConfiguration();
                m_SpectrogramConfig->samplingFrequency = qSettings->value("samplingFrequency").toDouble();
                m_SpectrogramConfig->windowWidth       = qSettings->value("windowWidth").toInt();
                m_SpectrogramConfig->zMaximum = qSettings->value("zMaximum").toDouble();

                for(int i = 0; i < plot3dCurveCount; i++){
                    Plot3dCurveConfiguration *plotCurveConf = new Plot3dCurveConfiguration();

                    qSettings->beginGroup(QString("plotSpectrogram") + QString().number(i));

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

                    m_Plot3dCurveConfigs.append(plotCurveConf);
                }
            }
            else if(m_plot3dType == Scatterplot3d){
                for(int i = 0; i < plot3dCurveCount; i++)
                {
                    Plot3dCurveConfiguration *plotCurveConf = new Plot3dCurveConfiguration();

                    qSettings->beginGroup(QString("plot3dCurve") + QString().number(i));

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

                    m_Plot3dCurveConfigs.append(plotCurveConf);
                }
            }



            //Stop reading XML block
            qSettings->endGroup();
        }
        else{
            //Whoops, the file must have been corrupted. Set to default.
            m_plotDimensions = Plot2d;
        }
    }
    else{
        //Nothing to do here...
//        // Default config is just a simple 2D scatterplot

//        Plot2dCurveConfiguration *plotCurveConf = new Plot2dCurveConfiguration();
//        plotCurveConf->color = 4294945407;
//        plotCurveConf->mathFunction = "None";
//        plotCurveConf->yMinimum = 0;
//        plotCurveConf->yMaximum = 100;
//        plotCurveConf->yMeanSamples = 1;
//        plotCurveConf->yScalePower = 1;

//        m_Plot2dCurveConfigs.append(plotCurveConf);

    }
}


/**
 * @brief ScopeGadgetConfiguration::~ScopeGadgetConfiguration Destructor clears 2D and 3D plot data
 */
ScopeGadgetConfiguration::~ScopeGadgetConfiguration()
{
    clearPlot2dData();
    clearPlot3dData();
}


/**
 * @brief ScopeGadgetConfiguration::clone Clones a configuration.
 * @return
 */
IUAVGadgetConfiguration *ScopeGadgetConfiguration::clone()
{
    ScopeGadgetConfiguration *m = new ScopeGadgetConfiguration(this->classId());

    if(m_plotDimensions == Plot2d){

        m->setPlot2dType( m_plot2dType);
        m->setDataSize( m_dataSize);
        m->setTimeHorizon( m_timeHorizon);
        m->setRefreshInterval( m_refreshInterval);

        int plotCurveCount = m_Plot2dCurveConfigs.size();

        if (m_plot2dType == Scatterplot2d){
            m->setScatterplot2dType(m_scatterplot2dType);

            for(int i = 0; i < plotCurveCount; i++)
            {
                Plot2dCurveConfiguration *currentPlotCurveConf = m_Plot2dCurveConfigs.at(i);
                Plot2dCurveConfiguration *newPlotCurveConf     = new Plot2dCurveConfiguration();

                newPlotCurveConf->uavObjectName = currentPlotCurveConf->uavObjectName;
                newPlotCurveConf->uavFieldName  = currentPlotCurveConf->uavFieldName;
                newPlotCurveConf->color         = currentPlotCurveConf->color;
                newPlotCurveConf->yScalePower   = currentPlotCurveConf->yScalePower;
                newPlotCurveConf->yMeanSamples  = currentPlotCurveConf->yMeanSamples;
                newPlotCurveConf->mathFunction  = currentPlotCurveConf->mathFunction;
                newPlotCurveConf->yMinimum = currentPlotCurveConf->yMinimum;
                newPlotCurveConf->yMaximum = currentPlotCurveConf->yMaximum;

                m->addPlot2dCurveConfig(newPlotCurveConf);
            }
        }
        else if (m_plot2dType == Histogram){
            HistogramDataConfiguration *newHistogramConfig = new HistogramDataConfiguration();
            newHistogramConfig->binWidth = m_HistogramConfig->binWidth;
            newHistogramConfig->windowWidth = m_HistogramConfig->windowWidth;
            m->replaceHistogramConfig(newHistogramConfig);

            for(int i = 0; i < plotCurveCount; i++)
            {
                Plot2dCurveConfiguration *currentPlotCurveConf = m_Plot2dCurveConfigs.at(i);
                Plot2dCurveConfiguration *newPlotCurveConf     = new Plot2dCurveConfiguration();

                newPlotCurveConf->uavObjectName = currentPlotCurveConf->uavObjectName;
                newPlotCurveConf->uavFieldName  = currentPlotCurveConf->uavFieldName;
                newPlotCurveConf->color         = currentPlotCurveConf->color;
                newPlotCurveConf->yScalePower   = currentPlotCurveConf->yScalePower;
                newPlotCurveConf->yMeanSamples  = currentPlotCurveConf->yMeanSamples;
                newPlotCurveConf->mathFunction  = currentPlotCurveConf->mathFunction;
                newPlotCurveConf->yMinimum = currentPlotCurveConf->yMinimum;
                newPlotCurveConf->yMaximum = currentPlotCurveConf->yMaximum;

                m->addPlot2dCurveConfig(newPlotCurveConf);
            }
        }
        else{
            Q_ASSERT(0);
        }

    }
    else if (m_plotDimensions == Plot3d){
        m->setTimeHorizon( m_timeHorizon);
        m->setPlot3dType( m_plot3dType);

        int plotCurveCount = m_Plot3dCurveConfigs.size();

        if (m_plot3dType == Spectrogram){
            SpectrogramDataConfiguration *newSpectrogramConfig = new SpectrogramDataConfiguration();
            newSpectrogramConfig->samplingFrequency = m_SpectrogramConfig->samplingFrequency;
            newSpectrogramConfig->windowWidth = m_SpectrogramConfig->windowWidth;
            newSpectrogramConfig->zMaximum = m_SpectrogramConfig->zMaximum;
            m->replaceSpectrogramConfig(newSpectrogramConfig);

            for(int i = 0; i < plotCurveCount; i++)
            {
                Plot3dCurveConfiguration *currentPlotCurveConf = m_Plot3dCurveConfigs.at(i);
                Plot3dCurveConfiguration *newPlotCurveConf     = new Plot3dCurveConfiguration();


                newPlotCurveConf->uavObjectName = currentPlotCurveConf->uavObjectName;
                newPlotCurveConf->uavFieldName  = currentPlotCurveConf->uavFieldName;
                newPlotCurveConf->color         = currentPlotCurveConf->color;
                newPlotCurveConf->yScalePower   = currentPlotCurveConf->yScalePower;
                newPlotCurveConf->yMeanSamples  = currentPlotCurveConf->yMeanSamples;
                newPlotCurveConf->mathFunction  = currentPlotCurveConf->mathFunction;

                newPlotCurveConf->yMinimum = currentPlotCurveConf->yMinimum;
                newPlotCurveConf->yMaximum = currentPlotCurveConf->yMaximum;
            }
        }
        else if (m_plot3dType == Scatterplot3d){
            for (int i = 0; i < plotCurveCount; i++){
                Plot3dCurveConfiguration *currentPlotCurveConf = m_Plot3dCurveConfigs.at(i);
                Plot3dCurveConfiguration *newPlotCurveConf     = new Plot3dCurveConfiguration();

                newPlotCurveConf->uavObjectName = currentPlotCurveConf->uavObjectName;
                newPlotCurveConf->uavFieldName  = currentPlotCurveConf->uavFieldName;
                newPlotCurveConf->color         = currentPlotCurveConf->color;
                newPlotCurveConf->yScalePower   = currentPlotCurveConf->yScalePower;
                newPlotCurveConf->yMeanSamples  = currentPlotCurveConf->yMeanSamples;
                newPlotCurveConf->mathFunction  = currentPlotCurveConf->mathFunction;

                newPlotCurveConf->yMinimum = currentPlotCurveConf->yMinimum;
                newPlotCurveConf->yMaximum = currentPlotCurveConf->yMaximum;
            }
        }
//            m->addPlot3dCurveConfig(newPlotCurveConf);
    }

    return m;
}


/**
 * @brief ScopeGadgetConfiguration::saveConfig Saves a configuration. //REDEFINES saveConfig CHILD BEHAVIOR?
 * @param qSettings
 */
void ScopeGadgetConfiguration::saveConfig(QSettings* qSettings) const {
    qSettings->setValue("plotDimensions", m_plotDimensions);
    qSettings->setValue("refreshInterval", m_refreshInterval);
    qSettings->setValue("timeHorizon", m_timeHorizon);

    if(m_plotDimensions == Plot2d)
    {
        //Start writing new XML block
        qSettings->beginGroup(QString("plot2d"));

        int plot2dCurveCount = m_Plot2dCurveConfigs.size();

        qSettings->setValue("plot2dType", m_plot2dType);
        qSettings->setValue("dataSize", m_dataSize);
        qSettings->setValue("plot2dCurveCount", plot2dCurveCount);

        if (m_plot2dType == Scatterplot2d){
            qSettings->setValue("scatterplot2dType", m_scatterplot2dType);

            // For each curve source in the plot
            for(int i = 0; i < plot2dCurveCount; i++)
            {
                Plot2dCurveConfiguration *plotCurveConf = m_Plot2dCurveConfigs.at(i);
                qSettings->beginGroup(QString("plot2dScatterplot") + QString().number(i));

                qSettings->setValue("uavObject",  plotCurveConf->uavObjectName);
                qSettings->setValue("uavField",  plotCurveConf->uavFieldName);
                qSettings->setValue("color",  plotCurveConf->color);
                qSettings->setValue("mathFunction",  plotCurveConf->mathFunction);
                qSettings->setValue("yScalePower",  plotCurveConf->yScalePower);
                qSettings->setValue("yMeanSamples",  plotCurveConf->yMeanSamples);
                qSettings->setValue("yMinimum",  plotCurveConf->yMinimum);
                qSettings->setValue("yMaximum",  plotCurveConf->yMaximum);

                //Stop writing XML block
                qSettings->endGroup();
            }
        }
        else if (m_plot2dType == Histogram){
            qSettings->setValue("binWidth", m_HistogramConfig->binWidth);
            qSettings->setValue("windowWidth", m_HistogramConfig->windowWidth);

            // For each curve source in the plot
            for(int i = 0; i < plot2dCurveCount; i++)
            {
                Plot2dCurveConfiguration *plotCurveConf = m_Plot2dCurveConfigs.at(i);
                qSettings->beginGroup(QString("plot2dHistogram") + QString().number(i));

                qSettings->setValue("uavObject",  plotCurveConf->uavObjectName);
                qSettings->setValue("uavField",  plotCurveConf->uavFieldName);
                qSettings->setValue("color",  plotCurveConf->color);
                qSettings->setValue("mathFunction",  plotCurveConf->mathFunction);
                qSettings->setValue("yScalePower",  plotCurveConf->yScalePower);
                qSettings->setValue("yMeanSamples",  plotCurveConf->yMeanSamples);
                qSettings->setValue("yMinimum",  plotCurveConf->yMinimum);
                qSettings->setValue("yMaximum",  plotCurveConf->yMaximum);

                //Stop writing XML block
                qSettings->endGroup();
            }
        }
        else{
            Q_ASSERT(0);
        }

        //Stop writing XML block
        qSettings->endGroup();
    }
    else if(m_plotDimensions == Plot3d)
    {
        //Start writing new XML block
        qSettings->beginGroup(QString("plot3d"));

        int plot3dCurveCount = m_Plot3dCurveConfigs.size();
        qSettings->setValue("plot3dType", m_plot3dType);
        qSettings->setValue("plot3dCurveCount", plot3dCurveCount);

        if(m_plot3dType == Spectrogram){
            qSettings->setValue("samplingFrequency", m_SpectrogramConfig->samplingFrequency);
            qSettings->setValue("windowWidth", m_SpectrogramConfig->windowWidth);
            qSettings->setValue("zMaximum",  m_SpectrogramConfig->zMaximum);

            for(int i = 0; i < plot3dCurveCount; i++){
                Plot3dCurveConfiguration *plotCurveConf = m_Plot3dCurveConfigs.at(i);

                //Start new XML block
                qSettings->beginGroup(QString("plotSpectrogram") + QString().number(i));

                qSettings->setValue("uavObject",  plotCurveConf->uavObjectName);
                qSettings->setValue("uavField",  plotCurveConf->uavFieldName);
                qSettings->setValue("colormap",  plotCurveConf->color);

                //Stop writing XML block
                qSettings->endGroup();
            }

        }
        else if(m_plot3dType == Scatterplot3d){
            for(int i = 0; i < plot3dCurveCount; i++){
                Plot3dCurveConfiguration *plotCurveConf = m_Plot3dCurveConfigs.at(i);
            //Start new XML block
            qSettings->beginGroup(QString("plot3dCurve") + QString().number(i));

            qSettings->setValue("uavObject",  plotCurveConf->uavObjectName);
            qSettings->setValue("uavField",  plotCurveConf->uavFieldName);
            qSettings->setValue("color",  plotCurveConf->color);
            qSettings->setValue("mathFunction",  plotCurveConf->mathFunction);
            qSettings->setValue("yScalePower",  plotCurveConf->yScalePower);
            qSettings->setValue("yMeanSamples",  plotCurveConf->yMeanSamples);
            qSettings->setValue("yMinimum",  plotCurveConf->yMinimum);
            qSettings->setValue("yMaximum",  plotCurveConf->yMaximum);

            //Stop writing XML block
            qSettings->endGroup();
            }
        }
        else{
            Q_ASSERT(0);
        }

        //Stop writing XML block
        qSettings->endGroup();
    }

}

void ScopeGadgetConfiguration::replacePlot2dCurveConfig(QList<Plot2dCurveConfiguration*> newPlot2dCurveConfigs)
{
    clearPlot2dData();

    m_Plot2dCurveConfigs.append(newPlot2dCurveConfigs);
}


void ScopeGadgetConfiguration::replacePlot3dCurveConfig(QList<Plot3dCurveConfiguration*> newPlot3dCurveConfigs)
{
    clearPlot3dData();

    m_Plot3dCurveConfigs.append(newPlot3dCurveConfigs);
}


//TODO: What is this function doing? Why create the configuration only to deleted it?
// Why can't we accomplish the same thing with a simple m_Plot2dCurveConfigs.clear()?
void ScopeGadgetConfiguration::clearPlot2dData()
{
    Plot2dCurveConfiguration *plotCurveConfig;

    while(m_Plot2dCurveConfigs.size() > 0)
    {
        plotCurveConfig = m_Plot2dCurveConfigs.first();
        m_Plot2dCurveConfigs.pop_front();

        delete plotCurveConfig;
    }
}


//TODO: What is this function doing? Why create the configuration only to deleted it?
// Why can't we accomplish the same thing with a simple m_Plot3dCurveConfigs.clear()?
void ScopeGadgetConfiguration::clearPlot3dData()
{
    Plot3dCurveConfiguration *plotCurveConfig;

    while(m_Plot3dCurveConfigs.size() > 0)
    {
        plotCurveConfig = m_Plot3dCurveConfigs.first();
        m_Plot3dCurveConfigs.pop_front();

        delete plotCurveConfig;
    }
}
