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

#include "scopes2d/scatterplotscopeconfig.h"
#include "scopes2d/histogramscopeconfig.h"
#include "scopes3d/spectrogramscopeconfig.h"
#include "scopegadgetconfiguration.h"

ScopeGadgetConfiguration::ScopeGadgetConfiguration(QString classId, QSettings* qSettings, QObject *parent) :
        IUAVGadgetConfiguration(classId, parent)
{
    //Default for scopes
    int refreshInterval = 50;

    //if a saved configuration exists load it
    if(qSettings != 0)
    {

        PlotDimensions plotDimensions =  (PlotDimensions) qSettings->value("plotDimensions").toInt();

        switch (plotDimensions)
        {
        case PLOT2D:
        {
            //Start reading new XML block
            qSettings->beginGroup(QString("plot2d"));

            Plot2dType plot2dType = (Plot2dType) qSettings->value("plot2dType").toUInt();
            switch (plot2dType){
            case SCATTERPLOT2D: {
                m_scope = new Scatterplot2dScope(qSettings);
                break;
            }
            case HISTOGRAM: {
                m_scope = new HistogramScope(qSettings);
                break;
            }
            default:
                //We shouldn't be able to get this far
                Q_ASSERT(0);
            }

            //Stop reading XML block
            qSettings->endGroup();

            break;
        }
        case PLOT3D:
        {
            //Start reading new XML block
            qSettings->beginGroup(QString("plot3d"));

            Plot3dType plot3dType = (Plot3dType) qSettings->value("plot3dType").toUInt(); //<--TODO: This requires that the enum values be defined at 0,1,...n
            switch (plot3dType){
            case SCATTERPLOT3D:
            {
//                m_scope = new Scatterplot3dScope(qSettings);
                break;
            }
            case SPECTROGRAM:
            {
                m_scope = new SpectrogramScope(qSettings);
                break;
            }
            default:
                //We shouldn't be able to get this far
                Q_ASSERT(0);
            }

            //Stop reading XML block
            qSettings->endGroup();

            break;
        }
        default:
            //We shouldn't be able to get this far
            Q_ASSERT(0);
        }

        m_scope->setRefreshInterval(refreshInterval);

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


void ScopeGadgetConfiguration::applyGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page)
{
    delete m_scope;

    //Default for scopes
    int refreshInterval = 50;

    if(options_page->tabWidget2d3d->currentWidget() == options_page->tabPlot2d)
    {   //--- 2D ---//
        Plot2dType plot2dType = (Plot2dType) options_page->cmb2dPlotType->itemData(options_page->cmb2dPlotType->currentIndex()).toUInt(); //This is safe because the int value is defined from the enum.
        switch (plot2dType){
        case SCATTERPLOT2D: {
            m_scope = new Scatterplot2dScope(options_page);
            break;
        }
        case HISTOGRAM: {
            m_scope = new HistogramScope(options_page);
            break;
        }
        default:
            //We shouldn't be able to get this far
            Q_ASSERT(0);
        }

    }
    else if(options_page->tabWidget2d3d->currentWidget() == options_page->tabPlot3d)
    {   //--- 3D ---//

        Plot3dType plot3dType = (Plot3dType) options_page->cmb3dPlotType->itemData(options_page->cmb3dPlotType->currentIndex()).toUInt(); //This is safe because the int value is defined from the enum

        if (options_page->stackedWidget3dPlots->currentWidget() == options_page->sw3dSpectrogramStack)
        {
            m_scope = new SpectrogramScope(options_page);
        }
        else if (options_page->stackedWidget3dPlots->currentWidget() == options_page->sw3dTimeSeriesStack)
        {
//            m_scope = new Scatterplot3dScope(options_page);
        }
        else{
            Q_ASSERT(0);
        }

    }
    else{
        Q_ASSERT(0);
    }

    m_scope->setRefreshInterval(refreshInterval);

}


/**
 * @brief ScopeGadgetConfiguration::~ScopeGadgetConfiguration Destructor clears 2D and 3D plot data
 */
ScopeGadgetConfiguration::~ScopeGadgetConfiguration()
{
}


/**
 * @brief ScopeGadgetConfiguration::clone Clones a configuration.
 * @return
 */
IUAVGadgetConfiguration *ScopeGadgetConfiguration::clone()
{
    ScopeGadgetConfiguration *m = new ScopeGadgetConfiguration(this->classId());
    m->clone(m_scope); //TODO: Fix this, it's broken. I need to instantiate the m Class properly.

    return m;
}


/**
 * @brief ScopeGadgetConfiguration::saveConfig Saves a configuration. //REDEFINES saveConfig CHILD BEHAVIOR?
 * @param qSettings
 */
void ScopeGadgetConfiguration::saveConfig(QSettings* qSettings) const {
    qSettings->setValue("plotDimensions", m_scope->getScopeDimensions());
    qSettings->setValue("refreshInterval", m_scope->getRefreshInterval());

    m_scope->saveConfiguration(qSettings);
}
