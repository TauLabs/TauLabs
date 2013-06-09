/**
 ******************************************************************************
 *
 * @file       scopegadgetconfiguration.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

/**
 * @brief ScopeGadgetConfiguration::ScopeGadgetConfiguration Constructor for scope gadget settings
 * @param classId
 * @param qSettings Settings file
 * @param parent
 */
ScopeGadgetConfiguration::ScopeGadgetConfiguration(QString classId, QSettings* qSettings, QObject *parent) :
        IUAVGadgetConfiguration(classId, parent),
        m_scope(0)
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
        default:
        {
            //Start reading new XML block
            qSettings->beginGroup(QString("plot2d"));

            Scopes2dConfig::Plot2dType plot2dType = (Scopes2dConfig::Plot2dType) qSettings->value("plot2dType").toUInt();
            switch (plot2dType){
            case Scopes2dConfig::HISTOGRAM: {
                m_scope = new HistogramScopeConfig(qSettings);
                break;
                }
            case Scopes2dConfig::SCATTERPLOT2D:
            default: {
                m_scope = new Scatterplot2dScopeConfig(qSettings);
                break;
                }
            }

            //Stop reading XML block
            qSettings->endGroup();

            break;
            }
        case PLOT3D: {
            //Start reading new XML block
            qSettings->beginGroup(QString("plot3d"));

            Scopes3dConfig::Plot3dType plot3dType = (Scopes3dConfig::Plot3dType) qSettings->value("plot3dType").toUInt(); //<--TODO: This requires that the enum values be defined at 0,1,...n
            switch (plot3dType){
            default:
            case Scopes3dConfig::SPECTROGRAM: {
                m_scope = new SpectrogramScopeConfig(qSettings);
                break;
                }
            }

            //Stop reading XML block
            qSettings->endGroup();

            break;
            }
        }
        m_scope->setRefreshInterval(refreshInterval);
    }
    else{
        // Default config is just a simple 2D scatterplot
        m_scope = new Scatterplot2dScopeConfig();
    }
}


/**
 * @brief ScopeGadgetConfiguration::applyGuiConfiguration Uses GUI information to create new scopes
 * @param options_page
 */
void ScopeGadgetConfiguration::applyGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page)
{
    //Default for scopes
    int refreshInterval = 50;

    if(options_page->tabWidget2d3d->currentWidget() == options_page->tabPlot2d)
    {   //--- 2D ---//
        Scopes2dConfig::Plot2dType plot2dType = (Scopes2dConfig::Plot2dType) options_page->cmb2dPlotType->itemData(options_page->cmb2dPlotType->currentIndex()).toUInt(); //This is safe because the item data is defined from the enum.
        switch (plot2dType){
        case Scopes2dConfig::HISTOGRAM: {
            m_scope = new HistogramScopeConfig(options_page);
            break;
            }
        case Scopes2dConfig::SCATTERPLOT2D:
        default: {
            m_scope = new Scatterplot2dScopeConfig(options_page);
            break;
            }
        }

    }
    else if(options_page->tabWidget2d3d->currentWidget() == options_page->tabPlot3d)
    {   //--- 3D ---//

        if (options_page->stackedWidget3dPlots->currentWidget() == options_page->sw3dSpectrogramStack)
        {
            m_scope = new SpectrogramScopeConfig(options_page);
        }
        else if (options_page->stackedWidget3dPlots->currentWidget() == options_page->sw3dTimeSeriesStack)
        {
        }
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
    m->m_scope=this->getScope()->cloneScope(m_scope);

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
