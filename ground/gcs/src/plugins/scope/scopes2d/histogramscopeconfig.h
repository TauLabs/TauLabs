/**
 ******************************************************************************
 *
 * @file       histogramscopeconfig.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

#ifndef HISTOGRAMSCOPECONFIG_H
#define HISTOGRAMSCOPECONFIG_H

#include "scopes2d/scopes2dconfig.h"


/**
 * @brief The HistogramScopeConfig class The histogram scope configuration
 */
class HistogramScopeConfig : public Scopes2dConfig
{
    Q_OBJECT
public:
    HistogramScopeConfig();
    HistogramScopeConfig(QSettings *qSettings);
    HistogramScopeConfig(Ui::ScopeGadgetOptionsPage *options_page);
    ~HistogramScopeConfig();

    virtual void saveConfiguration(QSettings* qSettings);
    void create(QSettings qSettings);

    QList<Plot2dCurveConfiguration*> getHistogramDataSource(){return m_HistogramSourceConfigs;}
    void addHistogramDataSource(Plot2dCurveConfiguration* value){m_HistogramSourceConfigs.append(value);}
    void replaceHistogramDataSource(QList<Plot2dCurveConfiguration*> histogramSourceConfigs);

    //Getter functions
    virtual int getScopeType(){return (int) HISTOGRAM;}
    double getBinWidth(){return binWidth;}
    unsigned int getMaxNumberOfBins(){return maxNumberOfBins;}
    virtual QList<Plot2dCurveConfiguration*> getDataSourceConfigs(){return m_HistogramSourceConfigs;}

    //Setter functions
    void setBinWidth(double val){binWidth = val;}
    void setMaxNumberOfBins(unsigned int val){maxNumberOfBins = val;}

    virtual ScopeConfig* cloneScope(ScopeConfig *histogramSourceConfigs);

    virtual void setGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page);

    virtual void loadConfiguration(ScopeGadgetWidget *scopeGadgetWidget);
    virtual void preparePlot(ScopeGadgetWidget *);
    void configureAxes(ScopeGadgetWidget *);

private:
    double binWidth;
    unsigned int maxNumberOfBins;
    QString units;

    QList<Plot2dCurveConfiguration*> m_HistogramSourceConfigs;

private slots:

};

#endif // HISTOGRAMSCOPECONFIG_H
