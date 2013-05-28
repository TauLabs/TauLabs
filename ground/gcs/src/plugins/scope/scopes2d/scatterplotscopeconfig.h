/**
 ******************************************************************************
 *
 * @file       scatterplotscopeconfig.h
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

#ifndef SCATTERPLOTSCOPECONFIG_H
#define SCATTERPLOTSCOPECONFIG_H

#include "scopes2d/scopes2dconfig.h"


/**
 * @brief The Scatterplot2dScopeConfig class The scatterplot scope configuration
 */
class Scatterplot2dScopeConfig : public Scopes2dConfig
{
    Q_OBJECT
public:
    /**
     * @brief The Scatterplot2dType enum Defines the different type of plots.
     */
    enum Scatterplot2dType {
        SERIES2D,
        TIMESERIES2D
    };



    Scatterplot2dScopeConfig();
    Scatterplot2dScopeConfig(QSettings *qSettings);
    Scatterplot2dScopeConfig(Ui::ScopeGadgetOptionsPage *options_page);
    ~Scatterplot2dScopeConfig();

    virtual void saveConfiguration(QSettings* qSettings);
    void create(QSettings qSettings);

    QList<Plot2dCurveConfiguration*> getScatterplotDataSource(){return m_scatterplotSourceConfigs;}
    void addScatterplotDataSource(Plot2dCurveConfiguration* value){m_scatterplotSourceConfigs.append(value);}
    void replaceScatterplotDataSource(QList<Plot2dCurveConfiguration*> scatterplotSourceConfigs);

    //Getter functions
    virtual int getScopeType(){return (int) SCATTERPLOT2D;}
    double getTimeHorizon(){return timeHorizon;}
    virtual QList<Plot2dCurveConfiguration*> getDataSourceConfigs(){return m_scatterplotSourceConfigs;}
    Scatterplot2dType getScatterplot2dType(){return scatterplot2dType;}

    //Setter functions
    void setTimeHorizon(double val){timeHorizon = val;}
    void setScatterplot2dType(Scatterplot2dType val){scatterplot2dType = val;}
    virtual void setGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page);

    virtual ScopeConfig* cloneScope(ScopeConfig *Scatterplot2dScopeConfig);

    virtual void loadConfiguration(ScopeGadgetWidget *scopeGadgetWidget);
    virtual void preparePlot(ScopeGadgetWidget *);
    void configureAxes(ScopeGadgetWidget *);

private:
    Scatterplot2dType scatterplot2dType;
    double timeHorizon;

    QList<Plot2dCurveConfiguration*> m_scatterplotSourceConfigs;

private slots:

};

#endif // SCATTERPLOTSCOPECONFIG_H
