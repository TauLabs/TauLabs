/**
 ******************************************************************************
 *
 * @file       scatterplotdata.h
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

#ifndef SCATTERPLOTSCOPE_H
#define SCATTERPLOTSCOPE_H

#include "scopes2d/scopes2dconfig.h"


///**
// * @brief The Scatterplot2dType enum Defines the different type of plots.
// */
//enum Scatterplot2dType {
//    SERIES2D,
//    TIMESERIES2D
//};


/**
 * @brief The Scatterplot2dScope class The scatterplot scope has a variable sized list of
 * data sources
 */
class Scatterplot2dScope : public Scopes2d
{
    Q_OBJECT
public:
    Scatterplot2dScope();
    Scatterplot2dScope(QSettings *qSettings);
    Scatterplot2dScope(Ui::ScopeGadgetOptionsPage *options_page);
    ~Scatterplot2dScope();

    virtual void saveConfiguration(QSettings* qSettings);
    void create(QSettings qSettings);

    QList<Plot2dCurveConfiguration*> getScatterplotDataSource(){return m_scatterplotSourceConfigs;}
    void addScatterplotDataSource(Plot2dCurveConfiguration* value){m_scatterplotSourceConfigs.append(value);}
    void replaceScatterplotDataSource(QList<Plot2dCurveConfiguration*> scatterplotSourceConfigs);

    //Getter functions
    virtual int getScopeType(){return (int) SCATTERPLOT2D;} //TODO: Fix this. It should return the true value, not HISTOGRAM
    double getTimeHorizon(){return timeHorizon;}
    QString getXAxisUnits(){return xAxisUnits;}
    virtual QList<Plot2dCurveConfiguration*> getDataSourceConfigs(){return m_scatterplotSourceConfigs;}
    Scatterplot2dType getScatterplot2dType(){return scatterplot2dType;}

    //Setter functions
    void setTimeHorizon(double val){timeHorizon = val;}
    void setXAxisUnits(QString val){xAxisUnits = val;}
    void setScatterplot2dType(Scatterplot2dType val){scatterplot2dType = val;}
    virtual void loadConfiguration(ScopeGadgetWidget **scopeGadgetWidget);
    virtual void setGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page);

    virtual ScopesGeneric* cloneScope(ScopesGeneric *Scatterplot2dScope);


private:
    Scatterplot2dType scatterplot2dType;
    QString xAxisUnits; //TODO: Remove this once 2d scatterplot is completed
    double timeHorizon;

    QList<Plot2dCurveConfiguration*> m_scatterplotSourceConfigs;

private slots:

};

#endif // SCATTERPLOTSCOPE_H
