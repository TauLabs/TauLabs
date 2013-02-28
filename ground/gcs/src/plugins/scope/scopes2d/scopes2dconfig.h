/**
 ******************************************************************************
 *
 * @file       scopes2d.h
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

#ifndef SCOPES2D_H
#define SCOPES2D_H


#include "scopesconfig.h"
#include "scopes2d/plotdata2d.h"

#include <coreplugin/iuavgadgetconfiguration.h>
#include "ui_scopegadgetoptionspage.h"


// This struct holds the configuration for individual 2D data sources
struct Plot2dCurveConfiguration
{
    QString uavObjectName;
    QString uavFieldName;
    int yScalePower; //This is the power to which each value must be raised
    QRgb color;
    int yMeanSamples;
    QString mathFunction;
    double yMinimum;
    double yMaximum;
};

/**
 * @brief The HistogramScope class The histogram scope has a variable sized list of
 * data sources
 */
class Scopes2d : public ScopesGeneric
{
    Q_OBJECT
public:
    /**
     * @brief The Plot2dType enum Defines the different type of plots.
     */
    enum Plot2dType {
        NO2DPLOT, //Signifies that there is no 2D plot configured
        SCATTERPLOT2D,
        HISTOGRAM,
        POLARPLOT
    };

    virtual void saveConfiguration(QSettings *qSettings) = 0;
    virtual int getScopeType() = 0;
    virtual int getScopeDimensions(){return PLOT2D;}
    virtual QList<Plot2dCurveConfiguration*> getDataSourceConfigs() = 0;
    virtual void loadConfiguration(ScopeGadgetWidget *scopeGadgetWidget) = 0;
    virtual void setGuiConfiguration(Ui::ScopeGadgetOptionsPage *) = 0;
    virtual ScopesGeneric* cloneScope(ScopesGeneric *) = 0;

    virtual void preparePlot(ScopeGadgetWidget *) = 0;
    virtual void plotNewData(ScopeGadgetWidget *) = 0;
    virtual void clearPlots(ScopeGadgetWidget *scopeGadgetWidget) = 0;

private:
};

#endif // SCOPES2D_H
