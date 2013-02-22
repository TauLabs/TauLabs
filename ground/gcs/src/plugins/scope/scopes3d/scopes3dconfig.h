/**
 ******************************************************************************
 *
 * @file       scopes3d.h
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

#ifndef SCOPES3D_H
#define SCOPES3D_H

#include "scopesconfig.h"
#include "plotdata3d.h"

#include <coreplugin/iuavgadgetconfiguration.h>
#include "scopegadgetwidget.h"
#include "ui_scopegadgetoptionspage.h"

// This struct holds the configuration for individual 2D data sources
struct Plot3dCurveConfiguration
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
class Scopes3d : public ScopesGeneric
{
    Q_OBJECT
public:
    virtual void saveConfiguration(QSettings *qSettings) = 0;
    virtual PlotDimensions getPlotDimensions() {return PLOT3D;}
    virtual int getScopeType(){};
    virtual int getScopeDimensions(){return PLOT3D;}
    virtual QList<Plot3dCurveConfiguration*> getDataSourceConfigs() = 0;
    virtual void loadConfiguration(ScopeGadgetWidget **scopeGadgetWidget) = 0;
    virtual void setGuiConfiguration(Ui::ScopeGadgetOptionsPage *) = 0;
    virtual void clone(ScopesGeneric *){};
    virtual ScopesGeneric* cloneScope(ScopesGeneric *) = 0;

protected:
    PlotDimensions m_plotDimensions;
private:
};

#endif // SCOPES3D_H
