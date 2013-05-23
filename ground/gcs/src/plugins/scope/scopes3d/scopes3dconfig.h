/**
 ******************************************************************************
 *
 * @file       scopes3dconfig.h
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

#ifndef SCOPES3DCONFIG_H
#define SCOPES3DCONFIG_H

#include "scopesconfig.h"
#include "scopes3d/plotdata3d.h"

#include <coreplugin/iuavgadgetconfiguration.h>
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
};

/**
 * @brief The Scopes3dConfig class The parent class for 3D scope configurations
 */
class Scopes3dConfig : public ScopeConfig
{
    Q_OBJECT
public:
    /**
     * @brief The Plot3dType enum Defines the different type of plots.
     */
    enum Plot3dType {
        NO3DPLOT,
        SCATTERPLOT3D,
        SPECTROGRAM
    };

    virtual int getScopeDimensions(){return PLOT3D;}

private:
};

#endif // SCOPES3DCONFIG_H
