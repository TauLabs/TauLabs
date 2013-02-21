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

#ifndef SCOPES_H
#define SCOPES_H

#include <coreplugin/iuavgadgetconfiguration.h>
//#include "scopegadgetwidget.h"
#include "qwt/src/qwt_color_map.h"
#include "scopegadgetwidget.h"

/**
 * @brief The HistogramScope class The histogram scope has a variable sized list of
 * data sources
 */

class ScopesGeneric : public QObject
{
    Q_OBJECT
public:
    int getScopeDimensions(){return m_plotDimensions;}
    void setScopeDimensions(int val){m_plotDimensions = val;}
    virtual void saveConfiguration(QSettings *qSettings) = 0;
    virtual int getScopeType(){}
    virtual void setScopeType(int){}
    virtual void loadConfiguration(ScopeGadgetWidget **scopeGadgetWidget){}
    virtual void clone(ScopesGeneric *){}

    int getRefreshInterval(){return m_refreshInterval;}
    void setRefreshInterval(int val){m_refreshInterval = val;}
//    virtual QList<Plot2dCurveConfiguration*> getDataSourceConfigs(){}

protected:
    int m_plotDimensions;
    int m_refreshInterval; //The interval to replot the curve widget. The data buffer is refresh as the data comes in.
private:
};

#endif // SCOPES_H
