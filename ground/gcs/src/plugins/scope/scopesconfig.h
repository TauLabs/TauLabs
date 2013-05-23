/**
 ******************************************************************************
 *
 * @file       scopes2dconfig.h
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

#ifndef SCOPESCONFIG_H
#define SCOPESCONFIG_H


#include "uavtalk/telemetrymanager.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"

#include <coreplugin/iuavgadgetconfiguration.h>
#include "qwt/src/qwt_color_map.h"
#include "scopegadgetwidget.h"
#include "ui_scopegadgetoptionspage.h"



/**
 * @brief The Plot3dType enum Defines the different type of plots.
 */
enum PlotDimensions {
    PLOT2D,
    PLOT3D
};

/**
 * @brief The ScopeConfig class The parent class for scope configuration classes
 * data sources
 */
class ScopeConfig : public QObject
{
    Q_OBJECT
public:
    virtual int getScopeDimensions() = 0;
    virtual void saveConfiguration(QSettings *qSettings) = 0;
    virtual int getScopeType() = 0;
    virtual void loadConfiguration(ScopeGadgetWidget *) = 0;
    virtual void setGuiConfiguration(Ui::ScopeGadgetOptionsPage *) = 0;

    int getRefreshInterval(){return m_refreshInterval;}
    void setRefreshInterval(int val){m_refreshInterval = val;}

    virtual void preparePlot(ScopeGadgetWidget *) = 0;
    virtual ScopeConfig* cloneScope(ScopeConfig *histogramSourceConfigs) = 0;

protected:
    int m_refreshInterval; //The interval to replot the curve widget. The data buffer is refresh as the data comes in.
    PlotDimensions m_plotDimensions;

    QMutex mutex;
    QString getUavObjectFieldUnits(QString uavObjectName, QString uavObjectFieldName)
    {
        //Get the uav object
        ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
        UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
        UAVDataObject* obj = dynamic_cast<UAVDataObject*>(objManager->getObject(uavObjectName));
        if(!obj) {
            qDebug() << "In scope gadget, UAVObject " << uavObjectName << " is missing";
            return "";
        }
        UAVObjectField* field = obj->getField(uavObjectFieldName);
        if(!field) {
            qDebug() << "In scope gadget, in fields loaded from GCS config file, field" << uavObjectFieldName << " of UAVObject " << uavObjectName << " is missing";
            return "";
        }

        //Get the units
        QString units = field->getUnits();
        if(units == 0)
            units = QString();

        return units;
    }
};

#endif // SCOPESCONFIG_H
