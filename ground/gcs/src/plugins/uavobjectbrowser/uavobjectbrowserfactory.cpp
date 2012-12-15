/**
 ******************************************************************************
 *
 * @file       uavobjectbrowserfactory.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVObjectBrowserPlugin UAVObject Browser Plugin
 * @{
 * @brief The UAVObject Browser gadget plugin
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
#include "uavobjectbrowserfactory.h"
#include "uavobjectbrowserwidget.h"
#include "uavobjectbrowser.h"
#include "uavobjectbrowserconfiguration.h"
#include "uavobjectbrowseroptionspage.h"
#include <coreplugin/iuavgadget.h>

UAVObjectBrowserFactory::UAVObjectBrowserFactory(QObject *parent) :
        IUAVGadgetFactory(QString("UAVObjectBrowser"), tr("UAVObject Browser"), parent)
{
}

UAVObjectBrowserFactory::~UAVObjectBrowserFactory()
{
}

Core::IUAVGadget* UAVObjectBrowserFactory::createGadget(QWidget *parent)
{
    UAVObjectBrowserWidget* gadgetWidget = new UAVObjectBrowserWidget(parent);
    return new UAVObjectBrowser(QString("UAVObjectBrowser"), gadgetWidget, parent);
}

IUAVGadgetConfiguration *UAVObjectBrowserFactory::createConfiguration(QSettings* qSettings)
{
    return new UAVObjectBrowserConfiguration(QString("UAVObjectBrowser"), qSettings);
}


IOptionsPage *UAVObjectBrowserFactory::createOptionsPage(IUAVGadgetConfiguration *config)
{
    return new UAVObjectBrowserOptionsPage(qobject_cast<UAVObjectBrowserConfiguration*>(config));
}

