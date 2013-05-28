/**
 ******************************************************************************
 *
 * @file       opmapgadgetfactory.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup OPMapPlugin Tau Labs Map Plugin
 * @{
 * @brief Tau Labs map plugin
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
#include "opmapgadgetfactory.h"
#include "opmapgadgetwidget.h"
#include "opmapgadget.h"
#include "opmapgadgetconfiguration.h"
#include "opmapgadgetoptionspage.h"
#include <coreplugin/iuavgadget.h>

OPMapGadgetFactory::OPMapGadgetFactory(QObject *parent) :
                IUAVGadgetFactory(QString("OPMapGadget"), tr("OPMap"), parent)
{
}

OPMapGadgetFactory::~OPMapGadgetFactory()
{
}

Core::IUAVGadget * OPMapGadgetFactory::createGadget(QWidget *parent)
{
    OPMapGadgetWidget *gadgetWidget = new OPMapGadgetWidget(parent);
    return new OPMapGadget(QString("OPMapGadget"), gadgetWidget, parent);
}

IUAVGadgetConfiguration *OPMapGadgetFactory::createConfiguration(QSettings* qSettings)
{
    return new OPMapGadgetConfiguration(QString("OPMapGadget"), qSettings);
}

IOptionsPage * OPMapGadgetFactory::createOptionsPage(IUAVGadgetConfiguration *config)
{
    return new OPMapGadgetOptionsPage(qobject_cast<OPMapGadgetConfiguration*>(config));
}

