/**
 ******************************************************************************
 * @file       telemetryschedulerplugin.cpp
 * @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
 * @addtogroup Telemetry Scheduler GCS Plugins
 * @{
 * @addtogroup TelemetrySchedulerGadgetPlugin Telemetry Scheduler Gadget Plugin
 * @{
 * @brief A gadget to edit the telemetry scheduling list
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
#include "telemetryschedulerplugin.h"
#include "telemetryschedulergadgetfactory.h"
#include <QDebug>
#include <QtPlugin>
#include <QStringList>
#include <extensionsystem/pluginmanager.h>


TelemetrySchedulerPlugin::TelemetrySchedulerPlugin()
{
   // Do nothing
}

TelemetrySchedulerPlugin::~TelemetrySchedulerPlugin()
{
   // Do nothing
}

bool TelemetrySchedulerPlugin::initialize(const QStringList& args, QString *errMsg)
{
   Q_UNUSED(args);
   Q_UNUSED(errMsg);
   mf = new TelemetrySchedulerGadgetFactory(this);
   addAutoReleasedObject(mf);

   return true;
}

void TelemetrySchedulerPlugin::extensionsInitialized()
{
   // Do nothing
}

void TelemetrySchedulerPlugin::shutdown()
{
   // Do nothing
}

/**
  * @}
  * @}
  */
