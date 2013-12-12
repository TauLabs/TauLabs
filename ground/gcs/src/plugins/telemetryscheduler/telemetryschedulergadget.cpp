/**
 ******************************************************************************
 * @file       telemetryschedulergadget.cpp
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
#include "telemetryschedulergadget.h"
#include "telemetryschedulergadgetconfiguration.h"
#include "telemetryschedulergadgetwidget.h"

#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include <QDebug>

TelemetrySchedulerGadget::TelemetrySchedulerGadget(QString classId, TelemetrySchedulerGadgetWidget *widget, QWidget *parent) :
        IUAVGadget(classId, parent),
        m_widget(widget)
{
}

TelemetrySchedulerGadget::~TelemetrySchedulerGadget()
{
    delete m_widget;
}

void TelemetrySchedulerGadget::loadConfiguration(IUAVGadgetConfiguration* config)
{

    TelemetrySchedulerConfiguration *tsgConfiguration = qobject_cast<TelemetrySchedulerConfiguration*>(config);
    TelemetrySchedulerGadgetWidget* widget = qobject_cast<TelemetrySchedulerGadgetWidget*>(m_widget);

    widget->setConfig(tsgConfiguration);

}
