/**
 ******************************************************************************
 *
 * @file       telemetryschedulergadgetconfiguration.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup TelemetryScheduler Telemetry Scheduler Plugin
 * @{
 * @brief The telemetry scheduler gadget, allows the user to set telemetry
 * rates for different physical link rates
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

#include "telemetryschedulergadgetconfiguration.h"

TelemetrySchedulerConfiguration::TelemetrySchedulerConfiguration(QString classId, QSettings* qSettings, QObject *parent) :
        IUAVGadgetConfiguration(classId, parent)
{
    Q_UNUSED(qSettings);
}


/**
 * Clones a configuration.
 *
 */
IUAVGadgetConfiguration *TelemetrySchedulerConfiguration::clone()
{
    return NULL;
}


/**
 * @brief TelemetrySchedulerConfiguration::saveConfig Saves the configuration to the GCS XML file
 * @param qSettings
 */
void TelemetrySchedulerConfiguration::saveConfig(QSettings* qSettings) const
{
    Q_UNUSED(qSettings);
}

TelemetrySchedulerConfiguration::~TelemetrySchedulerConfiguration()
{
}
