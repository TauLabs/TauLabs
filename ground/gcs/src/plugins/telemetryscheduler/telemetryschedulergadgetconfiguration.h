/**
 ******************************************************************************
 *
 * @file       telemetryschedulergadgetconfiguration.h
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

#ifndef TELEMETRYSCHEDULERCONFIGURATION_H
#define TELEMETRYSCHEDULERCONFIGURATION_H

#include <coreplugin/iuavgadgetconfiguration.h>

#include <QVector>

using namespace Core;

class TelemetrySchedulerConfiguration : public IUAVGadgetConfiguration
{
    Q_OBJECT
public:
    explicit TelemetrySchedulerConfiguration(QString classId, QSettings* qSettings = 0, QObject *parent = 0);

    ~TelemetrySchedulerConfiguration();
    IUAVGadgetConfiguration *clone();
    void saveConfig(QSettings* settings) const; //THIS SEEMS TO BE UNUSED


private:
};

#endif // TELEMETRYSCHEDULERCONFIGURATION_H
