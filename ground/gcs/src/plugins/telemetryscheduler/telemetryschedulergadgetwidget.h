/**
 ******************************************************************************
 * @file       telemetryschedulergadgetwidget.h
 * @author     Tau Labs, http://www.taulabls.org Copyright (C) 2013.
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

#ifndef TELEMETRYSCHEDULERGADGETWIDGET_H_
#define TELEMETRYSCHEDULERGADGETWIDGET_H_

#include <QtGui/QLabel>
#include <telemetrytable.h>
#include <waypointactive.h>

class Ui_TelemetryScheduler;

class TelemetrySchedulerGadgetWidget : public QLabel
{
    Q_OBJECT

public:
    TelemetrySchedulerGadgetWidget(QWidget *parent = 0);
    ~TelemetrySchedulerGadgetWidget();

signals:

protected slots:
    void waypointChanged(UAVObject *);
    void waypointActiveChanged(UAVObject *);
    void addInstance();

private:
    Ui_TelemetryScheduler * m_telemetryeditor;
    TelemetryTable *waypointTable;
    Waypoint *waypointObj;
};

#endif /* TELEMETRYSCHEDULERGADGETWIDGET_H_ */
