/**
 ******************************************************************************
 *
 * @file       configstabilizationwidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Stabilization configuration panel
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
#ifndef CONFIGSTABILIZATIONWIDGET_H
#define CONFIGSTABILIZATIONWIDGET_H

#include "ui_stabilization.h"
#include "../uavobjectwidgetutils/configtaskwidget.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include "stabilizationsettings.h"
#include <QWidget>
#include <QTimer>


class ConfigStabilizationWidget: public ConfigTaskWidget
{
    Q_OBJECT

public:
    ConfigStabilizationWidget(QWidget *parent = 0);
    ~ConfigStabilizationWidget();

private:
    Ui_StabilizationWidget *m_stabilization;
    QTimer * realtimeUpdates;
private slots:
    void realtimeUpdatesSlot(int);
    void linkCheckBoxes(int value);
    void processLinkedWidgets(QWidget*);
    void applyRateLimits();
};

#endif // ConfigStabilizationWidget_H
