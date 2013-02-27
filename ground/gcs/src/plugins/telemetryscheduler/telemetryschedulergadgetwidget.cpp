/**
 ******************************************************************************
 * @file       telemetryschedulergadgetwidget.cpp
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
#include "telemetryschedulergadgetwidget.h"
#include "ui_telemetryscheduler.h"

#include <QDebug>
#include <QString>
#include <QStringList>
#include <QtGui/QWidget>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>
#include <QtGui/QPushButton>

#include "extensionsystem/pluginmanager.h"

TelemetrySchedulerGadgetWidget::TelemetrySchedulerGadgetWidget(QWidget *parent) : QLabel(parent)
{
    m_telemetryeditor = new Ui_TelemetryScheduler();
    m_telemetryeditor->setupUi(this);

    waypointTable = new TelemetryTable(this);
    m_telemetryeditor->waypoints->setModel(waypointTable);

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm != NULL);
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objManager != NULL);
    waypointObj = Waypoint::GetInstance(objManager);
    Q_ASSERT(waypointObj != NULL);

    // Connect the signals
    connect(m_telemetryeditor->buttonNewWaypoint, SIGNAL(clicked()),
            this, SLOT(addInstance()));
}

TelemetrySchedulerGadgetWidget::~TelemetrySchedulerGadgetWidget()
{
   // Do nothing
}

void TelemetrySchedulerGadgetWidget::waypointChanged(UAVObject *)
{
}

void TelemetrySchedulerGadgetWidget::waypointActiveChanged(UAVObject *)
{
}

void TelemetrySchedulerGadgetWidget::addInstance()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm != NULL);
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objManager != NULL);

    qDebug() << "Instances before: " << objManager->getNumInstances(waypointObj->getObjID());
    Waypoint *obj = new Waypoint();
    quint32 newInstId = objManager->getNumInstances(waypointObj->getObjID());
    obj->initialize(newInstId,obj->getMetaObject());
    objManager->registerObject(obj);
    qDebug() << "Instances after: " << objManager->getNumInstances(waypointObj->getObjID());
}

/**
  * @}
  * @}
  */
