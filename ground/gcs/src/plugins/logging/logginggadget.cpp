/**
 ******************************************************************************
 *
 * @file       GCSControlgadget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup GCSControlGadgetPlugin GCSControl Gadget Plugin
 * @{
 * @brief A gadget to control the UAV, either from the keyboard or a joystick
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
#include "logginggadget.h"
#include "logginggadgetwidget.h"

#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"

LoggingGadget::LoggingGadget(QString classId, LoggingGadgetWidget *widget, QWidget *parent) :
        IUAVGadget(classId, parent),
        m_widget(widget)
{
}

LoggingGadget::~LoggingGadget()
{
    delete m_widget;
}

void LoggingGadget::loadConfiguration(IUAVGadgetConfiguration* config)
{
    Q_UNUSED(config);
}
