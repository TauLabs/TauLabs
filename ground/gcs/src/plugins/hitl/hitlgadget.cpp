/**
 ******************************************************************************
 *
 * @file       hitlgadget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup HITLPlugin HITL Plugin
 * @{
 * @brief The Hardware In The Loop plugin 
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
#include "hitlgadget.h"
#include "hitlwidget.h"
#include "hitlconfiguration.h"
#include "simulator.h"

HITLGadget::HITLGadget(QString classId, HITLWidget *widget, QWidget *parent) :
        IUAVGadget(classId, parent),
        m_widget(widget)
{
	connect(this,SIGNAL(changeConfiguration(void)),m_widget,SLOT(stopButtonClicked(void)));
}

HITLGadget::~HITLGadget()
{
    delete m_widget;
}

void HITLGadget::loadConfiguration(IUAVGadgetConfiguration* config)
{
    HITLConfiguration *m = qobject_cast<HITLConfiguration*>(config);
    // IL2 <-- Is this still necessary? [KDS]
	emit changeConfiguration();
	m_widget->setSettingParameters(m->Settings());
}

