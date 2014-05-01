/**
 ******************************************************************************
 *
 * @file       scopegadget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ScopePlugin Scope Gadget Plugin
 * @{
 * @brief The scope Gadget, graphically plots the states of UAVObjects
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

#include "scopeplugin.h"
#include "scopegadget.h"
#include "scopegadgetconfiguration.h"
#include "scopegadgetwidget.h"

#include <qcolor.h>

ScopeGadget::ScopeGadget(QString classId, ScopeGadgetWidget *widget, QWidget *parent) :
        IUAVGadget(classId, parent),
        scopeGadgetWidget(widget),
        configLoaded(false)
{

}


/**
 * @brief ScopeGadget::loadConfiguration Loads the plugin configuration
 * @param config
 */
void ScopeGadget::loadConfiguration(IUAVGadgetConfiguration* config)
{
    ScopeGadgetConfiguration *sgConfig = qobject_cast<ScopeGadgetConfiguration*>(config);
    if (sgConfig == NULL) //Check that the case succeeded.
        return;

    scopeGadgetWidget->clearPlotWidget();
    sgConfig->getScope()->loadConfiguration(scopeGadgetWidget);
}


/**
 * @brief ScopeGadget::~ScopeGadget   Scope gadget destructor: deletes the
 * associated scope gadget widget too.
 */
ScopeGadget::~ScopeGadget()
{
   delete scopeGadgetWidget;
}
