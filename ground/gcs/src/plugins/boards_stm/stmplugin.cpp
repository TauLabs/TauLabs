/**
 ******************************************************************************
 *
 * @file       stmplugin.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_Stm STM boards support Plugin
 * @{
 * @brief Plugin to support boards by STM
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

#include "stmplugin.h"
#include "flyingf3.h"
#include "flyingf4.h"
#include "discoveryf4.h"
#include <QtPlugin>


StmPlugin::StmPlugin()
{
   // Do nothing
}

StmPlugin::~StmPlugin()
{
   // Do nothing
}

bool StmPlugin::initialize(const QStringList& args, QString *errMsg)
{
   Q_UNUSED(args);
   Q_UNUSED(errMsg);
   return true;
}

void StmPlugin::extensionsInitialized()
{
    /**
     * Create the board objects here.
     *
     */
    FlyingF3* flyingf3 = new FlyingF3();
    addAutoReleasedObject(flyingf3);

    FlyingF4* flyingf4 = new FlyingF4();
    addAutoReleasedObject(flyingf4);

    DiscoveryF4* discoveryf4 = new DiscoveryF4();
    addAutoReleasedObject(discoveryf4);
}

void StmPlugin::shutdown()
{
}

