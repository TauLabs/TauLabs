/**
 ******************************************************************************
 *
 * @file       openpilotplugin.h
 * @author     The PhoenixPilot Team, http://github.com/PhoenixPilot Copyright (C) 2012.
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_OpenPilotPlugin OpenPilot boards support Plugin
 * @{
 * @brief Plugin to support boards by the OP project
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

#include "openpilotplugin.h"
#include "coptercontrol.h"
#include "pipxtreme.h"
#include "revolution.h"
#include "revomini.h"
#include <QtPlugin>


OpenPilotPlugin::OpenPilotPlugin()
{
   // Do nothing
}

OpenPilotPlugin::~OpenPilotPlugin()
{
   // Do nothing
}

bool OpenPilotPlugin::initialize(const QStringList& args, QString *errMsg)
{
   Q_UNUSED(args);
   Q_UNUSED(errMsg);
   return true;
}

void OpenPilotPlugin::extensionsInitialized()
{
    /**
     * Create the board objects here.
     *
     */
    CopterControl* cc3d = new CopterControl();
    addAutoReleasedObject(cc3d);

    PipXtreme* pipx = new PipXtreme();
    addAutoReleasedObject(pipx);

    Revolution* revo = new Revolution();
    addAutoReleasedObject(revo);

    RevoMini* rmini = new RevoMini();
    addAutoReleasedObject(rmini);

}

void OpenPilotPlugin::shutdown()
{
}

Q_EXPORT_PLUGIN(OpenPilotPlugin)
