/**
 ******************************************************************************
 *
 * @file       vbrainplugin.h
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2013
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_Vbrain Vbrain boards support Plugin
 * @{
 * @brief Plugin to support boards by Quantec
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

#include "vbrainplugin.h"
#include "vbrain.h"
#include <QtPlugin>


VbrainPlugin::VbrainPlugin()
{
   // Do nothing
}

VbrainPlugin::~VbrainPlugin()
{
   // Do nothing
}

bool VbrainPlugin::initialize(const QStringList& args, QString *errMsg)
{
   Q_UNUSED(args);
   Q_UNUSED(errMsg);
   return true;
}

void VbrainPlugin::extensionsInitialized()
{
    /**
     * Create the board objects here.
     *
     */
    Vbrain* vbrain = new Vbrain();
    addAutoReleasedObject(vbrain);

}

void VbrainPlugin::shutdown()
{
}

Q_EXPORT_PLUGIN(VbrainPlugin)
