/**
 ******************************************************************************
 * @file       taulabsplugin.cpp
 * @author     Tau Labs, http://github.com/TauLabs, Copyright (C) 2013.
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_TauLabsPlugin Tau Labs boards support Plugin
 * @{
 * @brief Plugin to support boards by the Tau Labs project
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

#include "taulabsplugin.h"
#include "freedom.h"
#include "sparky.h"
#include <QtPlugin>


TauLabsPlugin::TauLabsPlugin()
{
   // Do nothing
}

TauLabsPlugin::~TauLabsPlugin()
{
   // Do nothing
}

bool TauLabsPlugin::initialize(const QStringList& args, QString *errMsg)
{
   Q_UNUSED(args);
   Q_UNUSED(errMsg);
   return true;
}

void TauLabsPlugin::extensionsInitialized()
{
    /**
     * Create the board objects here.
     *
     */
    Sparky* sparky = new Sparky();
    addAutoReleasedObject(sparky);

    Freedom* freedom = new Freedom();
    addAutoReleasedObject(freedom);
}

void TauLabsPlugin::shutdown()
{
}

Q_EXPORT_PLUGIN(TauLabsPlugin)
