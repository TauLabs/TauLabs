/**
 ******************************************************************************
 *
 * @file       brainfpvplugin.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_BrainFPV BrainFPV boards support Plugin
 * @{
 * @brief Plugin to support boards by BrainFPV LLC
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
#ifndef BRAINFPVPLUGIN_H
#define BRAINFPVPLUGIN_H

#include <extensionsystem/iplugin.h>

class BrainFPVPlugin : public ExtensionSystem::IPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "TauLabs.plugins.TheBrainFPV" FILE "TheBrainFPV.json")

public:
   BrainFPVPlugin();
   ~BrainFPVPlugin();

   void extensionsInitialized();
   bool initialize(const QStringList & arguments, QString * errorString);
   void shutdown();

};

#endif // BRAINFPVPLUGIN_H
