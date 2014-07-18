/**
 ******************************************************************************
 *
 * @file       mocapplugin.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup MoCapPlugin Motion Capture Plugin
 * @{
 * @brief Motion capture plugin which communicates via UDP
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

#ifndef MOCAPPLUGIN_H
#define MOCAPPLUGIN_H

#include <extensionsystem/iplugin.h>
#include <QStringList>

#include "exporter.h"

class MoCapFactory;

class MoCapPlugin : public ExtensionSystem::IPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "AboveGroundLabs.plugins.motioncapture" FILE "motioncapture.json")
public:
    MoCapPlugin();
   ~MoCapPlugin();

   void extensionsInitialized();
   bool initialize(const QStringList & arguments, QString * errorString);
   void shutdown();


   static void addMocap(MocapCreator* creator)
   {
      MoCapPlugin::typeMocaps.append(creator);
   }

   static MocapCreator* getMocapCreator(const QString classId)
   {
       foreach(MocapCreator* creator, MoCapPlugin::typeMocaps)
	   {
		   if(classId == creator->ClassId())
			   return creator;
	   }
	   return 0;
   }

   static QList<MocapCreator* > typeMocaps;

private:
   MoCapFactory *mf;


};
#endif /* MOCAPPLUGIN_H */
