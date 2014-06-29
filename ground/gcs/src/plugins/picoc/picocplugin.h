/**
 ******************************************************************************
 * @file       picocplugin.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup PicoCGadgetPlugin PicoC Editor Gadget Plugin
 * @{
 * @brief A gadget to edit PicoC scripts
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

#ifndef PICOCPLUGIN_H_
#define PICOCPLUGIN_H_

#include <extensionsystem/iplugin.h>
#include "picoc_global.h"

class PicoCGadgetFactory;

class PICOC_EXPORT PicoCPlugin : public ExtensionSystem::IPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "TauLabs.plugins.PicoC" FILE "PicoC.json")

public:
    PicoCPlugin();
   ~PicoCPlugin();

   void extensionsInitialized();
   bool initialize(const QStringList & arguments, QString * errorString);
   void shutdown();
private:
   PicoCGadgetFactory *mf;
};

#endif /* PICOCPLUGIN_H_ */
