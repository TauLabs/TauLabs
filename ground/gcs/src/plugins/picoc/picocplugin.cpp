/**
 ******************************************************************************
 * @file       picocplugin.cpp
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
#include "picocplugin.h"
#include "picocgadgetfactory.h"
#include <QtPlugin>
#include <QStringList>


PicoCPlugin::PicoCPlugin()
{
   // Do nothing
}

PicoCPlugin::~PicoCPlugin()
{
   // Do nothing
}

bool PicoCPlugin::initialize(const QStringList& args, QString *errMsg)
{
    Q_UNUSED(args);
    Q_UNUSED(errMsg);
    mf = new PicoCGadgetFactory(this);
    addAutoReleasedObject(mf);

    return true;
}

void PicoCPlugin::extensionsInitialized()
{
   // Do nothing
}

void PicoCPlugin::shutdown()
{
   // Do nothing
}

/**
  * @}
  * @}
  */
