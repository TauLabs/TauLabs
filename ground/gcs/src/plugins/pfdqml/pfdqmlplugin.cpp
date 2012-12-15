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

#include "pfdqmlplugin.h"
#include "pfdqmlgadgetfactory.h"
#include <QDebug>
#include <QtPlugin>
#include <QStringList>
#include <extensionsystem/pluginmanager.h>


PfdQmlPlugin::PfdQmlPlugin()
{
   // Do nothing
}

PfdQmlPlugin::~PfdQmlPlugin()
{
   // Do nothing
}

bool PfdQmlPlugin::initialize(const QStringList& args, QString *errMsg)
{
   Q_UNUSED(args);
   Q_UNUSED(errMsg);
   mf = new PfdQmlGadgetFactory(this);
   addAutoReleasedObject(mf);

   return true;
}

void PfdQmlPlugin::extensionsInitialized()
{
   // Do nothing
}

void PfdQmlPlugin::shutdown()
{
   // Do nothing
}
Q_EXPORT_PLUGIN(PfdQmlPlugin)

