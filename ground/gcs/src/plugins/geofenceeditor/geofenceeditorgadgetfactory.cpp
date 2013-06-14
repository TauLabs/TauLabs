/**
 ******************************************************************************
 * @file       geofenceeditorgadgetfactor.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup GeoFenceEditorGadgetPlugin Geo-fence Editor Gadget Plugin
 * @{
 * @brief A gadget to edit a geo-fence
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
#include "geofenceeditorgadgetfactory.h"
#include "geofenceeditorgadgetwidget.h"
#include "geofenceeditorgadget.h"
#include <coreplugin/iuavgadget.h>

GeoFenceEditorGadgetFactory::GeoFenceEditorGadgetFactory(QObject *parent) :
        IUAVGadgetFactory(QString("GeoFenceEditorGadget"),
                          tr("Geo-fence Editor"),
                          parent)
{
}

GeoFenceEditorGadgetFactory::~GeoFenceEditorGadgetFactory()
{

}

IUAVGadget* GeoFenceEditorGadgetFactory::createGadget(QWidget *parent) {
    GeoFenceEditorGadgetWidget* gadgetWidget = new GeoFenceEditorGadgetWidget(parent);
    return new GeoFenceEditorGadget(QString("GeoFenceEditorGadget"), gadgetWidget, parent);
}
