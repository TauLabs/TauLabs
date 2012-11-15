/**
 ******************************************************************************
 *
 * @file       mocapfactory.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup MoCapPlugin Motion Capture Plugin
 * @{
 * @brief The Hardware In The Loop plugin 
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
#include "mocapfactory.h"
#include "mocapwidget.h"
#include "mocapgadget.h"
#include "mocapconfiguration.h"
#include "mocapoptionspage.h"
#include <coreplugin/iuavgadget.h>

MoCapFactory::MoCapFactory(QObject *parent)
    : IUAVGadgetFactory(QString("MoCap"), tr("Motion Capture"), parent)
{

}

MoCapFactory::~MoCapFactory()
{
}

Core::IUAVGadget* MoCapFactory::createGadget(QWidget *parent)
{


   MoCapWidget* gadgetWidget = new MoCapWidget(parent);
   return new MoCapGadget(QString("MoCap"), gadgetWidget, parent);
}

IUAVGadgetConfiguration *MoCapFactory::createConfiguration(QSettings* qSettings)
{
    return new MoCapConfiguration(QString("MoCap"), qSettings);
}

IOptionsPage *MoCapFactory::createOptionsPage(IUAVGadgetConfiguration *config)
{
    return new MoCapOptionsPage(qobject_cast<MoCapConfiguration*>(config));
}

