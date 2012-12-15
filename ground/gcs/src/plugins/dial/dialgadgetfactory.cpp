/**
 ******************************************************************************
 *
 * @file       dialgadgetfactory.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @see        The GNU Public License (GPL) Version 3
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup DialPlugin Dial Plugin
 * @{
 * @brief Plots flight information rotary style dials 
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
#include "dialgadgetfactory.h"
#include "dialgadgetwidget.h"
#include "dialgadget.h"
#include "dialgadgetconfiguration.h"
#include "dialgadgetoptionspage.h"
#include <coreplugin/iuavgadget.h>

DialGadgetFactory::DialGadgetFactory(QObject *parent) :
        IUAVGadgetFactory(QString("DialGadget"),
                          tr("Analog Dial"),
                          parent)
{
}

DialGadgetFactory::~DialGadgetFactory()
{
}

Core::IUAVGadget* DialGadgetFactory::createGadget(QWidget *parent)
{
    DialGadgetWidget* gadgetWidget = new DialGadgetWidget(parent);
    return new DialGadget(QString("DialGadget"), gadgetWidget, parent);
}

IUAVGadgetConfiguration *DialGadgetFactory::createConfiguration(QSettings* qSettings)
{
    return new DialGadgetConfiguration(QString("DialGadget"), qSettings);
}

IOptionsPage *DialGadgetFactory::createOptionsPage(IUAVGadgetConfiguration *config)
{
    return new DialGadgetOptionsPage(qobject_cast<DialGadgetConfiguration*>(config));
}

