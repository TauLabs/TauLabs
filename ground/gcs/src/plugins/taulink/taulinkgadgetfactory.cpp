/**
 ******************************************************************************
 * @file       taulinkgadgetfactory.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup TauLinkGadgetPlugin Tau Link Gadget Plugin
 * @{
 * @brief A gadget to monitor and configure the RFM22b link
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
#include "taulinkgadgetfactory.h"
#include "taulinkgadgetwidget.h"
#include "taulinkgadget.h"
#include <coreplugin/iuavgadget.h>

TauLinkGadgetFactory::TauLinkGadgetFactory(QObject *parent) :
        IUAVGadgetFactory(QString("TauLinkGadget"),
                          tr("Tau Link"),
                          parent)
{
    tauLinkPlugin = (TauLinkPlugin *) parent;
}

TauLinkGadgetFactory::~TauLinkGadgetFactory()
{

}

IUAVGadget* TauLinkGadgetFactory::createGadget(QWidget *parent) {
    TauLinkGadgetWidget *gadgetWidget = new TauLinkGadgetWidget(parent);
    return new TauLinkGadget(QString("TauLinkGadget"), gadgetWidget, parent);
}
