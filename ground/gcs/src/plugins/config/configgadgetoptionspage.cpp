/**
 ******************************************************************************
 *
 * @file       configgadgetoptionspage.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief The Configuration Gadget used to update settings in the firmware
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
#include "configgadgetoptionspage.h"

#include <QDebug>
#include <QStringList>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>

ConfigGadgetOptionsPage::ConfigGadgetOptionsPage(ConfigGadgetConfiguration *config, QObject *parent)
{
//todo:implement
    Q_UNUSED(config);
    Q_UNUSED(parent);
}

QWidget *ConfigGadgetOptionsPage::createPage(QWidget *parent)
{
    //todo:implement
    Q_UNUSED(parent);
    return new QWidget;
}

void ConfigGadgetOptionsPage::apply()
{
    //todo:implement
}

void ConfigGadgetOptionsPage::finish(){
    //todo:implement
}
