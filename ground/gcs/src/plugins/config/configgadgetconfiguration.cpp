/**
 ******************************************************************************
 *
 * @file       configgadgetconfiguration.cpp
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
#include "configgadgetconfiguration.h"

ConfigGadgetConfiguration::ConfigGadgetConfiguration(QString classId, QSettings* qSettings, QObject *parent) :
    IUAVGadgetConfiguration(classId, parent)
{
    Q_UNUSED(qSettings);

}

IUAVGadgetConfiguration *ConfigGadgetConfiguration::clone()
{
    ConfigGadgetConfiguration *m = new ConfigGadgetConfiguration(this->classId());
    return m;
}

void ConfigGadgetConfiguration::saveConfig(QSettings* settings) const {

    Q_UNUSED(settings)
}
