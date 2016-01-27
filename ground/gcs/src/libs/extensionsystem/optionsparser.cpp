/**
 ******************************************************************************
 *
 * @file       optionsparser.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 *             Parts by Nokia Corporation (qt-info@nokia.com) Copyright (C) 2009.
 * @brief      
 * @see        The GNU Public License (GPL) Version 3
 * @defgroup   
 * @{
 * 
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

#include "optionsparser.h"

#include <QtCore/QCoreApplication>

using namespace ExtensionSystem;
using namespace ExtensionSystem::Internal;

OptionsParser::OptionsParser(PluginManagerPrivate *pmPrivate):m_pmPrivate(pmPrivate)
{
    m_pmPrivate->arguments.clear();
}

QStringList OptionsParser::parse(QStringList pluginOptions, QStringList pluginTests, QStringList pluginNoLoad)
{
    checkForTestOption(pluginTests);
    checkForNoLoadOption(pluginNoLoad);
    checkForPluginOption(pluginOptions);
    return m_errorStrings;
}

void OptionsParser::checkForTestOption(QStringList pluginTests)
{
    foreach (QString plugin, pluginTests) {
        PluginSpec *spec = m_pmPrivate->pluginByName(plugin);
        if (!spec) {
            m_errorStrings.append(QCoreApplication::translate("PluginManager",
                                                              "Invalid test option, the plugin '%0' does not exist.").arg(plugin));
        } else {
            m_pmPrivate->testSpecs.append(spec);
        }
    }
    return;
}

void OptionsParser::checkForNoLoadOption(QStringList pluginNoLoad)
{
    foreach (QString plugin, pluginNoLoad) {
        PluginSpec *spec = m_pmPrivate->pluginByName(plugin);
        if (!spec) {
            m_errorStrings.append(QCoreApplication::translate("PluginManager",
                                                              "Invalid no-load option, the plugin '%0' does not exist.").arg(plugin));
        } else {
            m_pmPrivate->pluginSpecs.removeAll(spec);
            delete spec;
            m_pmPrivate->resolveDependencies();
        }
    }
    return;
}

void OptionsParser::checkForPluginOption(QStringList pluginOptions)
{
    foreach (QString option, pluginOptions) {
        QString simplified = option.simplified().replace(" ", "");
        if(!simplified.contains("=") || (simplified.split("=").length() != 2)) {
            m_errorStrings.append(QCoreApplication::translate(("PluginManager"),
                    "Wrong plugin options syntax: %0").arg(option));
            continue;
        }
        QString argument = simplified.split("=").at(0);
        QString value = simplified.split("=").at(1);
        bool requiresParameter;
        PluginSpec *spec = m_pmPrivate->pluginForOption(argument, &requiresParameter);
        if (!spec) {
            m_errorStrings.append(QCoreApplication::translate("PluginManager",
                    "No Plugin was found for given argument: %0").arg(argument));
            continue;
        }
        spec->addArgument(argument);
        spec->addArgument(value);
    }
    return;
}
