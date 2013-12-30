/**
 ******************************************************************************
 *
 * @file       uavobjectbrowserconfiguration.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVObjectBrowserPlugin UAVObject Browser Plugin
 * @{
 * @brief The UAVObject Browser gadget plugin
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

#include "uavobjectbrowserconfiguration.h"

UAVObjectBrowserConfiguration::UAVObjectBrowserConfiguration(QString classId, QSettings* qSettings, QObject *parent) :
    IUAVGadgetConfiguration(classId, parent),
    m_recentlyUpdatedColor(QColor(255, 230, 230)),
    m_manuallyChangedColor(QColor(230, 230, 255)),
    m_notPresentOnHwColor(QColor(170,170,170)),
    m_recentlyUpdatedTimeout(500),
    m_onlyHighlightChangedValues(false),
    m_useCategorizedView(false),
    m_useScientificView(false),
    m_showMetaData(false),
    m_hideNotPresentOnHw(false)
{
    //if a saved configuration exists load it
    if(qSettings != 0) {
        QColor recent = qSettings->value("recentlyUpdatedColor").value<QColor>();
        QColor manual = qSettings->value("manuallyChangedColor").value<QColor>();
        QColor present = qSettings->value("notPresentOnHwColor", QColor(170,170,170)).value<QColor>();
        int timeout = qSettings->value("recentlyUpdatedTimeout").toInt();
        bool highlight = qSettings->value("onlyHighlightChangedValues").toBool();

        m_useCategorizedView = qSettings->value("CategorizedView").toBool();
        m_useScientificView = qSettings->value("ScientificView").toBool();
        m_showMetaData = qSettings->value("showMetaData").toBool();
        m_hideNotPresentOnHw = qSettings->value("hideNotPresentOnHw",false).toBool();
        m_recentlyUpdatedColor = recent;
        m_manuallyChangedColor = manual;
        m_notPresentOnHwColor = present;
        m_recentlyUpdatedTimeout = timeout;
        m_onlyHighlightChangedValues = highlight;
    }
}

IUAVGadgetConfiguration *UAVObjectBrowserConfiguration::clone()
{
    UAVObjectBrowserConfiguration *m = new UAVObjectBrowserConfiguration(this->classId());
    m->m_recentlyUpdatedColor = m_recentlyUpdatedColor;
    m->m_manuallyChangedColor = m_manuallyChangedColor;
    m->m_notPresentOnHwColor = m_notPresentOnHwColor;
    m->m_recentlyUpdatedTimeout = m_recentlyUpdatedTimeout;
    m->m_onlyHighlightChangedValues = m_onlyHighlightChangedValues;
    m->m_useCategorizedView = m_useCategorizedView;
    m->m_useScientificView = m_useScientificView;
    m->m_showMetaData = m_showMetaData;
    m->m_hideNotPresentOnHw = m_hideNotPresentOnHw;
    return m;
}

/**
 * Saves a configuration.
 *
 */
void UAVObjectBrowserConfiguration::saveConfig(QSettings* qSettings) const {
    qSettings->setValue("recentlyUpdatedColor", m_recentlyUpdatedColor);
    qSettings->setValue("manuallyChangedColor", m_manuallyChangedColor);
    qSettings->setValue("notPresentOnHwColor", m_notPresentOnHwColor);
    qSettings->setValue("recentlyUpdatedTimeout", m_recentlyUpdatedTimeout);
    qSettings->setValue("onlyHighlightChangedValues", m_onlyHighlightChangedValues);
    qSettings->setValue("CategorizedView", m_useCategorizedView);
    qSettings->setValue("ScientificView", m_useScientificView);
    qSettings->setValue("showMetaData", m_showMetaData);
    qSettings->setValue("hideNotPresentOnHw", m_hideNotPresentOnHw);
}
