/**
 ******************************************************************************
 *
 * @file       scopegadgetconfiguration.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ScopePlugin Scope Gadget Plugin
 * @{
 * @brief The scope Gadget, graphically plots the states of UAVObjects
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

#ifndef SCOPEGADGETCONFIGURATION_H
#define SCOPEGADGETCONFIGURATION_H

#include "scopesconfig.h"
#include <coreplugin/iuavgadgetconfiguration.h>
#include "ui_scopegadgetoptionspage.h"

#include <QVector>

#include "qwt/src/qwt_color_map.h"

using namespace Core;


class ScopeGadgetConfiguration : public IUAVGadgetConfiguration
{
    Q_OBJECT
public:
    explicit ScopeGadgetConfiguration(QString classId, QSettings* qSettings = 0, QObject *parent = 0);
    ~ScopeGadgetConfiguration();

    //configurations getter functions
    ScopeConfig* getScope(){return m_scope;}

    void saveConfig(QSettings* settings) const; //THIS SEEMS TO BE UNUSED
    IUAVGadgetConfiguration* clone();

    void applyGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page);

private:
    ScopeConfig *m_scope;

};

#endif // SCOPEGADGETCONFIGURATION_H
