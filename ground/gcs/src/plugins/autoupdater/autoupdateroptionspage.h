/**
 ******************************************************************************
 * @file       autoupdateroptionspage.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup AutoUpdater plugin
 * @{
 *
 * @brief Auto updates the GCS from GitHub releases
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

#ifndef AUTOUPDATEROPTIONSPAGE_H
#define AUTOUPDATEROPTIONSPAGE_H

#include "coreplugin/dialogs/ioptionspage.h"
#include <coreplugin/iconfigurableplugin.h>
#include <QHash>

class AutoUpdaterPlugin;
namespace Core {
    class IUAVGadgetConfiguration;
}

namespace Ui {
    class AutoUpdaterOptionsPage;
}

using namespace Core;

class AutoUpdaterOptionsPage : public IOptionsPage
{
Q_OBJECT
public:
    AutoUpdaterOptionsPage(QObject *parent = 0);
    virtual ~AutoUpdaterOptionsPage();

    QString id() const { return QLatin1String("settings"); }
    QString trName() const { return tr("settings"); }
    QString category() const { return "Auto Updater";}
    QString trCategory() const { return "Auto Updater"; }

    QWidget *createPage(QWidget *parent);
    void apply();
    void finish();
signals:
    void settingsUpdated();
private slots:

private:
    Ui::AutoUpdaterOptionsPage *m_page;
    AutoUpdaterPlugin * m_config;
};

#endif // UAVTALKRELAYOPTIONSPAGE_H
