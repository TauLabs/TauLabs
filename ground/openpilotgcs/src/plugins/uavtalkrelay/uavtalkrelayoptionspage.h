/**
 ******************************************************************************
 * @file       uavtalkrelayoptionspage.h
 * @author     The PhoenixPilot Team, http://github.com/PhoenixPilot
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVTalk relay plugin
 * @{
 *
 * @brief Relays UAVTalk data trough UDP to another GCS
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

#ifndef UAVTALKRELAYOPTIONSPAGE_H
#define UAVTALKRELAYOPTIONSPAGE_H

#include "coreplugin/dialogs/ioptionspage.h"
#include <coreplugin/iconfigurableplugin.h>
#include <QHash>

class UavTalkRelayPlugin;
namespace Core {
    class IUAVGadgetConfiguration;
}

namespace Ui {
    class UavTalkRelayOptionsPage;
}

using namespace Core;

class UavTalkRelayOptionsPage : public IOptionsPage
{
Q_OBJECT
public:
    UavTalkRelayOptionsPage(QObject *parent = 0);
    virtual ~UavTalkRelayOptionsPage();

    QString id() const { return QLatin1String("settings"); }
    QString trName() const { return tr("settings"); }
    QString category() const { return "UAV Talk Relay";}
    QString trCategory() const { return "UAV Talk Relay"; }

    QWidget *createPage(QWidget *parent);
    void apply();
    void finish();
signals:
    void settingsUpdated();
private slots:
    void addRule();
    void deleteRule();
private:
    Ui::UavTalkRelayOptionsPage *m_page;
    UavTalkRelayPlugin * m_config;
};

#endif // UAVTALKRELAYOPTIONSPAGE_H
