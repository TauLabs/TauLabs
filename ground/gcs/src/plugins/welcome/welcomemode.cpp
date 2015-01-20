/**
 ******************************************************************************
 *
 * @file       welcomemode.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 *             Parts by Nokia Corporation (qt-info@nokia.com) Copyright (C) 2009.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup WelcomePlugin Welcome Plugin
 * @{
 * @brief The GCS Welcome plugin
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


#include "welcomemode.h"
#include <extensionsystem/pluginmanager.h>

#include <coreplugin/icore.h>
#include <coreplugin/coreconstants.h>
#include <coreplugin/uniqueidmanager.h>
#include <coreplugin/modemanager.h>

#include <utils/styledbar.h>
#include <utils/welcomemodetreewidget.h>
#include <utils/iwelcomepage.h>

#include <QDesktopServices>

#include <QtCore/QSettings>
#include <QtCore/QUrl>
#include <QtCore/QDebug>

#include <QtQuick>
#include <QtQuickWidgets/QQuickWidget>
#include <QQmlEngine>
#include <QQmlContext>

#include <cstdlib>

using namespace ExtensionSystem;
using namespace Utils;

namespace Welcome {

// ---  WelcomeMode
WelcomeMode::WelcomeMode() :
    m_widget(NULL),
    m_priority(Core::Constants::P_MODE_WELCOME)
{
}

WelcomeMode::~WelcomeMode()
{
}

QString WelcomeMode::name() const
{
    return tr("Welcome");
}

QIcon WelcomeMode::icon() const
{
    return QIcon(QLatin1String(":/core/images/openpiloticon.png"));
}

int WelcomeMode::priority() const
{
    return m_priority;
}

QWidget* WelcomeMode::widget()
{
    if (!m_widget) {
        QQuickWidget *qWidget = new QQuickWidget(QUrl("qrc:/welcome/qml/main.qml"));
        qWidget->setResizeMode(QQuickWidget::SizeRootObjectToView);
        qWidget->engine()->rootContext()->setContextProperty("welcomPlugin", this);
        m_widget = qWidget;
    }
    return m_widget;
}

const char* WelcomeMode::uniqueModeName() const
{
    return Core::Constants::MODE_WELCOME;
}

QList<int> WelcomeMode::context() const
{
    static QList<int> contexts = QList<int>()
                                 << Core::UniqueIDManager::instance()->uniqueIdentifier(Core::Constants::C_WELCOME_MODE);
    return contexts;
}

void WelcomeMode::openUrl(const QString &url)
{
    QDesktopServices::openUrl(QUrl(url));
}

void WelcomeMode::openPage(const QString &page)
{
    Core::ModeManager::instance()->activateModeByWorkspaceName(page);
}

void WelcomeMode::triggerAction(const QString &actionId)
{
    Core::ModeManager::instance()->triggerAction(actionId);
}

} // namespace Welcome
