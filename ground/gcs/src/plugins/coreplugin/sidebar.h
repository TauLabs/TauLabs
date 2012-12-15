/**
 ******************************************************************************
 *
 * @file       sidebar.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 *             Parts by Nokia Corporation (qt-info@nokia.com) Copyright (C) 2009.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup CorePlugin Core Plugin
 * @{
 * @brief The Core GCS plugin
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

#ifndef SIDEBAR_H
#define SIDEBAR_H

#include <QtCore/QMap>
#include <QtCore/QPointer>
#include <QtGui/QWidget>
#include <QtGui/QComboBox>

#include <coreplugin/minisplitter.h>

QT_BEGIN_NAMESPACE
class QSettings;
class QToolBar;
class QAction;
class QToolButton;
QT_END_NAMESPACE

namespace Core {

class Command;

namespace Internal {
class SideBarWidget;
class ComboBox;
} // namespace Internal

/*
 * An item in the sidebar. Has a widget that is displayed in the sidebar and
 * optionally a list of tool buttons that are added to the toolbar above it.
 * The window title of the widget is displayed in the combo box.
 *
 * The SideBarItem takes ownership over the widget.
 */
class CORE_EXPORT SideBarItem
{
public:
    SideBarItem(QWidget *widget)
        : m_widget(widget)
    {}

    virtual ~SideBarItem();

    QWidget *widget() { return m_widget; }

    /* Should always return a new set of tool buttons.
     *
     * Workaround since there doesn't seem to be a nice way to remove widgets
     * that have been added to a QToolBar without either not deleting the
     * associated QAction or causing the QToolButton to be deleted.
     */
    virtual QList<QToolButton *> createToolBarWidgets()
    {
        return QList<QToolButton *>();
    }

private:
    QWidget *m_widget;
};

class CORE_EXPORT SideBar : public MiniSplitter
{
    Q_OBJECT
public:
    /*
     * The SideBar takes ownership of the SideBarItems.
     */
    SideBar(QList<SideBarItem*> widgetList,
            QList<SideBarItem*> defaultVisible);
    ~SideBar();

    QStringList availableItems() const;
    void makeItemAvailable(SideBarItem *item);
    SideBarItem *item(const QString &title);

    void saveSettings(QSettings *settings);
    void readSettings(QSettings *settings);

    void activateItem(SideBarItem *item);

    void setShortcutMap(const QMap<QString, Core::Command*> &shortcutMap);
    QMap<QString, Core::Command*> shortcutMap() const;

private slots:
    void splitSubWidget();
    void closeSubWidget();
    void updateWidgets();

private:
    Internal::SideBarWidget *insertSideBarWidget(int position,
                                                 const QString &title = QString());
    void removeSideBarWidget(Internal::SideBarWidget *widget);

    QList<Internal::SideBarWidget*> m_widgets;

    QMap<QString, SideBarItem*> m_itemMap;
    QStringList m_availableItems;
    QStringList m_defaultVisible;
    QMap<QString, Core::Command*> m_shortcutMap;
};

namespace Internal {

class SideBarWidget : public QWidget
{
    Q_OBJECT
public:
    SideBarWidget(SideBar *sideBar, const QString &title);
    ~SideBarWidget();

    QString currentItemTitle() const;
    void setCurrentItem(const QString &title);

    void updateAvailableItems();
    void removeCurrentItem();

    Core::Command *command(const QString &title) const;

signals:
    void splitMe();
    void closeMe();
    void currentWidgetChanged();

private slots:
    void setCurrentIndex(int);

private:
    ComboBox *m_comboBox;
    SideBarItem *m_currentItem;
    QToolBar *m_toolbar;
    QAction *m_splitAction;
    QList<QAction *> m_addedToolBarActions;
    SideBar *m_sideBar;
    QToolButton *m_splitButton;
    QToolButton *m_closeButton;
};

class ComboBox : public QComboBox
{
    Q_OBJECT

public:
    ComboBox(SideBarWidget *sideBarWidget);

protected:
    bool event(QEvent *event);

private:
    SideBarWidget *m_sideBarWidget;
};

} // namespace Internal
} // namespace Core

#endif // SIDEBAR_H
