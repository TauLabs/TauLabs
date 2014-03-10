/**
 ******************************************************************************
 *
 * @file       sidebar.cpp
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

#include "sidebar.h"
#include "imode.h"
#include "modemanager.h"

#include "actionmanager/actionmanager.h"

#include <QtCore/QDebug>
#include <QtCore/QEvent>
#include <QtCore/QSettings>
#include <QLayout>
#include <QToolBar>
#include <QAction>
#include <QToolButton>

using namespace Core;
using namespace Core::Internal;

SideBarItem::~SideBarItem()
{
    delete m_widget;
}

SideBar::SideBar(QList<SideBarItem*> itemList,
                 QList<SideBarItem*> defaultVisible)
{
    setOrientation(Qt::Vertical);
    foreach (SideBarItem *item, itemList) {
        const QString title = item->widget()->windowTitle();
        m_itemMap.insert(title, item);
    }
    foreach (SideBarItem *item, defaultVisible) {
        if (!itemList.contains(item))
            continue;
        m_defaultVisible.append(item->widget()->windowTitle());
    }

    m_availableItems = m_itemMap.keys();
}

SideBar::~SideBar()
{
    qDeleteAll(m_itemMap.values());
}

QStringList SideBar::availableItems() const
{
    return m_availableItems;
}

void SideBar::makeItemAvailable(SideBarItem *item)
{
    QMap<QString, SideBarItem*>::const_iterator it = m_itemMap.constBegin();
    while (it != m_itemMap.constEnd()) {
        if (it.value() == item) {
            m_availableItems.append(it.key());
            qSort(m_availableItems);
            break;
        }
        ++it;
    }
}

SideBarItem *SideBar::item(const QString &title)
{
    if (m_itemMap.contains(title)) {
        m_availableItems.removeAll(title);
        return m_itemMap.value(title);
    }
    return 0;
}

SideBarWidget *SideBar::insertSideBarWidget(int position, const QString &title)
{
    SideBarWidget *item = new SideBarWidget(this, title);
    connect(item, SIGNAL(splitMe()), this, SLOT(splitSubWidget()));
    connect(item, SIGNAL(closeMe()), this, SLOT(closeSubWidget()));
    connect(item, SIGNAL(currentWidgetChanged()), this, SLOT(updateWidgets()));
    insertWidget(position, item);
    m_widgets.insert(position, item);
    updateWidgets();
    return item;
}

void SideBar::removeSideBarWidget(SideBarWidget *widget)
{
    widget->removeCurrentItem();
    m_widgets.removeOne(widget);
    widget->hide();
    widget->deleteLater();
}

void SideBar::splitSubWidget()
{
    SideBarWidget *original = qobject_cast<SideBarWidget*>(sender());
    int pos = indexOf(original) + 1;
    insertSideBarWidget(pos);
    updateWidgets();
}

void SideBar::closeSubWidget()
{
    if (m_widgets.count() != 1) {
        SideBarWidget *widget = qobject_cast<SideBarWidget*>(sender());
        if (!widget)
            return;
        removeSideBarWidget(widget);
        updateWidgets();
    }
}

void SideBar::updateWidgets()
{
    foreach (SideBarWidget *i, m_widgets)
        i->updateAvailableItems();
}

void SideBar::saveSettings(QSettings *settings)
{
    QStringList views;
    for (int i = 0; i < m_widgets.count(); ++i)
        views.append(m_widgets.at(i)->currentItemTitle());
    settings->setValue("HelpSideBar/Views", views);
    settings->setValue("HelpSideBar/Visible", true);//isVisible());
    settings->setValue("HelpSideBar/VerticalPosition", saveState());
    settings->setValue("HelpSideBar/Width", width());
}

void SideBar::readSettings(QSettings *settings)
{
    foreach (SideBarWidget *widget, m_widgets)
        removeSideBarWidget(widget);

    if (settings->contains("HelpSideBar/Views")) {
        QStringList views = settings->value("HelpSideBar/Views").toStringList();
        if (views.count()) {
            foreach (const QString &title, views)
                insertSideBarWidget(m_widgets.count(), title);
        } else {
            insertSideBarWidget(0);
        }
    } else {
        foreach (const QString &title, m_defaultVisible)
            insertSideBarWidget(m_widgets.count(), title);
    }

    if (settings->contains("HelpSideBar/Visible"))
        setVisible(settings->value("HelpSideBar/Visible").toBool());

    if (settings->contains("HelpSideBar/VerticalPosition"))
        restoreState(settings->value("HelpSideBar/VerticalPosition").toByteArray());

    if (settings->contains("HelpSideBar/Width")) {
        QSize s = size();
        s.setWidth(settings->value("HelpSideBar/Width").toInt());
        resize(s);
    }
}

void SideBar::activateItem(SideBarItem *item)
{
    QMap<QString, SideBarItem*>::const_iterator it = m_itemMap.constBegin();
    QString title;
    while (it != m_itemMap.constEnd()) {
        if (it.value() == item) {
            title = it.key();
            break;
        }
        ++it;
    }

    if (title.isEmpty())
        return;

    for (int i = 0; i < m_widgets.count(); ++i) {
        if (m_widgets.at(i)->currentItemTitle() == title) {
            item->widget()->setFocus();
            return;
        }
    }

    SideBarWidget *widget = m_widgets.first();
    widget->setCurrentItem(title);
    updateWidgets();
    item->widget()->setFocus();
}

void SideBar::setShortcutMap(const QMap<QString, Core::Command*> &shortcutMap)
{
    m_shortcutMap = shortcutMap;
}

QMap<QString, Core::Command*> SideBar::shortcutMap() const
{
    return m_shortcutMap;
}



SideBarWidget::SideBarWidget(SideBar *sideBar, const QString &title)
    : m_currentItem(0)
    , m_sideBar(sideBar)
{
    m_comboBox = new ComboBox(this);
    m_comboBox->setMinimumContentsLength(15);

    m_toolbar = new QToolBar(this);
    m_toolbar->setContentsMargins(0, 0, 0, 0);
    m_toolbar->addWidget(m_comboBox);

    m_splitButton = new QToolButton;
    m_splitButton->setIcon(QIcon(":/core/images/splitbutton_horizontal.png"));
    m_splitButton->setToolTip(tr("Split"));
    connect(m_splitButton, SIGNAL(clicked(bool)), this, SIGNAL(splitMe()));

    m_closeButton = new QToolButton;
    m_closeButton->setIcon(QIcon(":/core/images/closebutton.png"));
    m_closeButton->setToolTip(tr("Close"));

    connect(m_closeButton, SIGNAL(clicked(bool)), this, SIGNAL(closeMe()));

    QWidget *spacerItem = new QWidget(this);
    spacerItem->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    m_toolbar->addWidget(spacerItem);
    m_splitAction = m_toolbar->addWidget(m_splitButton);
    m_toolbar->addWidget(m_closeButton);

    QVBoxLayout *lay = new QVBoxLayout();
    lay->setMargin(0);
    lay->setSpacing(0);
    setLayout(lay);
    lay->addWidget(m_toolbar);

    const QStringList lst = m_sideBar->availableItems();
    QString t = title;
    if (lst.count()) {
        m_comboBox->addItems(lst);
        m_comboBox->setCurrentIndex(0);
        if (t.isEmpty())
            t = m_comboBox->currentText();
    }
    setCurrentItem(t);

    connect(m_comboBox, SIGNAL(currentIndexChanged(int)),
            this, SLOT(setCurrentIndex(int)));
}

SideBarWidget::~SideBarWidget()
{
}

QString SideBarWidget::currentItemTitle() const
{
    return m_comboBox->currentText();
}

void SideBarWidget::setCurrentItem(const QString &title)
{
    if (!title.isEmpty()) {
        int idx = m_comboBox->findText(title);
        if (idx < 0)
            idx = 0;
        bool blocked = m_comboBox->blockSignals(true);
        m_comboBox->setCurrentIndex(idx);
        m_comboBox->blockSignals(blocked);
    }

    SideBarItem *item = m_sideBar->item(title);
    if (!item)
        return;
    removeCurrentItem();
    m_currentItem = item;
    layout()->addWidget(m_currentItem->widget());

    // Add buttons and remember their actions for later removal
    foreach (QToolButton *b, m_currentItem->createToolBarWidgets())
        m_addedToolBarActions.append(m_toolbar->insertWidget(m_splitAction, b));
}

void SideBarWidget::updateAvailableItems()
{
    bool blocked = m_comboBox->blockSignals(true);
    QString current = m_comboBox->currentText();
    m_comboBox->clear();
    QStringList itms = m_sideBar->availableItems();
    if (!current.isEmpty() && !itms.contains(current))
        itms.append(current);
    qSort(itms);
    m_comboBox->addItems(itms);
    int idx = m_comboBox->findText(current);
    if (idx < 0)
        idx = 0;
    m_comboBox->setCurrentIndex(idx);
    m_splitButton->setEnabled(itms.count() > 1);
    m_comboBox->blockSignals(blocked);
}

void SideBarWidget::removeCurrentItem()
{
    if (!m_currentItem)
        return;

    QWidget *w = m_currentItem->widget();
    layout()->removeWidget(w);
    w->setParent(0);
    m_sideBar->makeItemAvailable(m_currentItem);

    // Delete custom toolbar widgets
    qDeleteAll(m_addedToolBarActions);
    m_addedToolBarActions.clear();

    m_currentItem = 0;
}

void SideBarWidget::setCurrentIndex(int)
{
    setCurrentItem(m_comboBox->currentText());
    emit currentWidgetChanged();
}

Core::Command *SideBarWidget::command(const QString &title) const
{
    const QMap<QString, Core::Command*> shortcutMap = m_sideBar->shortcutMap();
    QMap<QString, Core::Command*>::const_iterator r = shortcutMap.find(title);
    if (r != shortcutMap.end())
        return r.value();
    return 0;
}



ComboBox::ComboBox(SideBarWidget *sideBarWidget)
    : m_sideBarWidget(sideBarWidget)
{
}

bool ComboBox::event(QEvent *e)
{
    if (e->type() == QEvent::ToolTip) {
        QString txt = currentText();
        Core::Command *cmd = m_sideBarWidget->command(txt);
        if (cmd) {
            txt = tr("Activate %1").arg(txt);
            setToolTip(cmd->stringWithAppendedShortcut(txt));
        } else {
            setToolTip(txt);
        }
    }
    return QComboBox::event(e);
}
