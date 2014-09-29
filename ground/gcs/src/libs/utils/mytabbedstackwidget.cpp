/**
 ******************************************************************************
 *
 * @file       mytabbedstackwidget.cpp
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
#include "mytabbedstackwidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QtCore/QDebug>

MyTabbedStackWidget::MyTabbedStackWidget(QWidget *parent, bool isVertical, bool iconAbove)
    : QWidget(parent),
      m_vertical(isVertical),
      m_iconAbove(iconAbove),
      m_maxLabelSize(8)
{
    m_listWidget = new QListWidget(this);
    m_listWidget->setStyleSheet(QString::fromUtf8(
                                      "QScrollBar:vertical {"
                                      "    border: 1px solid #999999;"
                                      "    background:#c8c8c8;;"
                                      "    width:10px;    "
                                      "    margin: 0px 0px 0px 0px;"
                                      "}"
                                      "QScrollBar::handle:vertical {"
                                      "background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,"
                                      "    stop: 0 #66e, stop: 1 #bbf);"
                                      "background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,"
                                      "    stop: 0 #bbf, stop: 1 #55f);"
                                      "    min-height: 0px;"
                                      ""
                                      "}"
                                      "QScrollBar::add-line:vertical {"
                                      "background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,"
                                      "    stop: 0 #66e, stop: 1 #bbf);"
                                      "background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,"
                                      "    stop: 0 #bbf, stop: 1 #55f);"
                                      "    height: px;"
                                      "    subcontrol-position: bottom;"
                                      "    subcontrol-origin: margin;"
                                      "}"
                                      "QScrollBar::sub-line:vertical {"
                                      "background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,"
                                      "    stop: 0 #66e, stop: 1 #bbf);"
                                      "background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,"
                                      "    stop: 0 #bbf, stop: 1 #55f);"
                                      "    height: 0px;"
                                      "    subcontrol-position: top;"
                                      "    subcontrol-origin: margin;"
                                      "}"
                                      ""));
    m_stackWidget = new QStackedWidget();
    m_stackWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    QBoxLayout *toplevelLayout;
    if (m_vertical) {
        toplevelLayout = new QHBoxLayout;
        toplevelLayout->addWidget(m_listWidget);
        toplevelLayout->addWidget(m_stackWidget);
        m_listWidget->setFlow(QListView::TopToBottom);
        m_listWidget->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);
        m_listWidget->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    } else {
        toplevelLayout = new QVBoxLayout;
        toplevelLayout->addWidget(m_stackWidget);
        toplevelLayout->addWidget(m_listWidget);
        m_listWidget->setFlow(QListView::LeftToRight);
        m_listWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        m_listWidget->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    }

    if (m_iconAbove && m_vertical) {
        m_listWidget->setFixedWidth(80); // this should be computed instead
        m_listWidget->setWrapping(false);
    }

    toplevelLayout->setSpacing(0);
    toplevelLayout->setContentsMargins(0, 0, 0, 0);
    m_listWidget->setContentsMargins(0, 0, 0, 0);
    m_listWidget->setSpacing(0);
    m_listWidget->setViewMode(QListView::IconMode);
    m_stackWidget->setContentsMargins(0, 0, 0, 0);
    setLayout(toplevelLayout);

    connect(m_listWidget, SIGNAL(currentRowChanged(int)), this, SLOT(showWidget(int)),Qt::QueuedConnection);
}

void MyTabbedStackWidget::insertTab(const int index, QWidget *tab, const QIcon &icon, const QString &label)
{
    tab->setContentsMargins(0, 0, 0, 0);
    m_stackWidget->insertWidget(index, tab);
    QString adjLabel = label;
    if (adjLabel.size() > m_maxLabelSize)
        adjLabel = adjLabel.left(m_maxLabelSize - 3) + "...";
    QListWidgetItem *item = new QListWidgetItem(icon, adjLabel);
    item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
    item->setTextAlignment(Qt::AlignHCenter | Qt::AlignBottom);
    item->setToolTip(label);
    m_listWidget->insertItem(index, item);
}

void MyTabbedStackWidget::removeTab(int index)
{
    QWidget * wid=m_stackWidget->widget(index);
    m_stackWidget->removeWidget(wid);
    delete wid;
    QListWidgetItem *item = m_listWidget->item(index);
    m_listWidget->removeItemWidget(item);
    delete item;
}

int MyTabbedStackWidget::currentIndex() const
{
    return m_listWidget->currentRow();
}

void MyTabbedStackWidget::setCurrentIndex(int index)
{
    m_listWidget->setCurrentRow(index);
}

void MyTabbedStackWidget::showWidget(int index)
{
    if(m_stackWidget->currentIndex()==index)
        return;
    bool proceed=false;
    emit currentAboutToShow(index,&proceed);
    if(proceed)
    {
        m_stackWidget->setCurrentIndex(index);
        emit currentChanged(index);
    }
    else
    {
        m_listWidget->setCurrentRow(m_stackWidget->currentIndex(),QItemSelectionModel::ClearAndSelect);
    }
}

void MyTabbedStackWidget::setMaxLabelSize(int maxLabelSize)
{
    m_maxLabelSize = maxLabelSize;
}

void MyTabbedStackWidget::insertCornerWidget(int index, QWidget *widget)
{
    Q_UNUSED(index);

    widget->hide();
}

