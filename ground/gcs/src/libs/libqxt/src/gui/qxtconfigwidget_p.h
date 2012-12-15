/****************************************************************************
 **
 ** Copyright (C) Qxt Foundation. Some rights reserved.
 **
 ** This file is part of the QxtGui module of the Qxt library.
 **
 ** This library is free software; you can redistribute it and/or modify it
 ** under the terms of the Common Public License, version 1.0, as published
 ** by IBM, and/or under the terms of the GNU Lesser General Public License,
 ** version 2.1, as published by the Free Software Foundation.
 **
 ** This file is provided "AS IS", without WARRANTIES OR CONDITIONS OF ANY
 ** KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT LIMITATION, ANY
 ** WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY OR
 ** FITNESS FOR A PARTICULAR PURPOSE.
 **
 ** You should have received a copy of the CPL and the LGPL along with this
 ** file. See the LICENSE file and the cpl1.0.txt/lgpl-2.1.txt files
 ** included with the source distribution for more information.
 ** If you did not receive a copy of the licenses, contact the Qxt Foundation.
 **
 ** <http://libqxt.org>  <foundation@libqxt.org>
 **
 ****************************************************************************/
#ifndef QXTCONFIGWIDGET_P_H
#define QXTCONFIGWIDGET_P_H

#include "qxtconfigwidget.h"
#include <QItemDelegate>
#include <QTableWidget>

QT_FORWARD_DECLARE_CLASS(QSplitter)
QT_FORWARD_DECLARE_CLASS(QStackedWidget)
QT_FORWARD_DECLARE_CLASS(QDialogButtonBox)

class QxtConfigTableWidget : public QTableWidget
{
public:
    QxtConfigTableWidget(QWidget* parent = 0);
    QStyleOptionViewItem viewOptions() const;
    QSize sizeHint() const;

    bool hasHoverEffect() const;
    void setHoverEffect(bool enabled);
};

class QxtConfigDelegate : public QItemDelegate
{
public:
    QxtConfigDelegate(QObject* parent = 0);
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const;
    bool hover;
};

class QxtConfigWidgetPrivate : public QObject, public QxtPrivate<QxtConfigWidget>
{
    Q_OBJECT

public:
    QXT_DECLARE_PUBLIC(QxtConfigWidget)

    void init(QxtConfigWidget::IconPosition position = QxtConfigWidget::West);
    void initTable();
    void relayout();
    QTableWidgetItem* item(int index) const;

    QSplitter* splitter;
    QStackedWidget* stack;
    QxtConfigTableWidget* table;
    QxtConfigWidget::IconPosition pos;

public Q_SLOTS:
    void setCurrentIndex(int row, int column);
    void setCurrentIndex(int index);
};

#endif // QXTCONFIGWIDGET_P_H
