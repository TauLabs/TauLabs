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
#include "qxttablewidgetitem.h"
#include "qxttablewidget.h"

/*!
    \class QxtTableWidgetItem
    \inmodule QxtGui
    \brief The QxtTableWidgetItem class is an extended QTableWidgetItem.

    QxtTableWidgetItem provides means for offering check state change signals and
    convenience methods for testing and setting flags.

    \sa QxtTableWidget
 */

/*!
    Constructs a new QxtTableWidgetItem with \a type.
 */
QxtTableWidgetItem::QxtTableWidgetItem(int type)
        : QTableWidgetItem(type)
{
}

/*!
    Constructs a new QxtTableWidgetItem with \a text and \a type.
 */
QxtTableWidgetItem::QxtTableWidgetItem(const QString& text, int type)
        : QTableWidgetItem(text, type)
{
}

/*!
    Constructs a new QxtTableWidgetItem with \a icon, \a text and \a type.
 */
QxtTableWidgetItem::QxtTableWidgetItem(const QIcon& icon, const QString& text, int type)
        : QTableWidgetItem(icon, text, type)
{
}

/*!
    Constructs a copy of \a other.
 */
QxtTableWidgetItem::QxtTableWidgetItem(const QTableWidgetItem& other)
        : QTableWidgetItem(other)
{
}

/*!
    Destructs the table widget item.
 */
QxtTableWidgetItem::~QxtTableWidgetItem()
{
}

/*!
    Returns \c true if the \a flag is set, otherwise \c false.

    \sa setFlag(), QTableWidgetItem::flags(), Qt::ItemFlag
 */
bool QxtTableWidgetItem::testFlag(Qt::ItemFlag flag) const
{
    return (flags() & flag);
}

/*!
    If \a enabled is \c true, the item \a flag is enabled; otherwise, it is disabled.

    \sa testFlag(), QTableWidgetItem::setFlags(), Qt::ItemFlag
 */
void QxtTableWidgetItem::setFlag(Qt::ItemFlag flag, bool enabled)
{
    if (enabled)
        setFlags(flags() | flag);
    else
        setFlags(flags() & ~flag);
}

/*!
    \reimp
 */
void QxtTableWidgetItem::setData(int role, const QVariant& value)
{
    if (role == Qt::CheckStateRole)
    {
        const Qt::CheckState newState = static_cast<Qt::CheckState>(value.toInt());
        const Qt::CheckState oldState = static_cast<Qt::CheckState>(data(role).toInt());

        QTableWidgetItem::setData(role, value);

        if (newState != oldState)
        {
            QxtTableWidget* table = qobject_cast<QxtTableWidget*>(tableWidget());
            if (table)
            {
                emit table->itemCheckStateChanged(this);
            }
        }
    }
    else
    {
        QTableWidgetItem::setData(role, value);
    }
}
