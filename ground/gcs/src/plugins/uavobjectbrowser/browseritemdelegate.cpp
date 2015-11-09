/**
 ******************************************************************************
 *
 * @file       browseritemdelegate.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
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

#include "uavobjectbrowserwidget.h"
#include "browseritemdelegate.h"
#include "fieldtreeitem.h"

BrowserItemDelegate::BrowserItemDelegate(TreeSortFilterProxyModel *proxyModel, QObject *parent) :
        QStyledItemDelegate(parent)
{
    this->proxyModel=proxyModel;
}

QWidget *BrowserItemDelegate::createEditor(QWidget *parent,
                                           const QStyleOptionViewItem & option ,
                                           const QModelIndex & proxyIndex ) const
{
    Q_UNUSED(option)
    QModelIndex index = proxyModel->mapToSource(proxyIndex);
    FieldTreeItem *item = static_cast<FieldTreeItem*>(index.internalPointer());
    QWidget *editor = item->createEditor(parent);
    Q_ASSERT(editor);
    return editor;
}

/**
 * @brief BrowserItemDelegate::eventFilter Filter any events that are
 * on the combox box from going to the view.  This makes the combo
 * box contents pop up and be selectable.
 */
bool BrowserItemDelegate::eventFilter(QObject *object, QEvent *event)
{
    QComboBox * comboBox = dynamic_cast<QComboBox*>(object);
    if (comboBox)
    {
        if (event->type() == QEvent::MouseButtonRelease)
        {
            comboBox->showPopup();
            return true;
        }
    }
    else
    {
        return QStyledItemDelegate::eventFilter( object, event );
    }
    return false;
}

void BrowserItemDelegate::setEditorData(QWidget *editor,
                                        const QModelIndex &proxyIndex) const
{
    QModelIndex index = proxyModel->mapToSource(proxyIndex);
    FieldTreeItem *item = static_cast<FieldTreeItem*>(index.internalPointer());
    QVariant value = proxyIndex.model()->data(proxyIndex, Qt::EditRole);
    item->setEditorValue(editor, value);
}

void BrowserItemDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                       const QModelIndex &proxyIndex) const
{
    QModelIndex index = proxyModel->mapToSource(proxyIndex);
    FieldTreeItem *item = static_cast<FieldTreeItem*>(index.internalPointer());
    QVariant value = item->getEditorValue(editor);
    bool ret = model->setData(proxyIndex, value, Qt::EditRole);
    Q_ASSERT(ret);
}

void BrowserItemDelegate::updateEditorGeometry(QWidget *editor,
                                               const QStyleOptionViewItem &option, const QModelIndex &/* index */) const
{
    editor->setGeometry(option.rect);
}

QSize BrowserItemDelegate::sizeHint(const QStyleOptionViewItem & option, const QModelIndex &index) const
{
    Q_UNUSED(option);
    Q_UNUSED(index);
    return QSpinBox().sizeHint();
}
