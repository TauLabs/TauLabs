/**
 ******************************************************************************
 * @file       waypointdelegate.cpp
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @addtogroup PhoenixPilot GCS Plugins
 * @{
 * @addtogroup PathPlanner Map Plugin
 * @{
 * @brief Delegate between the flight data model and the views (provides a
 * QComboBox)
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

#include <waypointdelegate.h>
#include <flightdatamodel.h>

WaypointDelegate::WaypointDelegate(QObject *parent):QStyledItemDelegate(parent)
{
}

/**
 * @brief WaypointDelegate::createEditor Returns the widget used to edit the item
 * specified by index for editing. The parent widget and style option are used to
 * control how the editor widget appears.
 * @return The widget for the index
 */
QWidget *WaypointDelegate::createEditor(QWidget *parent,
                                        const QStyleOptionViewItem & option,
                                        const QModelIndex & index) const
{
    int column=index.column();
    QComboBox * box;
    switch(column)
    {
    case FlightDataModel::MODE:
        box=new QComboBox(parent);
        loadComboBox(box);
        return box;
        break;
    default:
        return QStyledItemDelegate::createEditor(parent,option,index);
        break;
    }

    QComboBox *editor = new QComboBox(parent);
    return editor;
}

/**
 * @brief WaypointDelegate::setEditorData Set the data in the UI from the model
 * for a particular element index
 * @param editor The editor dialog
 * @param index The model parameter index to use
 */
void WaypointDelegate::setEditorData(QWidget *editor,
                                     const QModelIndex &index) const
{
    if(!index.isValid())
        return;
    if (index.column() == (int) FlightDataModel::MODE) {
        QComboBox *comboBox = static_cast<QComboBox*>(editor);
        int value = index.model()->data(index, Qt::EditRole).toInt();
        comboBox->setCurrentIndex(value);
    }
    else
        QStyledItemDelegate::setEditorData(editor, index);
}

/**
 * @brief WaypointDelegate::setModelData Update the model from the UI for a particular
 * element index
 * @param editor
 * @param model The editor dialog
 * @param index The model parameter index to change
 */
void WaypointDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                    const QModelIndex &index) const
{
    if(!index.isValid())
        return;
    if (index.column() == (int) FlightDataModel::MODE) {
        QComboBox *comboBox = static_cast<QComboBox*>(editor);
        Q_ASSERT(comboBox != NULL);
        int value = comboBox->itemData(comboBox->currentIndex()).toInt();
        model->setData(index, value, Qt::EditRole);
    }
    else
        QStyledItemDelegate::setModelData(editor,model,index);
}

/**
 * @brief WaypointDelegate::updateEditorGeometry Update the size of the editor widget
 */
void WaypointDelegate::updateEditorGeometry(QWidget *editor,
                                            const QStyleOptionViewItem &option, const QModelIndex &/* index */) const
{
    editor->setGeometry(option.rect);
}

/**
 * @brief WaypointDelegate::loadComboBox Populate the combo box with the list of flight modes
 * @param combo The QComboBox to add options to
 * @param type
 */
void WaypointDelegate::loadComboBox(QComboBox *combo) const
{
    QList<int> keys = FlightDataModel::modeNames.keys();
    foreach (const int k, keys)
        combo->addItem(FlightDataModel::modeNames.value(k), k);
}
