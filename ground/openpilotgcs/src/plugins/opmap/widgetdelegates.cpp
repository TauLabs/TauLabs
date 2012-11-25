/**
 ******************************************************************************
 *
 * @file       widgetdelegates.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup OPMapPlugin OpenPilot Map Plugin
 * @{
 * @brief The OpenPilot Map plugin
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
#include "widgetdelegates.h"
#include <QComboBox>
#include <QRadioButton>
#include <QDebug>
QWidget *MapDataDelegate::createEditor(QWidget *parent,
                                        const QStyleOptionViewItem & option,
                                        const QModelIndex & index) const
{
    int column=index.column();
    QComboBox * box;
    switch(column)
    {
    case flightDataModel::MODE:
        box=new QComboBox(parent);
        MapDataDelegate::loadComboBox(box,flightDataModel::MODE);
        return box;
        break;
    default:
        return QItemDelegate::createEditor(parent,option,index);
        break;
    }

    QComboBox *editor = new QComboBox(parent);
    return editor;
}

void MapDataDelegate::setEditorData(QWidget *editor,
                                     const QModelIndex &index) const
{
    if(!index.isValid())
        return;
    QString className=editor->metaObject()->className();
    if (className.contains("QComboBox")) {
        int value = index.model()->data(index, Qt::EditRole).toInt();
        QComboBox *comboBox = static_cast<QComboBox*>(editor);
        int x=comboBox->findData(value);
        qDebug()<<"VALUE="<<x;
        comboBox->setCurrentIndex(x);
    }
    else
        QItemDelegate::setEditorData(editor, index);
}

void MapDataDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                    const QModelIndex &index) const
{
    QString className=editor->metaObject()->className();
    if (className.contains("QComboBox")) {
        QComboBox *comboBox = static_cast<QComboBox*>(editor);
        int value = comboBox->itemData(comboBox->currentIndex()).toInt();
        model->setData(index, value, Qt::EditRole);
    }
    else
        QItemDelegate::setModelData(editor,model,index);
}

void MapDataDelegate::updateEditorGeometry(QWidget *editor,
                                            const QStyleOptionViewItem &option, const QModelIndex &/* index */) const
{
    editor->setGeometry(option.rect);
}

void MapDataDelegate::loadComboBox(QComboBox *combo, flightDataModel::pathPlanDataEnum type)
{
    switch(type)
    {
    case flightDataModel::MODE:
        combo->addItem("Fly Direct",MODE_FLYENDPOINT);
        combo->addItem("Fly Vector",MODE_FLYVECTOR);
        combo->addItem("Fly Circle Right",MODE_FLYCIRCLERIGHT);
        combo->addItem("Fly Circle Left",MODE_FLYCIRCLELEFT);

        combo->addItem("Drive Direct",MODE_DRIVEENDPOINT);
        combo->addItem("Drive Vector",MODE_DRIVEVECTOR);
        combo->addItem("Drive Circle Right",MODE_DRIVECIRCLELEFT);
        combo->addItem("Drive Circle Left",MODE_DRIVECIRCLERIGHT);

        break;
    default:
        break;
    }
}

MapDataDelegate::MapDataDelegate(QObject *parent):QItemDelegate(parent)
{
}
