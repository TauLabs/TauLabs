/**
 ******************************************************************************
 * @file       waypointdelegate.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @addtogroup GCSPlugins GCS Plugins
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

#ifndef WAYPOINTDELEGATE_H
#define WAYPOINTDELEGATE_H

#include <QStyledItemDelegate>
#include <QComboBox>
#include <QEvent>

/**
 * @brief The WaypointDelegate class is used to handle updating the values in
 * the mode combo box to the data model.
 */
class WaypointDelegate : public QStyledItemDelegate
 {
        Q_OBJECT

 public:

    WaypointDelegate(QObject *parent = 0);

    //! Create the QComboxBox for the mode or pass to the default implementation
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                          const QModelIndex &index) const;

    //! This filter is required to make combo boxes work
    bool eventFilter(QObject *object, QEvent *event);

    //! Set data in the UI when the model is changed
    void setEditorData(QWidget *editor, const QModelIndex &index) const;

    //! Set data in the model when the UI is changed
    void setModelData(QWidget *editor, QAbstractItemModel *model,
                      const QModelIndex &index) const;

    //!  Update the size of the editor widget
    void updateEditorGeometry(QWidget *editor,
                              const QStyleOptionViewItem &option, const QModelIndex &index) const;

    //! Convert the variant to a string value
    QString displayText ( const QVariant & value, const QLocale & locale ) const;

    //! Populate the selections in the mode combo box
    void loadComboBox(QComboBox * combo) const;
 };


#endif // WAYPOINTDELEGATE_H
