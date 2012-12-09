/**
 ******************************************************************************
 * @file       waypointdialog.h
 * @author     PhoenixPilot Project, http://github.com/PhoenixPilot Copyright (C) 2012.
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup PathPlanner OpenPilot Map Plugin
 * @{
 * @brief Waypoint editor dialog
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

#ifndef WAYPOINT_DIALOG_H
#define WAYPOINT_DIALOG_H

#include <QComboBox>
#include <QDialog>
#include <QDataWidgetMapper>
#include <QItemDelegate>
#include <QItemSelectionModel>
#include "flightdatamodel.h"

class Ui_waypoint_dialog;

class PATHPLANNER_EXPORT WaypointDialog : public QWidget
{
    Q_OBJECT

public:
    WaypointDialog(QWidget *parent, QAbstractItemModel * model, QItemSelectionModel * selection);
    ~WaypointDialog();

    //! Edit the requested waypoint, show dialog if it is not showing
    void editWaypoint(int number);

private:
    Ui_waypoint_dialog *ui;
    QDataWidgetMapper *mapper;
    QAbstractItemModel * model;
    QItemSelectionModel * itemSelection;

private slots:

private slots:
    void currentIndexChanged(int index);
    void setupModeWidgets();

    //! Close the dialog, abort any changes
    void on_cancelButton_clicked();

    //! Close the dialog, accept any changes
    void on_okButton_clicked();

    //! User requests the previous waypoint
    void on_previousButton_clicked();

    //! User requests the next waypoint
    void on_nextButton_clicked();
    void enableEditWidgets(bool);
    void currentRowChanged(QModelIndex,QModelIndex);
};


/**
 * @brief The WaypointDataDelegate class is used to handle updating the values in
 * the mode combo box to the data model.
 */
class WaypointDataDelegate : public QItemDelegate
 {
        Q_OBJECT

 public:
    typedef enum { MODE_FLYENDPOINT=0, MODE_FLYVECTOR=1, MODE_FLYCIRCLERIGHT=2, MODE_FLYCIRCLELEFT=3,
                   MODE_DRIVEENDPOINT=4, MODE_DRIVEVECTOR=5, MODE_DRIVECIRCLELEFT=6, MODE_DRIVECIRCLERIGHT=7
                 } ModeOptions;

     WaypointDataDelegate(QObject *parent = 0);

     QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                           const QModelIndex &index) const;

     //! Set data in the UI when the model is changed
     void setEditorData(QWidget *editor, const QModelIndex &index) const;

     //! Set data in the model when the UI is changed
     void setModelData(QWidget *editor, QAbstractItemModel *model,
                       const QModelIndex &index) const;

     void updateEditorGeometry(QWidget *editor,
                               const QStyleOptionViewItem &option, const QModelIndex &index) const;

     //! Populate the selections in the mode combo box
     static void loadComboBox(QComboBox * combo,FlightDataModel::pathPlanDataEnum type);
 };

#endif /* WAYPOINT_DIALOG_H */
