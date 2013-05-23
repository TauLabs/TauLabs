/**
 ******************************************************************************
 * @file       waypointdialog.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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

/**
 * @brief The WaypointDialog class creates a dialog for editing the properties of a single waypoint
 */
class PATHPLANNER_EXPORT WaypointDialog : public QDialog
{
    Q_OBJECT

public:
    WaypointDialog(QWidget *parent, QAbstractItemModel *model, QItemSelectionModel * selection);
    ~WaypointDialog();

    //! Edit the requested waypoint, show dialog if it is not showing
    void editWaypoint(int number);

private slots:
    //! Called when the data widget selector index changes
    void currentIndexChanged(int index);

    //! Updates the UI to display the available options (e.g. radius) when mode changes
    void setupModeWidgets();

    //! Close the dialog, abort any changes
    void onCancelButton_clicked();

    //! Close the dialog, accept any changes
    void onOkButton_clicked();

    //! User requests the previous waypoint
    void onPreviousButton_clicked();

    //! User requests the next waypoint
    void onNextButton_clicked();

    //! When the selector changes pass the update to the data mapper
    void currentRowChanged(QModelIndex,QModelIndex);

    //! Enable or disable the controls based on the lock control
    void enableEditWidgets();
private:

    //! The handle to the UI
    Ui_waypoint_dialog *ui;

    //! Delegate between the model (one waypoint) and the view
    QDataWidgetMapper *mapper;

    //! The data model for the flight plan
    QAbstractItemModel *model;

    //! Indicates which waypoint is selected for editing
    QItemSelectionModel * itemSelection;

    //! Store the current waypoint index
    int index;
};

#endif /* WAYPOINT_DIALOG_H */
