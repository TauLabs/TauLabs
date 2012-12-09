/**
 ******************************************************************************
 * @file       pathplannergadgetwidget.h
 * @author     PhoenixPilot Project, http://github.com/PhoenixPilot Copyright (C) 2012.
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup Path Planner GCS Plugins
 * @{
 * @addtogroup PathPlannerGadgetPlugin Path Planner Gadget Plugin
 * @{
 * @brief A gadget to edit a list of waypoints
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

#ifndef PathPlannerGADGETWIDGET_H_
#define PathPlannerGADGETWIDGET_H_

#include <QtGui/QLabel>
#include <QItemSelectionModel>
#include "flightdatamodel.h"
#include "modeluavoproxy.h"

class Ui_PathPlanner;

class PATHPLANNER_EXPORT PathPlannerGadgetWidget : public QLabel
{
    Q_OBJECT

public:
    PathPlannerGadgetWidget(QWidget *parent = 0);
    ~PathPlannerGadgetWidget();

    void setModel(FlightDataModel *model, QItemSelectionModel *selection);
private slots:
    void rowsInserted ( const QModelIndex & parent, int start, int end );

    void on_tbAdd_clicked();

    void on_tbDelete_clicked();

    void on_tbInsert_clicked();

    void on_tbReadFromFile_clicked();

    void on_tbSaveToFile_clicked();

    void on_tbDetails_clicked();

    void on_tbSendToUAV_clicked();

    void on_tbFetchFromUAV_clicked();

private:
    Ui_PathPlanner  *ui;
    FlightDataModel *model;
    ModelUavoProxy  *proxy;
    QItemSelectionModel *selection;

signals:
    void sendPathPlanToUAV();
    void receivePathPlanFromUAV();
};

#endif /* PathPlannerGADGETWIDGET_H_ */
