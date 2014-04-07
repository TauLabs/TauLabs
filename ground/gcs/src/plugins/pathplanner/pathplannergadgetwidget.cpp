/**
 ******************************************************************************
 * @file       pathplannergadgetwidget.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @addtogroup GCSPlugins GCS Plugins
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
#include "pathplannergadgetwidget.h"
#include "waypointdialog.h"
#include "waypointdelegate.h"
#include "ui_pathplanner.h"
#include <QFileDialog>
#include <QString>
#include <QStringList>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>

#include "algorithms/pathfillet.h"
#include "extensionsystem/pluginmanager.h"

PathPlannerGadgetWidget::PathPlannerGadgetWidget(QWidget *parent) : QLabel(parent), prevModel(NULL)
{
    ui = new Ui_PathPlanner();
    ui->setupUi(this);
    ui->statusTL->setVisible(false);
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    FlightDataModel *model = pm->getObject<FlightDataModel>();
    Q_ASSERT(model);

    QItemSelectionModel *selection = pm->getObject<QItemSelectionModel>();
    Q_ASSERT(selection);
    setModel(model, selection);
}

PathPlannerGadgetWidget::~PathPlannerGadgetWidget()
{
   // Do nothing
}


void PathPlannerGadgetWidget::setModel(FlightDataModel *model, QItemSelectionModel *selection)
{
    proxy = new ModelUavoProxy(this, model);

    this->model = model;
    this->selection = selection;

    ui->tableView->setModel(model);
    ui->tableView->setSelectionModel(selection);
    ui->tableView->setSelectionBehavior(QAbstractItemView::SelectRows);

    ui->tableView->setItemDelegate(new WaypointDelegate(this));
    ui->tableView->resizeColumnsToContents();

    ui->tableView->resizeColumnsToContents();

    connect(proxy, SIGNAL(sendPathPlanToUavProgress(int)), this, SLOT(on_waypointSendProgress(int)));
}

void PathPlannerGadgetWidget::on_tbAdd_clicked()
{
    ui->tableView->model()->insertRow(ui->tableView->model()->rowCount());
}

void PathPlannerGadgetWidget::on_tbDelete_clicked()
{
    ui->tableView->model()->removeRow(ui->tableView->selectionModel()->currentIndex().row());
}

void PathPlannerGadgetWidget::on_tbInsert_clicked()
{
    ui->tableView->model()->insertRow(ui->tableView->selectionModel()->currentIndex().row());
}

void PathPlannerGadgetWidget::on_tbReadFromFile_clicked()
{
    if(!model)
        return;
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"));
    model->readFromFile(fileName);
}

void PathPlannerGadgetWidget::on_tbSaveToFile_clicked()
{
    if(!model)
        return;
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"));
    model->writeToFile(fileName);
}

/**
 * @brief PathPlannerGadgetWidget::on_tbDetails_clicked Display a dialog to show
 * and edit details of a waypoint.  The waypoint selected initially will be the
 * highlighted one.
 */
void PathPlannerGadgetWidget::on_tbDetails_clicked()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm);
    if (pm == NULL)
        return;

    WaypointDialog *dialog =  pm->getObject<WaypointDialog>();
    Q_ASSERT(dialog);
    dialog->show();
}

/**
 * @brief PathPlannerGadgetWidget::on_tbSendToUAV_clicked Use the proxy to send
 * the data from the flight model to the UAV
 */
void PathPlannerGadgetWidget::on_tbSendToUAV_clicked()
{
    enableButtons(false);
    ui->statusTL->setVisible(false);
    ui->statusPB->setValue(0);
    bool result;
    result = proxy->modelToObjects();
    if(result)
        ui->statusTL->setText("All waypoints were sent successfully");
    else
        ui->statusTL->setText("WARNING: Not all waypoints were sent successfully");
    ui->statusTL->setVisible(true);
    enableButtons(true);
}

/**
 * @brief PathPlannerGadgetWidget::on_tbFetchFromUAV_clicked Use the flight model to
 * get data from the UAV
 */
void PathPlannerGadgetWidget::on_tbFetchFromUAV_clicked()
{
    proxy->objectsToModel();
    ui->tableView->resizeColumnsToContents();
}

/**
 * @brief PathPlannerGadgetWidget::on_tbFilletPath_clicked Apply fillets to the current path
 */
void PathPlannerGadgetWidget::on_tbFilletPath_clicked()
{
    // Create a copy of the model before filleting
    if (!prevModel)
        prevModel = new FlightDataModel(this);
    Q_ASSERT(prevModel);
    if (prevModel)
        prevModel->replaceData(model);

    IPathAlgorithm * algo = new PathFillet(this);
    // Only process is successfully configured and the verification of the model succeeds
    QString err;
    if(algo->configure(this) && algo->verifyPath(model, err)) {
        // If unsuccessful delete the cached model
        if (!algo->processPath(model)) {
            delete prevModel;
            prevModel = NULL;
        }
    }
}

/**
 * @brief PathPlannerGadgetWidget::on_tbFilletPath_clicked Apply fillets to the current path
 */
void PathPlannerGadgetWidget::on_tbUnfilletPath_clicked()
{
    if (prevModel)
        model->replaceData(prevModel);
}

void PathPlannerGadgetWidget::on_waypointSendProgress(int value)
{
    ui->statusPB->setValue(value);
}

void PathPlannerGadgetWidget::enableButtons(bool enable)
{
    QList<QToolButton*> blist = this->findChildren<QToolButton*>();
    foreach (QToolButton * button, blist) {
        button->setEnabled(enable);
}
}

/**
  * @}
  * @}
  */
