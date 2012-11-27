/**
 ******************************************************************************
 *
 * @file       opmap_edit_waypoint_dialog.cpp
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

#include "opmap_edit_waypoint_dialog.h"
#include "ui_opmap_edit_waypoint_dialog.h"
#include "opmapcontrol/opmapcontrol.h"
#include "widgetdelegates.h"
// *********************************************************************

// constructor
opmap_edit_waypoint_dialog::opmap_edit_waypoint_dialog(QWidget *parent,QAbstractItemModel * model,QItemSelectionModel * selection) :
    QWidget(parent,Qt::Window),model(model),itemSelection(selection),
    ui(new Ui::opmap_edit_waypoint_dialog)
{  
    ui->setupUi(this);
    connect(ui->checkBoxLocked,SIGNAL(toggled(bool)),this,SLOT(enableEditWidgets(bool)));
    connect(ui->cbMode,SIGNAL(currentIndexChanged(int)),this,SLOT(setupModeWidgets()));
    connect(ui->pushButtonCancel,SIGNAL(clicked()),this,SLOT(pushButtonCancel_clicked()));
    MapDataDelegate::loadComboBox(ui->cbMode,FlightDataModel::MODE);

    mapper = new QDataWidgetMapper(this);

    mapper->setItemDelegate(new MapDataDelegate(this));
    connect(mapper,SIGNAL(currentIndexChanged(int)),this,SLOT(currentIndexChanged(int)));
    mapper->setModel(model);
    mapper->setSubmitPolicy(QDataWidgetMapper::AutoSubmit);
    mapper->addMapping(ui->checkBoxLocked,FlightDataModel::LOCKED);
    mapper->addMapping(ui->doubleSpinBoxLatitude,FlightDataModel::LATPOSITION);
    mapper->addMapping(ui->doubleSpinBoxLongitude,FlightDataModel::LNGPOSITION);
    mapper->addMapping(ui->doubleSpinBoxAltitude,FlightDataModel::ALTITUDE);
    mapper->addMapping(ui->lineEditDescription,FlightDataModel::WPDESCRITPTION);
    mapper->addMapping(ui->checkBoxRelative,FlightDataModel::ISRELATIVE);
    mapper->addMapping(ui->doubleSpinBoxBearing,FlightDataModel::BEARELATIVE);
    mapper->addMapping(ui->doubleSpinBoxVelocity,FlightDataModel::VELOCITY);
    mapper->addMapping(ui->doubleSpinBoxDistance,FlightDataModel::DISRELATIVE);
    mapper->addMapping(ui->doubleSpinBoxRelativeAltitude,FlightDataModel::ALTITUDERELATIVE);
    mapper->addMapping(ui->cbMode,FlightDataModel::MODE);
    mapper->addMapping(ui->dsb_modeParams,FlightDataModel::MODE_PARAMS);

    connect(itemSelection,SIGNAL(currentRowChanged(QModelIndex,QModelIndex)),this,SLOT(currentRowChanged(QModelIndex,QModelIndex)));
}
void opmap_edit_waypoint_dialog::currentIndexChanged(int index)
{
    ui->lbNumber->setText(QString::number(index+1));
    QModelIndex idx=mapper->model()->index(index,0);
    if(index==itemSelection->currentIndex().row())
        return;
    itemSelection->clear();
    itemSelection->setCurrentIndex(idx,QItemSelectionModel::Select | QItemSelectionModel::Rows);
}

opmap_edit_waypoint_dialog::~opmap_edit_waypoint_dialog()
{
    delete ui;
}

void opmap_edit_waypoint_dialog::on_pushButtonOK_clicked()
{
    mapper->submit();
    close();
}

void opmap_edit_waypoint_dialog::setupModeWidgets()
{
    MapDataDelegate::ModeOptions mode=(MapDataDelegate::ModeOptions)ui->cbMode->itemData(ui->cbMode->currentIndex()).toInt();
    switch(mode)
    {
    case MapDataDelegate::MODE_FLYENDPOINT:
    case MapDataDelegate::MODE_FLYVECTOR:
    case MapDataDelegate::MODE_FLYCIRCLERIGHT:
    case MapDataDelegate::MODE_FLYCIRCLELEFT:
    case MapDataDelegate::MODE_DRIVEENDPOINT:
    case MapDataDelegate::MODE_DRIVEVECTOR:
    case MapDataDelegate::MODE_DRIVECIRCLELEFT:
    case MapDataDelegate::MODE_DRIVECIRCLERIGHT:
        ui->modeParams->setVisible(false);
        break;
    }
}

void opmap_edit_waypoint_dialog::pushButtonCancel_clicked()
{
    mapper->revert();
    close();
}
void opmap_edit_waypoint_dialog::editWaypoint(mapcontrol::WayPointItem *waypoint_item)
{
    if (!waypoint_item) return;
    if(!isVisible())
        show();
    if(isMinimized())
        showNormal();
    if(!isActiveWindow())
        activateWindow();
    raise();
    setFocus(Qt::OtherFocusReason);
    mapper->setCurrentIndex(waypoint_item->Number());
}

void opmap_edit_waypoint_dialog::on_pushButton_clicked()
{
    mapper->toPrevious();
}

void opmap_edit_waypoint_dialog::on_pushButton_2_clicked()
{
    mapper->toNext();
}

void opmap_edit_waypoint_dialog::enableEditWidgets(bool value)
{
    QWidget * w;
    foreach(QWidget * obj,this->findChildren<QWidget *>())
    {
        w=qobject_cast<QComboBox*>(obj);
        if(w)
            w->setEnabled(!value);
        w=qobject_cast<QLineEdit*>(obj);
        if(w)
            w->setEnabled(!value);
        w=qobject_cast<QDoubleSpinBox*>(obj);
        if(w)
            w->setEnabled(!value);
        w=qobject_cast<QCheckBox*>(obj);
        if(w && w!=ui->checkBoxLocked)
            w->setEnabled(!value);
        w=qobject_cast<QSpinBox*>(obj);
        if(w)
            w->setEnabled(!value);
    }
}

void opmap_edit_waypoint_dialog::currentRowChanged(QModelIndex current, QModelIndex previous)
{
    Q_UNUSED(previous);

    mapper->setCurrentIndex(current.row());
}
