/**
 ******************************************************************************
 * @file       waypointdialog.cpp
 * @author     PhoenixPilot Project, http://github.com/PhoenixPilot Copyright (C) 2012.
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup PhoenixPilot GCS Plugins
 * @{
 * @addtogroup PathPlanner Map Plugin
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

#include <QDebug>
#include "waypointdialog.h"
#include "ui_waypoint_dialog.h"

WaypointDialog::WaypointDialog(QWidget *parent,QAbstractItemModel * model,QItemSelectionModel * selection) :
    QWidget(parent,Qt::Window), model(model), itemSelection(selection),
    ui(new Ui_waypoint_dialog)
{
    ui->setupUi(this);
    connect(ui->checkBoxLocked,SIGNAL(toggled(bool)),this,SLOT(enableEditWidgets(bool)));
    connect(ui->cbMode,SIGNAL(currentIndexChanged(int)),this,SLOT(setupModeWidgets()));

    // Connect up the buttons
    connect(ui->pushButtonOK, SIGNAL(clicked()), this, SLOT(on_okButton_clicked()));
    connect(ui->pushButtonCancel, SIGNAL(clicked()), this, SLOT(on_cancelButton_clicked()));
    connect(ui->pushButtonPrevious, SIGNAL(clicked()), this, SLOT(on_previousButton_clicked()));
    connect(ui->pushButtonNext, SIGNAL(clicked()), this, SLOT(on_nextButton_clicked()));

    WaypointDataDelegate::loadComboBox(ui->cbMode,FlightDataModel::MODE);

    mapper = new QDataWidgetMapper(this);

    mapper->setItemDelegate(new WaypointDataDelegate(this));
    connect (mapper,SIGNAL(currentIndexChanged(int)),this,SLOT(currentIndexChanged(int)));
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

    mapper->setCurrentIndex(selection->currentIndex().row());

    connect(itemSelection,SIGNAL(currentRowChanged(QModelIndex,QModelIndex)),this,SLOT(currentRowChanged(QModelIndex,QModelIndex)));
}

void WaypointDialog::currentIndexChanged(int index)
{
    ui->lbNumber->setText(QString::number(index+1));
    QModelIndex idx=mapper->model()->index(index,0);
    if(index==itemSelection->currentIndex().row())
        return;
    itemSelection->clear();
    itemSelection->setCurrentIndex(idx,QItemSelectionModel::Select | QItemSelectionModel::Rows);
}

WaypointDialog::~WaypointDialog()
{
    delete ui;
}

void WaypointDialog::setupModeWidgets()
{
    WaypointDataDelegate::ModeOptions mode = (WaypointDataDelegate::ModeOptions)
                   ui->cbMode->itemData(ui->cbMode->currentIndex()).toInt();
    switch(mode)
    {
    case WaypointDataDelegate::MODE_FLYENDPOINT:
    case WaypointDataDelegate::MODE_FLYVECTOR:
    case WaypointDataDelegate::MODE_FLYCIRCLERIGHT:
    case WaypointDataDelegate::MODE_FLYCIRCLELEFT:
    case WaypointDataDelegate::MODE_DRIVEENDPOINT:
    case WaypointDataDelegate::MODE_DRIVEVECTOR:
    case WaypointDataDelegate::MODE_DRIVECIRCLELEFT:
    case WaypointDataDelegate::MODE_DRIVECIRCLERIGHT:
        ui->modeParams->setVisible(false);
        break;
    }
}

/**
 * @brief WaypointDialog::editWaypoint Edit the requested waypoint, show dialog if it is not showing
 * @param[in] number The waypoint to edit
 */
void WaypointDialog::editWaypoint(int number)
{
    if(!isVisible())
        show();
    if(isMinimized())
        showNormal();
    if(!isActiveWindow())
        activateWindow();
    raise();
    setFocus(Qt::OtherFocusReason);
    mapper->setCurrentIndex(number);
}

//! Close the dialog button, accept the changes
void WaypointDialog::on_okButton_clicked()
{
    mapper->submit();
    close();
}

//! Close the dialog button, revert any changes
void WaypointDialog::on_cancelButton_clicked()
{
    mapper->revert();
    close();
}

//! User requests the previous waypoint
void WaypointDialog::on_previousButton_clicked()
{
    mapper->toPrevious();
}

//! User requests the next waypoint
void WaypointDialog::on_nextButton_clicked()
{
    mapper->toNext();
}

/**
 * @brief WaypointDialog::enableEditWidgets Enable or disable the controls
 * @param[in] value True if they should be enabled, false to disable
 */
void WaypointDialog::enableEditWidgets(bool value)
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

void WaypointDialog::currentRowChanged(QModelIndex current, QModelIndex previous)
{
    Q_UNUSED(previous);

    mapper->setCurrentIndex(current.row());
}

/**
 * @brief WaypointDataDelegate::createEditor Returns the widget used to edit the item
 * specified by index for editing. The parent widget and style option are used to
 * control how the editor widget appears.
 * @return The widget for the index
 */
QWidget *WaypointDataDelegate::createEditor(QWidget *parent,
                                        const QStyleOptionViewItem & option,
                                        const QModelIndex & index) const
{
    int column=index.column();
    QComboBox * box;
    switch(column)
    {
    case FlightDataModel::MODE:
        box=new QComboBox(parent);
        WaypointDataDelegate::loadComboBox(box, FlightDataModel::MODE);
        return box;
        break;
    default:
        return QItemDelegate::createEditor(parent,option,index);
        break;
    }

    QComboBox *editor = new QComboBox(parent);
    return editor;
}

/**
 * @brief WaypointDataDelegate::setEditorData Set the data in the UI from the model
 * for a particular element index
 * @param editor The editor dialog
 * @param index The model parameter index to use
 */
void WaypointDataDelegate::setEditorData(QWidget *editor,
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

/**
 * @brief WaypointDataDelegate::setModelData Update the model from the UI for a particular
 * element index
 * @param editor
 * @param model The editor dialog
 * @param index The model parameter index to change
 */
void WaypointDataDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
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

void WaypointDataDelegate::updateEditorGeometry(QWidget *editor,
                                            const QStyleOptionViewItem &option, const QModelIndex &/* index */) const
{
    editor->setGeometry(option.rect);
}

void WaypointDataDelegate::loadComboBox(QComboBox *combo, FlightDataModel::pathPlanDataEnum type)
{
    switch(type)
    {
    case FlightDataModel::MODE:
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

WaypointDataDelegate::WaypointDataDelegate(QObject *parent):QItemDelegate(parent)
{
}
