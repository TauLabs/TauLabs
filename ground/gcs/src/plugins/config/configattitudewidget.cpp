/**
 ******************************************************************************
 *
 * @file       configattitudewidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief The Configuration Gadget used to update settings in the firmware
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
#include "configattitudewidget.h"
#include "physical_constants.h"

#include "math.h"
#include <QDebug>
#include <QTimer>
#include <QStringList>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include <QMessageBox>
#include <QThread>
#include <QErrorMessage>
#include <iostream>
#include <QDesktopServices>
#include <QUrl>
#include <coreplugin/iboardtype.h>
#include <attitudesettings.h>
#include <sensorsettings.h>
#include <inssettings.h>
#include <homelocation.h>
#include <accels.h>
#include <gyros.h>
#include <magnetometer.h>
#include <baroaltitude.h>

#include "assertions.h"
#include "calibration.h"

#define sign(x) ((x < 0) ? -1 : 1)

// Uncomment this to enable 6 point calibration on the accels
#define SIX_POINT_CAL_ACCEL

const double ConfigAttitudeWidget::maxVarValue = 0.1;

// *****************

class Thread : public QThread
{
public:
    static void usleep(unsigned long usecs)
    {
        QThread::usleep(usecs);
    }
};

// *****************

ConfigAttitudeWidget::ConfigAttitudeWidget(QWidget *parent) :
    ConfigTaskWidget(parent),
    m_ui(new Ui_AttitudeWidget()),
    board_has_accelerometer(false),
    board_has_magnetometer(false)
{
    m_ui->setupUi(this);

    // Initialization of the Paper plane widget
    m_ui->sixPointHelp->setScene(new QGraphicsScene(this));

    paperplane = new QGraphicsSvgItem();
    paperplane->setSharedRenderer(new QSvgRenderer());
    paperplane->renderer()->load(QString(":/configgadget/images/paper-plane.svg"));
    paperplane->setElementId("plane-horizontal");
    m_ui->sixPointHelp->scene()->addItem(paperplane);
    m_ui->sixPointHelp->setSceneRect(paperplane->boundingRect());

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectUtilManager* utilMngr = pm->getObject<UAVObjectUtilManager>();
    Q_ASSERT(utilMngr);
    if (utilMngr != NULL) {
        Core::IBoardType *board = utilMngr->getBoardType();
        if (board != NULL) {
            board_has_accelerometer = board->queryCapabilities(Core::IBoardType::BOARD_CAPABILITIES_ACCELS);
            board_has_magnetometer = board->queryCapabilities(Core::IBoardType::BOARD_CAPABILITIES_MAGS);
        }
        else
            qDebug() << "Board not found";
    }

    // Must set up the UI (above) before setting up the UAVO mappings or refreshWidgetValues
    // will be dealing with some null pointers
    addUAVObject("AttitudeSettings");
    addUAVObject("SensorSettings");
    if (board_has_magnetometer) {
        addUAVObject("INSSettings");
    }
    autoLoadWidgets();

    // Configure the calibration object
    calibration.initialize(board_has_accelerometer, board_has_magnetometer);

    // Configure the calibration UI
    m_ui->cbCalibrateAccels->setChecked(board_has_accelerometer);
    m_ui->cbCalibrateMags->setChecked(board_has_magnetometer);
    if (!board_has_accelerometer || !board_has_magnetometer) { // If both are not available, don't provide any choices.
        m_ui->cbCalibrateAccels->setEnabled(false);
        m_ui->cbCalibrateMags->setEnabled(false);
    }

    // Must connect the graphs to the calibration object to see the calibration results
    calibration.configureTempCurves(m_ui->xGyroTemp, m_ui->yGyroTemp, m_ui->zGyroTemp);

    // Connect the signals
    connect(m_ui->yawOrientationStart, SIGNAL(clicked()), &calibration, SLOT(doStartOrientation()));
    connect(m_ui->levelingStart, SIGNAL(clicked()), &calibration, SLOT(doStartNoBiasLeveling()));
    connect(m_ui->levelingAndBiasStart, SIGNAL(clicked()), &calibration, SLOT(doStartBiasAndLeveling()));
    connect(m_ui->sixPointStart, SIGNAL(clicked()), &calibration, SLOT(doStartSixPoint()));
    connect(m_ui->sixPointSave, SIGNAL(clicked()), &calibration, SLOT(doSaveSixPointPosition()));
    connect(m_ui->sixPointCancel, SIGNAL(clicked()), &calibration, SLOT(doCancelSixPoint()));
    connect(m_ui->cbCalibrateAccels, SIGNAL(clicked()), this, SLOT(configureSixPoint()));
    connect(m_ui->cbCalibrateMags, SIGNAL(clicked()), this, SLOT(configureSixPoint()));
    connect(m_ui->startTempCal, SIGNAL(clicked()), &calibration, SLOT(doStartTempCal()));
    connect(m_ui->acceptTempCal, SIGNAL(clicked()), &calibration, SLOT(doAcceptTempCal()));
    connect(m_ui->cancelTempCal, SIGNAL(clicked()), &calibration, SLOT(doCancelTempCalPoint()));
    connect(m_ui->tempCalRange, SIGNAL(valueChanged(int)), &calibration, SLOT(setTempCalRange(int)));
    calibration.setTempCalRange(m_ui->tempCalRange->value());

    // Let calibration update the UI
    connect(&calibration, SIGNAL(yawOrientationProgressChanged(int)), m_ui->pb_yawCalibration, SLOT(setValue(int)));
    connect(&calibration, SIGNAL(levelingProgressChanged(int)), m_ui->accelBiasProgress, SLOT(setValue(int)));
    connect(&calibration, SIGNAL(tempCalProgressChanged(int)), m_ui->tempCalProgress, SLOT(setValue(int)));
    connect(&calibration, SIGNAL(showTempCalMessage(QString)), m_ui->tempCalMessage, SLOT(setText(QString)));
    connect(&calibration, SIGNAL(sixPointProgressChanged(int)), m_ui->sixPointProgress, SLOT(setValue(int)));
    connect(&calibration, SIGNAL(showSixPointMessage(QString)), m_ui->sixPointCalibInstructions, SLOT(setText(QString)));
    connect(&calibration, SIGNAL(updatePlane(int)), this, SLOT(displayPlane(int)));

    // Let the calibration gadget control some control enables
    connect(&calibration, SIGNAL(toggleSavePosition(bool)), m_ui->sixPointSave, SLOT(setEnabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->sixPointStart, SLOT(setEnabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->sixPointCancel, SLOT(setDisabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->yawOrientationStart, SLOT(setEnabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->levelingStart, SLOT(setEnabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->levelingAndBiasStart, SLOT(setEnabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->startTempCal, SLOT(setEnabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->acceptTempCal, SLOT(setDisabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->cancelTempCal, SLOT(setDisabled(bool)));

    // Let the calibration gadget mark the tab as dirty, i.e. having unsaved data.
    connect(&calibration, SIGNAL(calibrationCompleted()), this, SLOT(do_SetDirty()));

    m_ui->sixPointStart->setEnabled(true);
    m_ui->yawOrientationStart->setEnabled(true);
    m_ui->levelingStart->setEnabled(true);
    m_ui->levelingAndBiasStart->setEnabled(true);

    refreshWidgetsValues();
}

ConfigAttitudeWidget::~ConfigAttitudeWidget()
{
    // Do nothing
}


void ConfigAttitudeWidget::showEvent(QShowEvent *event)
{
    Q_UNUSED(event)
    m_ui->sixPointHelp->fitInView(paperplane,Qt::KeepAspectRatio);
}

void ConfigAttitudeWidget::resizeEvent(QResizeEvent *event)
{
    Q_UNUSED(event)
    m_ui->sixPointHelp->fitInView(paperplane,Qt::KeepAspectRatio);
}

/**
  Rotate the paper plane
  */
void ConfigAttitudeWidget::displayPlane(int position)
{
    QString displayElement;
    switch(position) {
    case 1:
        displayElement = "plane-horizontal";
        break;
    case 2:
        displayElement = "plane-left";
        break;
    case 3:
        displayElement = "plane-flip";
        break;
    case 4:
        displayElement = "plane-right";
        break;
    case 5:
        displayElement = "plane-up";
        break;
    case 6:
        displayElement = "plane-down";
        break;
    default:
        return;
    }

    paperplane->setElementId(displayElement);
    m_ui->sixPointHelp->setSceneRect(paperplane->boundingRect());
    m_ui->sixPointHelp->fitInView(paperplane,Qt::KeepAspectRatio);
}

/********** UI Functions *************/

/**
  * Called by the ConfigTaskWidget parent when variances are updated
  * to update the UI
  */
void ConfigAttitudeWidget::refreshWidgetsValues(UAVObject *)
{
    ConfigTaskWidget::refreshWidgetsValues();
}

/**
 * @brief ConfigAttitudeWidget::setUpdated Slot that receives signals indicating the UI is updated
 */
void ConfigAttitudeWidget::do_SetDirty()
{
    setDirty(true);
}


void ConfigAttitudeWidget::configureSixPoint()
{
    if (!m_ui->cbCalibrateAccels->isChecked() && !m_ui->cbCalibrateMags->isChecked()) {
        QMessageBox::information(this, "No sensors chosen", "At least one of the sensors must be chosen. \n\nResetting six-point sensor calibration selection.");
        m_ui->cbCalibrateAccels->setChecked(true && board_has_accelerometer);
        m_ui->cbCalibrateMags->setChecked(true && board_has_magnetometer);
    }
    calibration.initialize(m_ui->cbCalibrateAccels->isChecked(), m_ui->cbCalibrateMags->isChecked());
}


/**
  @}
  @}
  */
