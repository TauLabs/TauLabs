/**
 ******************************************************************************
 *
 * @file       ConfigRevoWidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
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
#include "configrevowidget.h"

#include "math.h"
#include <QDebug>
#include <QTimer>
#include <QStringList>
#include <QtGui/QWidget>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>
#include <QtGui/QPushButton>
#include <QMessageBox>
#include <QThread>
#include <QErrorMessage>
#include <iostream>
#include <QDesktopServices>
#include <QUrl>
#include <attitudesettings.h>
#include <inertialsensorsettings.h>
#include <revocalibration.h>
#include <inssettings.h>
#include <homelocation.h>
#include <accels.h>
#include <gyros.h>
#include <magnetometer.h>
#include <baroaltitude.h>

#define GRAVITY 9.81f
#include "assertions.h"
#include "calibration.h"

#define sign(x) ((x < 0) ? -1 : 1)

// Uncomment this to enable 6 point calibration on the accels
#define SIX_POINT_CAL_ACCEL

const double ConfigRevoWidget::maxVarValue = 0.1;

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

ConfigRevoWidget::ConfigRevoWidget(QWidget *parent) :
    ConfigTaskWidget(parent),
    m_ui(new Ui_RevoSensorsWidget())
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

    // Initialization of the Revo sensor noise bargraph graph
    m_ui->sensorsBargraph->setScene(new QGraphicsScene(this));

    QSvgRenderer *renderer = new QSvgRenderer();
    sensorsBargraph = new QGraphicsSvgItem();
    renderer->load(QString(":/configgadget/images/ahrs-calib.svg"));
    sensorsBargraph->setSharedRenderer(renderer);
    sensorsBargraph->setElementId("background");
    sensorsBargraph->setObjectName("background");
    m_ui->sensorsBargraph->scene()->addItem(sensorsBargraph);
    m_ui->sensorsBargraph->setSceneRect(sensorsBargraph->boundingRect());

    // Initialize the 9 bargraph values:

    QMatrix lineMatrix = renderer->matrixForElement("accel_x");
    QRectF rect = lineMatrix.mapRect(renderer->boundsOnElement("accel_x"));
    qreal startX = rect.x();
    qreal startY = rect.y()+ rect.height();
    // maxBarHeight will be used for scaling it later.
    maxBarHeight = rect.height();
    // Then once we have the initial location, we can put it
    // into a QGraphicsSvgItem which we will display at the same
    // place: we do this so that the heading scale can be clipped to
    // the compass dial region.
    accel_x = new QGraphicsSvgItem();
    accel_x->setSharedRenderer(renderer);
    accel_x->setElementId("accel_x");
    m_ui->sensorsBargraph->scene()->addItem(accel_x);
    accel_x->setPos(startX, startY);
    accel_x->setTransform(QTransform::fromScale(1,0),true);

    lineMatrix = renderer->matrixForElement("accel_y");
    rect = lineMatrix.mapRect(renderer->boundsOnElement("accel_y"));
    startX = rect.x();
    startY = rect.y()+ rect.height();
    accel_y = new QGraphicsSvgItem();
    accel_y->setSharedRenderer(renderer);
    accel_y->setElementId("accel_y");
    m_ui->sensorsBargraph->scene()->addItem(accel_y);
    accel_y->setPos(startX,startY);
    accel_y->setTransform(QTransform::fromScale(1,0),true);

    lineMatrix = renderer->matrixForElement("accel_z");
    rect = lineMatrix.mapRect(renderer->boundsOnElement("accel_z"));
    startX = rect.x();
    startY = rect.y()+ rect.height();
    accel_z = new QGraphicsSvgItem();
    accel_z->setSharedRenderer(renderer);
    accel_z->setElementId("accel_z");
    m_ui->sensorsBargraph->scene()->addItem(accel_z);
    accel_z->setPos(startX,startY);
    accel_z->setTransform(QTransform::fromScale(1,0),true);

    lineMatrix = renderer->matrixForElement("gyro_x");
    rect = lineMatrix.mapRect(renderer->boundsOnElement("gyro_x"));
    startX = rect.x();
    startY = rect.y()+ rect.height();
    gyro_x = new QGraphicsSvgItem();
    gyro_x->setSharedRenderer(renderer);
    gyro_x->setElementId("gyro_x");
    m_ui->sensorsBargraph->scene()->addItem(gyro_x);
    gyro_x->setPos(startX,startY);
    gyro_x->setTransform(QTransform::fromScale(1,0),true);

    lineMatrix = renderer->matrixForElement("gyro_y");
    rect = lineMatrix.mapRect(renderer->boundsOnElement("gyro_y"));
    startX = rect.x();
    startY = rect.y()+ rect.height();
    gyro_y = new QGraphicsSvgItem();
    gyro_y->setSharedRenderer(renderer);
    gyro_y->setElementId("gyro_y");
    m_ui->sensorsBargraph->scene()->addItem(gyro_y);
    gyro_y->setPos(startX,startY);
    gyro_y->setTransform(QTransform::fromScale(1,0),true);


    lineMatrix = renderer->matrixForElement("gyro_z");
    rect = lineMatrix.mapRect(renderer->boundsOnElement("gyro_z"));
    startX = rect.x();
    startY = rect.y()+ rect.height();
    gyro_z = new QGraphicsSvgItem();
    gyro_z->setSharedRenderer(renderer);
    gyro_z->setElementId("gyro_z");
    m_ui->sensorsBargraph->scene()->addItem(gyro_z);
    gyro_z->setPos(startX,startY);
    gyro_z->setTransform(QTransform::fromScale(1,0),true);

    lineMatrix = renderer->matrixForElement("mag_x");
    rect = lineMatrix.mapRect(renderer->boundsOnElement("mag_x"));
    startX = rect.x();
    startY = rect.y()+ rect.height();
    mag_x = new QGraphicsSvgItem();
    mag_x->setSharedRenderer(renderer);
    mag_x->setElementId("mag_x");
    m_ui->sensorsBargraph->scene()->addItem(mag_x);
    mag_x->setPos(startX,startY);
    mag_x->setTransform(QTransform::fromScale(1,0),true);

    lineMatrix = renderer->matrixForElement("mag_y");
    rect = lineMatrix.mapRect(renderer->boundsOnElement("mag_y"));
    startX = rect.x();
    startY = rect.y()+ rect.height();
    mag_y = new QGraphicsSvgItem();
    mag_y->setSharedRenderer(renderer);
    mag_y->setElementId("mag_y");
    m_ui->sensorsBargraph->scene()->addItem(mag_y);
    mag_y->setPos(startX,startY);
    mag_y->setTransform(QTransform::fromScale(1,0),true);

    lineMatrix = renderer->matrixForElement("mag_z");
    rect = lineMatrix.mapRect(renderer->boundsOnElement("mag_z"));
    startX = rect.x();
    startY = rect.y()+ rect.height();
    mag_z = new QGraphicsSvgItem();
    mag_z->setSharedRenderer(renderer);
    mag_z->setElementId("mag_z");
    m_ui->sensorsBargraph->scene()->addItem(mag_z);
    mag_z->setPos(startX,startY);
    mag_z->setTransform(QTransform::fromScale(1,0),true);

    lineMatrix = renderer->matrixForElement("baro");
    rect = lineMatrix.mapRect(renderer->boundsOnElement("baro"));
    startX = rect.x();
    startY = rect.y()+ rect.height();
    baro = new QGraphicsSvgItem();
    baro->setSharedRenderer(renderer);
    baro->setElementId("baro");
    m_ui->sensorsBargraph->scene()->addItem(baro);
    baro->setPos(startX,startY);
    baro->setTransform(QTransform::fromScale(1,0),true);

    // Must set up the UI (above) before setting up the UAVO mappings or refreshWidgetValues
    // will be dealing with some null pointers
    addUAVObject("RevoCalibration");
    addUAVObject("INSSettings");
    addUAVObject("AttitudeSettings");
    autoLoadWidgets();

    // Configure the calibration object
    calibration.initialize(true);

    // Connect the signals
    connect(m_ui->accelBiasStart, SIGNAL(clicked()), &calibration, SLOT(doStartLeveling()));
    connect(m_ui->sixPointStart, SIGNAL(clicked()), &calibration ,SLOT(doStartSixPoint()));
    connect(m_ui->sixPointSave, SIGNAL(clicked()), &calibration ,SLOT(doSaveSixPointPosition()));
    connect(m_ui->sixPointCancel, SIGNAL(clicked()), &calibration ,SLOT(doCancelSixPoint()));

    // Let calibration update the UI
    connect(&calibration, SIGNAL(levelingProgressChanged(int)), m_ui->accelBiasProgress, SLOT(setValue(int)));
    connect(&calibration, SIGNAL(sixPointProgressChanged(int)), m_ui->sixPointProgress, SLOT(setValue(int)));
    connect(&calibration, SIGNAL(showSixPointMessage(QString)), m_ui->sixPointCalibInstructions, SLOT(setText(QString)));
    connect(&calibration, SIGNAL(updatePlane(int)), this, SLOT(displayPlane(int)));

    // Let the calibration gadget control some control enables
    connect(&calibration, SIGNAL(toggleSavePosition(bool)), m_ui->sixPointSave, SLOT(setEnabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->sixPointStart, SLOT(setEnabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->sixPointCancel, SLOT(setDisabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->noiseMeasurementStart, SLOT(setEnabled(bool)));
    connect(&calibration, SIGNAL(toggleControls(bool)), m_ui->accelBiasStart, SLOT(setEnabled(bool)));

    m_ui->noiseMeasurementStart->setEnabled(true);
    m_ui->sixPointStart->setEnabled(true);
    m_ui->accelBiasStart->setEnabled(true);

    // Currently not in the calibration object
    connect(m_ui->noiseMeasurementStart, SIGNAL(clicked()), this, SLOT(doStartNoiseMeasurement()));

    refreshWidgetsValues();
}

ConfigRevoWidget::~ConfigRevoWidget()
{
    // Do nothing
}


void ConfigRevoWidget::showEvent(QShowEvent *event)
{
    Q_UNUSED(event)
    // Thit fitInView method should only be called now, once the
    // widget is shown, otherwise it cannot compute its values and
    // the result is usually a sensorsBargraph that is way too small.
    m_ui->sensorsBargraph->fitInView(sensorsBargraph, Qt::KeepAspectRatio);
    m_ui->sixPointHelp->fitInView(paperplane,Qt::KeepAspectRatio);
}

void ConfigRevoWidget::resizeEvent(QResizeEvent *event)
{
    Q_UNUSED(event)
    m_ui->sensorsBargraph->fitInView(sensorsBargraph, Qt::KeepAspectRatio);
    m_ui->sixPointHelp->fitInView(paperplane,Qt::KeepAspectRatio);
}

/**
  Rotate the paper plane
  */
void ConfigRevoWidget::displayPlane(int position)
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

/*********** Noise measurement functions **************/
/**
  * Connect sensor updates and timeout for measuring the noise
  */
void ConfigRevoWidget::doStartNoiseMeasurement()
{
    QMutexLocker lock(&sensorsUpdateLock);
    Q_UNUSED(lock);

    accel_accum_x.clear();
    accel_accum_y.clear();
    accel_accum_z.clear();
    gyro_accum_x.clear();
    gyro_accum_y.clear();
    gyro_accum_z.clear();
    mag_accum_x.clear();
    mag_accum_y.clear();
    mag_accum_z.clear();
    baro_accum.clear();

    /* Need to get as many accel, mag and gyro updates as possible */
    Accels * accels = Accels::GetInstance(getObjectManager());
    Q_ASSERT(accels);
    Gyros * gyros = Gyros::GetInstance(getObjectManager());
    Q_ASSERT(gyros);
    Magnetometer * mag = Magnetometer::GetInstance(getObjectManager());
    Q_ASSERT(mag);
    BaroAltitude * baro = BaroAltitude::GetInstance(getObjectManager());
    Q_ASSERT(baro);

    UAVObject::Metadata mdata;

    initialAccelsMdata = accels->getMetadata();
    mdata = initialAccelsMdata;
    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
    mdata.flightTelemetryUpdatePeriod = 100;
    accels->setMetadata(mdata);

    initialGyrosMdata = gyros->getMetadata();
    mdata = initialGyrosMdata;
    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
    mdata.flightTelemetryUpdatePeriod = 100;
    gyros->setMetadata(mdata);

    initialMagMdata = mag->getMetadata();
    mdata = initialMagMdata;
    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
    mdata.flightTelemetryUpdatePeriod = 100;
    mag->setMetadata(mdata);

    initialBaroMdata = baro->getMetadata();
    mdata = initialBaroMdata;
    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
    mdata.flightTelemetryUpdatePeriod = 100;
    baro->setMetadata(mdata);

    /* Connect for updates */
    connect(accels, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(doGetNoiseSample(UAVObject*)));
    connect(gyros, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(doGetNoiseSample(UAVObject*)));
    connect(mag, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(doGetNoiseSample(UAVObject*)));
    connect(baro, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(doGetNoiseSample(UAVObject*)));
}

/**
  * Called when any of the sensors are updated.  Stores the sample for measuring the
  * variance at the end
  */
void ConfigRevoWidget::doGetNoiseSample(UAVObject * obj)
{
    QMutexLocker lock(&sensorsUpdateLock);
    Q_UNUSED(lock);

    Q_ASSERT(obj);

    switch(obj->getObjID()) {
    case Gyros::OBJID:
    {
        Gyros * gyros = Gyros::GetInstance(getObjectManager());
        Q_ASSERT(gyros);
        Gyros::DataFields gyroData = gyros->getData();
        gyro_accum_x.append(gyroData.x);
        gyro_accum_y.append(gyroData.y);
        gyro_accum_z.append(gyroData.z);
        break;
    }
    case Accels::OBJID:
    {
        Accels * accels = Accels::GetInstance(getObjectManager());
        Q_ASSERT(accels);
        Accels::DataFields accelsData = accels->getData();
        accel_accum_x.append(accelsData.x);
        accel_accum_y.append(accelsData.y);
        accel_accum_z.append(accelsData.z);
        break;
    }
    case Magnetometer::OBJID:
    {
        Magnetometer * mags = Magnetometer::GetInstance(getObjectManager());
        Q_ASSERT(mags);
        Magnetometer::DataFields magData = mags->getData();
        mag_accum_x.append(magData.x);
        mag_accum_y.append(magData.y);
        mag_accum_z.append(magData.z);
        break;
    }
    case BaroAltitude::OBJID:
    {
        BaroAltitude * baro = BaroAltitude::GetInstance(getObjectManager());
        Q_ASSERT(baro);
        BaroAltitude::DataFields baroData = baro->getData();
        baro_accum.append(baroData.Altitude);
        break;
    }
    default:
        Q_ASSERT(0);
    }

    //Calculate progress as the minimum number of samples from any given sensor
    float p1 = (float) mag_accum_x.length() / (float) NOISE_SAMPLES;
    float p2 = (float) gyro_accum_x.length() / (float) NOISE_SAMPLES;
    float p3 = (float) accel_accum_x.length() / (float) NOISE_SAMPLES;
    float p4 = (float) baro_accum.length() / (float) NOISE_SAMPLES;

    float prog = (p1 < p2) ? p1 : p2;
    prog = (prog < p3) ? prog : p3;
    prog = (prog < p4) ? prog : p4;

    m_ui->noiseMeasurementProgress->setValue(prog * 100);

    if(mag_accum_x.length() >= NOISE_SAMPLES &&
            gyro_accum_x.length() >= NOISE_SAMPLES &&
            accel_accum_x.length() >= NOISE_SAMPLES &&
            baro_accum.length() >= NOISE_SAMPLES) {

        // No need to for more updates
        Magnetometer * mags = Magnetometer::GetInstance(getObjectManager());
        Accels * accels = Accels::GetInstance(getObjectManager());
        Gyros * gyros = Gyros::GetInstance(getObjectManager());
        BaroAltitude * baro = BaroAltitude::GetInstance(getObjectManager());
        disconnect(accels, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(doGetNoiseSample(UAVObject*)));
        disconnect(gyros, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(doGetNoiseSample(UAVObject*)));
        disconnect(mags, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(doGetNoiseSample(UAVObject*)));
        disconnect(baro, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(doGetNoiseSample(UAVObject*)));

        //Store the variance
        INSSettings *insSettings = INSSettings::GetInstance(getObjectManager());
        Q_ASSERT(insSettings);
        if(insSettings) {
            INSSettings::DataFields insSettingsData = insSettings->getData();
            insSettingsData.accel_var[INSSettings::ACCEL_VAR_X] = listVar(accel_accum_x);
            insSettingsData.accel_var[INSSettings::ACCEL_VAR_Y] = listVar(accel_accum_y);
            insSettingsData.accel_var[INSSettings::ACCEL_VAR_Z] = listVar(accel_accum_z);
            insSettingsData.gyro_var[INSSettings::GYRO_VAR_X] = listVar(gyro_accum_x);
            insSettingsData.gyro_var[INSSettings::GYRO_VAR_Y] = listVar(gyro_accum_y);
            insSettingsData.gyro_var[INSSettings::GYRO_VAR_Z] = listVar(gyro_accum_z);
            insSettingsData.mag_var[INSSettings::MAG_VAR_X] = listVar(mag_accum_x);
            insSettingsData.mag_var[INSSettings::MAG_VAR_Y] = listVar(mag_accum_y);
            insSettingsData.mag_var[INSSettings::MAG_VAR_Z] = listVar(mag_accum_z);
            insSettingsData.baro_var = listVar(baro_accum);
            insSettings->setData(insSettingsData);
        }
    }
}

/********** UI Functions *************/
/**
  Draws the sensor variances bargraph
  */
void ConfigRevoWidget::drawVariancesGraph()
{
    INSSettings * insSettings = INSSettings::GetInstance(getObjectManager());
    Q_ASSERT(insSettings);
    if(!insSettings)
        return;
    INSSettings::DataFields insSettingsData = insSettings->getData();

    // The expected range is from 1E-6 to 1E-1
    double steps = 6; // 6 bars on the graph
    float accel_x_var = -1/steps*(1+steps+log10(insSettingsData.accel_var[INSSettings::ACCEL_VAR_X]));
    if(accel_x)
        accel_x->setTransform(QTransform::fromScale(1,accel_x_var),false);
    float accel_y_var = -1/steps*(1+steps+log10(insSettingsData.accel_var[INSSettings::ACCEL_VAR_Y]));
    if(accel_y)
        accel_y->setTransform(QTransform::fromScale(1,accel_y_var),false);
    float accel_z_var = -1/steps*(1+steps+log10(insSettingsData.accel_var[INSSettings::ACCEL_VAR_Z]));
    if(accel_z)
        accel_z->setTransform(QTransform::fromScale(1,accel_z_var),false);

    float gyro_x_var = -1/steps*(1+steps+log10(insSettingsData.gyro_var[INSSettings::GYRO_VAR_X]));
    if(gyro_x)
        gyro_x->setTransform(QTransform::fromScale(1,gyro_x_var),false);
    float gyro_y_var = -1/steps*(1+steps+log10(insSettingsData.gyro_var[INSSettings::GYRO_VAR_Y]));
    if(gyro_y)
        gyro_y->setTransform(QTransform::fromScale(1,gyro_y_var),false);
    float gyro_z_var = -1/steps*(1+steps+log10(insSettingsData.gyro_var[INSSettings::GYRO_VAR_Z]));
    if(gyro_z)
        gyro_z->setTransform(QTransform::fromScale(1,gyro_z_var),false);

    // Scale by 1e-3 because mag vars are much higher.
    float mag_x_var = -1/steps*(1+steps+log10(1e-3*insSettingsData.mag_var[INSSettings::MAG_VAR_X]));
    if(mag_x)
        mag_x->setTransform(QTransform::fromScale(1,mag_x_var),false);
    float mag_y_var = -1/steps*(1+steps+log10(1e-3*insSettingsData.mag_var[INSSettings::MAG_VAR_Y]));
    if(mag_y)
        mag_y->setTransform(QTransform::fromScale(1,mag_y_var),false);
    float mag_z_var = -1/steps*(1+steps+log10(1e-3*insSettingsData.mag_var[INSSettings::MAG_VAR_Z]));
    if(mag_z)
        mag_z->setTransform(QTransform::fromScale(1,mag_z_var),false);

    float baro_var = -1/steps*(1+steps+log10(insSettingsData.baro_var));
    if(baro)
        baro->setTransform(QTransform::fromScale(1,baro_var),false);

}

/**
  * Called by the ConfigTaskWidget parent when RevoCalibration is updated
  * to update the UI
  */
void ConfigRevoWidget::refreshWidgetsValues(UAVObject *)
{
    drawVariancesGraph();

    ConfigTaskWidget::refreshWidgetsValues();
}

/**
  @}
  @}
  */
