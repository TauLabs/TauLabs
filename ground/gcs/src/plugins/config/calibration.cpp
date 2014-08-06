/**
 ******************************************************************************
 * @file       calibration.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Gui-less support class for calibration
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
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

#include "calibration.h"

#include "physical_constants.h"

#include "utils/coordinateconversions.h"
#include <QMessageBox>
#include <QDebug>
#include <QThread>

#include "accels.h"
#include "attitudesettings.h"
#include "gyros.h"
#include "homelocation.h"
#include "magnetometer.h"
#include "sensorsettings.h"
#include "trimanglessettings.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <cstdlib>

class Thread : public QThread
{
public:
    static void usleep(unsigned long usecs)
    {
        QThread::usleep(usecs);
    }
};

enum calibrationSuccessMessages{
    CALIBRATION_SUCCESS,
    ACCELEROMETER_FAILED,
    MAGNETOMETER_FAILED
};

#define sign(x) ((x < 0) ? -1 : 1)

Calibration::Calibration() : calibrateMags(false), accelLength(GRAVITY),
    xCurve(NULL), yCurve(NULL), zCurve(NULL)
{
}

Calibration::~Calibration()
{
}

/**
 * @brief Calibration::initialize Configure whether to calibrate the magnetometer
 * and/or accelerometer during 6-point calibration
 * @param calibrateMags
 */
void Calibration::initialize(bool calibrateAccels, bool calibrateMags) {
    this->calibrateAccels = calibrateAccels;
    this->calibrateMags = calibrateMags;
}

/**
 * @brief Calibration::connectSensor
 * @param sensor The sensor to change
 * @param con Whether to connect or disconnect to this sensor
 */
void Calibration::connectSensor(sensor_type sensor, bool con)
{
    if (con) {
        switch (sensor) {
        case ACCEL:
        {
            Accels * accels = Accels::GetInstance(getObjectManager());
            Q_ASSERT(accels);

            assignUpdateRate(accels, SENSOR_UPDATE_PERIOD);
            connect(accels, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        case MAG:
        {
            Magnetometer * mag = Magnetometer::GetInstance(getObjectManager());
            Q_ASSERT(mag);

            assignUpdateRate(mag, SENSOR_UPDATE_PERIOD);
            connect(mag, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        case GYRO:
        {
            Gyros * gyros = Gyros::GetInstance(getObjectManager());
            Q_ASSERT(gyros);

            assignUpdateRate(gyros, SENSOR_UPDATE_PERIOD);
            connect(gyros, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        }
    } else {
        switch (sensor) {
        case ACCEL:
        {
            Accels * accels = Accels::GetInstance(getObjectManager());
            Q_ASSERT(accels);
            disconnect(accels, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        case MAG:
        {
            Magnetometer * mag = Magnetometer::GetInstance(getObjectManager());
            Q_ASSERT(mag);
            disconnect(mag, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        case GYRO:
        {
            Gyros * gyros = Gyros::GetInstance(getObjectManager());
            Q_ASSERT(gyros);
            disconnect(gyros, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        }
    }
}

/**
 * @brief Calibration::assignUpdateRate Assign a new update rate. The new metadata is sent
 * to the flight controller board in a separate operation.
 * @param obj
 * @param updatePeriod
 */
void Calibration::assignUpdateRate(UAVObject* obj, quint32 updatePeriod)
{
    // Fetch value from QMap
    UAVObject::Metadata mdata = metaDataList.value(obj->getName());

    // Fetch value from QMap, and change settings
    mdata = metaDataList.value(obj->getName());
    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
    mdata.flightTelemetryUpdatePeriod = updatePeriod;

    // Update QMap value
    metaDataList.insert(obj->getName(), mdata);
}

/**
 * @brief Calibration::dataUpdated Receive updates of any connected sensors and
 * process them based on the calibration state (e.g. six point, leveling, or
 * yaw orientation.)
 * @param obj The object that was updated
 */
void Calibration::dataUpdated(UAVObject * obj) {

    if (!timer.isActive()) {
        // ignore updates that come in after the timer has expired
        return;
    }

    switch(calibration_state) {
    case IDLE:
    case SIX_POINT_WAIT1:
    case SIX_POINT_WAIT2:
    case SIX_POINT_WAIT3:
    case SIX_POINT_WAIT4:
    case SIX_POINT_WAIT5:
    case SIX_POINT_WAIT6:
        // Do nothing
        return;
        break;
    case YAW_ORIENTATION:
        // Store data while computing the yaw orientation
        // and if completed go back to the idle state
        if(storeYawOrientationMeasurement(obj)) {
            //Disconnect sensors and reset metadata
            connectSensor(ACCEL, false);
            getObjectUtilManager()->setAllNonSettingsMetadata(originalMetaData);

            calibration_state = IDLE;

            emit showYawOrientationMessage(tr("Orientation computed"));
            emit toggleControls(true);
            emit yawOrientationProgressChanged(0);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case LEVELING:
        // Store data while computing the level attitude
        // and if completed go back to the idle state
        if(storeLevelingMeasurement(obj)) {
            //Disconnect sensors and reset metadata
            connectSensor(GYRO, false);
            connectSensor(ACCEL, false);
            getObjectUtilManager()->setAllNonSettingsMetadata(originalMetaData);

            calibration_state = IDLE;

            emit showLevelingMessage(tr("Level computed"));
            emit toggleControls(true);
            emit levelingProgressChanged(0);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT1:
        // These state collect each position for six point calibration and
        // when enough data is acquired advance to the next step
        if(storeSixPointMeasurement(obj,1)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT2;
            emit showSixPointMessage(tr("Rotate left side down and press Save Position..."));
            emit toggleSavePosition(true);
            emit updatePlane(2);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT2:
        if(storeSixPointMeasurement(obj,2)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT3;
            emit showSixPointMessage(tr("Rotate upside down and press Save Position..."));
            emit toggleSavePosition(true);
            emit updatePlane(3);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT3:
        if(storeSixPointMeasurement(obj,3)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT4;
            emit showSixPointMessage(tr("Point right side down and press Save Position..."));
            emit toggleSavePosition(true);
            emit updatePlane(4);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT4:
        if(storeSixPointMeasurement(obj,4)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT5;
            emit showSixPointMessage(tr("Point nose up and press Save Position..."));
            emit toggleSavePosition(true);
            emit updatePlane(5);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT5:
        if(storeSixPointMeasurement(obj,5)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT6;
            emit showSixPointMessage(tr("Point nose down and press Save Position..."));
            emit toggleSavePosition(true);
            emit updatePlane(6);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT6:
        // Store data points in the final position and if enough data
        // has been computed attempt to calculate the scale and bias
        // for the accel and optionally the mag.
        if(storeSixPointMeasurement(obj,6)) {
            // All data collected.  Disconnect and reset all UAVOs, and compute value
            connectSensor(GYRO, false);
            if (calibrateAccels)
                connectSensor(ACCEL, false);
            if (calibrateMags)
                connectSensor(MAG, false);
            getObjectUtilManager()->setAllNonSettingsMetadata(originalMetaData);

            calibration_state = IDLE;
            emit toggleControls(true);
            emit updatePlane(0);
            emit sixPointProgressChanged(0);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

            // Do calculation
            int ret=computeScaleBias();
            if (ret==CALIBRATION_SUCCESS) {
                // Load calibration results
                SensorSettings * sensorSettings = SensorSettings::GetInstance(getObjectManager());
                SensorSettings::DataFields sensorSettingsData = sensorSettings->getData();


                // Generate result messages
                QString accelCalibrationResults = "";
                QString magCalibrationResults = "";
                if (calibrateAccels == true) {
                    accelCalibrationResults = QString(tr("Accelerometer bias, in [m/s^2]: x=%1, y=%2, z=%3\n")).arg(sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_X], -9).arg(sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Y], -9).arg(sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Z], -9) +
                                              QString(tr("Accelerometer scale, in [-]:    x=%1, y=%2, z=%3\n")).arg(sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_X], -9).arg(sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Y], -9).arg(sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Z], -9);

                }
                if (calibrateMags == true) {
                    magCalibrationResults = QString(tr("Magnetometer bias, in [mG]: x=%1, y=%2, z=%3\n")).arg(sensorSettingsData.MagBias[SensorSettings::MAGBIAS_X], -9).arg(sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Y], -9).arg(sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Z], -9) +
                                            QString(tr("Magnetometer scale, in [-]: x=%4, y=%5, z=%6")).arg(sensorSettingsData.MagScale[SensorSettings::MAGSCALE_X], -9).arg(sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Y], -9).arg(sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Z], -9);
                }

                // Emit SIGNAL containing calibration success message
                emit showSixPointMessage(QString(tr("Calibration succeeded")) + QString("\n") + accelCalibrationResults + QString("\n") + magCalibrationResults);
            }
            else{
                //Return sensor calibration values to their original settings
                resetSensorCalibrationToOriginalValues();

                if(ret==ACCELEROMETER_FAILED){
                    emit showSixPointMessage(tr("Acceleromter calibration failed. Original values have been written back to device. Perhaps you moved too much during the calculation? Please repeat calibration."));
                }
                else if(ret==MAGNETOMETER_FAILED){
                    emit showSixPointMessage(tr("Magnetometer calibration failed. Original values have been written back to device. Perhaps you performed the calibration near iron? Please repeat calibration."));
                }
            }
        }
        break;
    case GYRO_TEMP_CAL:
        if (storeTempCalMeasurement(obj)) {
            // Disconnect and reset data and metadata
            connectSensor(GYRO, false);
            resetSensorCalibrationToOriginalValues();
            getObjectUtilManager()->setAllNonSettingsMetadata(originalMetaData);

            calibration_state = IDLE;
            emit toggleControls(true);

            int ret = computeTempCal();
            if (ret == CALIBRATION_SUCCESS) {
                emit showTempCalMessage(tr("Temperature compensation calibration succeeded"));
            } else {
                emit showTempCalMessage(tr("Temperature compensation calibration succeeded"));
            }
        }
    }

}

/**
 * @brief Calibration::timeout When collecting data for leveling, orientation, or
 * six point calibration times out. Clean up the state and reset
 */
void Calibration::timeout()
{
    // Reset metadata update rates
    getObjectUtilManager()->setAllNonSettingsMetadata(originalMetaData);

    switch(calibration_state) {
    case IDLE:
        // Do nothing
        return;
        break;
    case YAW_ORIENTATION:
        // Disconnect appropriate sensors
        connectSensor(ACCEL, false);
        calibration_state = IDLE;
        emit showYawOrientationMessage(tr("Orientation timed out ..."));
        emit yawOrientationProgressChanged(0);
    case LEVELING:
        // Disconnect appropriate sensors
        connectSensor(GYRO, false);
        connectSensor(ACCEL, false);
        calibration_state = IDLE;
        emit showLevelingMessage(tr("Leveling timed out ..."));
        emit levelingProgressChanged(0);
    case SIX_POINT_WAIT1:
    case SIX_POINT_WAIT2:
    case SIX_POINT_WAIT3:
    case SIX_POINT_WAIT4:
    case SIX_POINT_WAIT5:
    case SIX_POINT_WAIT6:
        // Do nothing, shouldn't happen
        return;
        break;
    case SIX_POINT_COLLECT1:
    case SIX_POINT_COLLECT2:
    case SIX_POINT_COLLECT3:
    case SIX_POINT_COLLECT4:
    case SIX_POINT_COLLECT5:
    case SIX_POINT_COLLECT6:
        connectSensor(GYRO, false);
        if (calibrateAccels)
            connectSensor(ACCEL, false);
        if (calibrateMags)
            connectSensor(MAG, false);
        calibration_state = IDLE;
        emit showSixPointMessage(tr("Six point data collection timed out"));
        emit sixPointProgressChanged(0);
        break;
    case GYRO_TEMP_CAL:
        connectSensor(GYRO, false);
        calibration_state = IDLE;
        emit showTempCalMessage(tr("Temperature calibration timed out"));
        emit tempCalProgressChanged(0);
        break;
    }

    emit updatePlane(0);
    emit toggleControls(true);

    disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

    QMessageBox msgBox;
    msgBox.setText(tr("Calibration timed out before receiving required updates."));
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
}

/**
 * @brief Calibration::doStartLeveling Called by UI to start collecting data to calculate level
 */
void Calibration::doStartOrientation() {
    accel_accum_x.clear();
    accel_accum_y.clear();
    accel_accum_z.clear();

    calibration_state = YAW_ORIENTATION;

    // Save previous sensor states
    originalMetaData = getObjectUtilManager()->readAllNonSettingsMetadata();

    // Set all UAVObject rates to update slowly
    UAVObjectManager *objManager = getObjectManager();
    QVector< QVector<UAVDataObject*> > objList = objManager->getDataObjectsVector();
    foreach (QVector<UAVDataObject*> list, objList) {
        foreach (UAVDataObject* obj, list) {
            if(!obj->isSettings()) {
                UAVObject::Metadata mdata = obj->getMetadata();
                UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);

                mdata.flightTelemetryUpdatePeriod = NON_SENSOR_UPDATE_PERIOD;
                metaDataList.insert(obj->getName(), mdata);
            }
        }
    }

    // Connect to the sensor updates and set higher rates
    connectSensor(ACCEL, true);

    // Set new metadata
    getObjectUtilManager()->setAllNonSettingsMetadata(metaDataList);

    emit toggleControls(false);
    emit showYawOrientationMessage(tr("Pitch vehicle forward approximately 30 degrees. Ensure it absolutely does not roll"));

    // Set up timeout timer
    timer.setSingleShot(true);
    timer.start(5000 + (NUM_SENSOR_UPDATES_YAW_ORIENTATION * SENSOR_UPDATE_PERIOD));
    connect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
}

//! Start collecting data while vehicle is level
void Calibration::doStartBiasAndLeveling()
{
    zeroVertical = true;
    doStartLeveling();
}

//! Start collecting data while vehicle is level
void Calibration::doStartNoBiasLeveling()
{
    zeroVertical = false;
    doStartLeveling();
}

/**
 * @brief Calibration::doStartLeveling Called by UI to start collecting data to calculate level
 */
void Calibration::doStartLeveling() {
    accel_accum_x.clear();
    accel_accum_y.clear();
    accel_accum_z.clear();
    gyro_accum_x.clear();
    gyro_accum_y.clear();
    gyro_accum_z.clear();
    gyro_accum_temp.clear();

    // Disable gyro bias correction to see raw data
    AttitudeSettings *attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
    Q_ASSERT(attitudeSettings);
    AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();
    attitudeSettingsData.BiasCorrectGyro = AttitudeSettings::BIASCORRECTGYRO_FALSE;
    attitudeSettings->setData(attitudeSettingsData);
    attitudeSettings->updated();

    calibration_state = LEVELING;

    // Save previous sensor states
    originalMetaData = getObjectUtilManager()->readAllNonSettingsMetadata();

    // Set all UAVObject rates to update slowly
    UAVObjectManager *objManager = getObjectManager();
    QVector< QVector<UAVDataObject*> > objList = objManager->getDataObjectsVector();
    foreach (QVector<UAVDataObject*> list, objList) {
        foreach (UAVDataObject* obj, list) {
            if(!obj->isSettings()) {
                UAVObject::Metadata mdata = obj->getMetadata();
                UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);

                mdata.flightTelemetryUpdatePeriod = NON_SENSOR_UPDATE_PERIOD;
                metaDataList.insert(obj->getName(), mdata);
            }
        }
    }

    // Connect to the sensor updates and set higher rates
    connectSensor(ACCEL, true);
    connectSensor(GYRO, true);

    // Set new metadata
    getObjectUtilManager()->setAllNonSettingsMetadata(metaDataList);

    emit toggleControls(false);
    emit showLevelingMessage(tr("Leave vehicle flat"));

    // Set up timeout timer
    timer.setSingleShot(true);
    timer.start(5000 + (NUM_SENSOR_UPDATES_LEVELING * SENSOR_UPDATE_PERIOD));
    connect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
}

/**
  * Called by the "Start" button.  Sets up the meta data and enables the
  * buttons to perform six point calibration of the magnetometer (optionally
  * accel) to compute the scale and bias of this sensor based on the current
  * home location magnetic strength.
  */
void Calibration::doStartSixPoint()
{

    // Save initial rotation settings
    AttitudeSettings * attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
    Q_ASSERT(attitudeSettings);
    AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();

    initialBoardRotation[0]=attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL];
    initialBoardRotation[1]=attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH];
    initialBoardRotation[2]=attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW];

    //Set board rotation to (0,0,0)
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] =0;
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH]=0;
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW]  =0;
    attitudeSettings->setData(attitudeSettingsData);

    // Save initial sensor settings
    SensorSettings * sensorSettings = SensorSettings::GetInstance(getObjectManager());
    Q_ASSERT(sensorSettings);
    SensorSettings::DataFields sensorSettingsData = sensorSettings->getData();

    // If calibrating the accelerometer, remove any scaling
    if (calibrateAccels) {
        initialAccelsScale[0]=sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_X];
        initialAccelsScale[1]=sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Y];
        initialAccelsScale[2]=sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Z];
        initialAccelsBias[0]=sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_X];
        initialAccelsBias[1]=sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Y];
        initialAccelsBias[2]=sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Z];

        // Reset the scale and bias to get a correct result
        sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_X] = 1.0;
        sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Y] = 1.0;
        sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Z] = 1.0;
        sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_X] = 0.0;
        sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Y] = 0.0;
        sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Z] = 0.0;
        sensorSettingsData.ZAccelOffset = 0.0;
    }

    // If calibrating the magnetometer, remove any scaling
    if (calibrateMags) {
        initialMagsScale[0]=sensorSettingsData.MagScale[SensorSettings::MAGSCALE_X];
        initialMagsScale[1]=sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Y];
        initialMagsScale[2]=sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Z];
        initialMagsBias[0]= sensorSettingsData.MagBias[SensorSettings::MAGBIAS_X];
        initialMagsBias[1]= sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Y];
        initialMagsBias[2]= sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Z];

        // Reset the scale to get a correct result
        sensorSettingsData.MagScale[SensorSettings::MAGSCALE_X] = 1;
        sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Y] = 1;
        sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Z] = 1;
        sensorSettingsData.MagBias[SensorSettings::MAGBIAS_X] = 0;
        sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Y] = 0;
        sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Z] = 0;
    }

    sensorSettings->setData(sensorSettingsData);

    // Clear the accumulators
    accel_accum_x.clear();
    accel_accum_y.clear();
    accel_accum_z.clear();
    mag_accum_x.clear();
    mag_accum_y.clear();
    mag_accum_z.clear();

    // TODO: Document why the thread needs to wait 100ms.
    Thread::usleep(100000);

    // Save previous sensor states
    originalMetaData = getObjectUtilManager()->readAllNonSettingsMetadata();

    // Make all UAVObject rates update slowly
    UAVObjectManager *objManager = getObjectManager();
    QVector< QVector<UAVDataObject*> > objList = objManager->getDataObjectsVector();
    foreach (QVector<UAVDataObject*> list, objList) {
        foreach (UAVDataObject* obj, list) {
            if(!obj->isSettings()) {
                UAVObject::Metadata mdata = obj->getMetadata();
                UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);

                mdata.flightTelemetryUpdatePeriod = NON_SENSOR_UPDATE_PERIOD;
                metaDataList.insert(obj->getName(), mdata);
            }
        }
    }

    // Connect sensors and set higher update rate
    if (calibrateAccels)
        connectSensor(ACCEL, true);
    if(calibrateMags)
        connectSensor(MAG, true);

    // Set new metadata
    getObjectUtilManager()->setAllNonSettingsMetadata(metaDataList);

    // Show UI parts and update the calibration state
    emit showSixPointMessage(tr("Place horizontally and click save position..."));
    emit updatePlane(1);
    emit toggleControls(false);
    emit toggleSavePosition(true);
    calibration_state = SIX_POINT_WAIT1;
}


/**
 * @brief Calibration::doCancelSixPoint Cancels six point calibration and returns all values to their original settings.
 */
void Calibration::doCancelSixPoint(){
    //Return sensor calibration values to their original settings
    resetSensorCalibrationToOriginalValues();

    // Disconnect sensors and reset UAVO update rates
    if (calibrateAccels)
        connectSensor(ACCEL, false);
    if(calibrateMags)
        connectSensor(MAG, false);

    getObjectUtilManager()->setAllNonSettingsMetadata(originalMetaData);

    calibration_state = IDLE;
    emit toggleControls(true);
    emit toggleSavePosition(false);
    emit updatePlane(0);
    emit sixPointProgressChanged(0);
    disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

    emit showSixPointMessage(tr("Calibration canceled."));

}


/**
  * Tells the calibration utility the UAV is in position and to collect data.
  */
void Calibration::doSaveSixPointPosition()
{
    switch(calibration_state) {
    case SIX_POINT_WAIT1:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT1;
        break;
    case SIX_POINT_WAIT2:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT2;
        break;
    case SIX_POINT_WAIT3:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT3;
        break;
    case SIX_POINT_WAIT4:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT4;
        break;
    case SIX_POINT_WAIT5:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT5;
        break;
    case SIX_POINT_WAIT6:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT6;
        break;
    default:
        return;
        break;
    }

    emit toggleSavePosition(false);

    // Set up timeout timer
    timer.setSingleShot(true);
    timer.start(5000 + (NUM_SENSOR_UPDATES_SIX_POINT * SENSOR_UPDATE_PERIOD));
    connect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
}

/**
 * Start collecting gyro calibration data
 */
void Calibration::doStartTempCal()
{
    gyro_accum_x.clear();
    gyro_accum_y.clear();
    gyro_accum_z.clear();
    gyro_accum_temp.clear();

    // Disable gyro sensor-frame rotation and bias correction to see raw data
    AttitudeSettings *attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
    Q_ASSERT(attitudeSettings);
    AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();

    initialBoardRotation[0] = attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL];
    initialBoardRotation[1] = attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH];
    initialBoardRotation[2] = attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW];

    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] = 0;
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH]= 0;
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW]  = 0;
    attitudeSettingsData.BiasCorrectGyro = AttitudeSettings::BIASCORRECTGYRO_FALSE;

    attitudeSettings->setData(attitudeSettingsData);
    attitudeSettings->updated();

    calibration_state = GYRO_TEMP_CAL;

    // Save previous sensor states
    originalMetaData = getObjectUtilManager()->readAllNonSettingsMetadata();

    // Connect to the sensor updates and speed them up
    connectSensor(GYRO, true);

    // Set new metadata
    getObjectUtilManager()->setAllNonSettingsMetadata(metaDataList);

    emit toggleControls(false);
    emit showTempCalMessage(tr("Leave board flat and very still while it changes temperature"));
    emit tempCalProgressChanged(0);

    // Set up timeout timer
    timer.setSingleShot(true);
    timer.start(1800000);
    connect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
}

/**
 * @brief Calibration::doCancelTempCalPoint Abort the temperature calibration
 */
void Calibration::doAcceptTempCal()
{
    if (calibration_state == GYRO_TEMP_CAL) {
        qDebug() << "Accepting";
        // Disconnect sensor and reset UAVO update rates
        connectSensor(GYRO, false);
        getObjectUtilManager()->setAllNonSettingsMetadata(originalMetaData);

        calibration_state = IDLE;
        emit showTempCalMessage(tr("Temperature calibration accepted"));
        emit tempCalProgressChanged(0);
        emit toggleControls(true);

        timer.stop();
        disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

        computeTempCal();
    }
}

/**
 * @brief Calibration::doCancelTempCalPoint Abort the temperature calibration
 */
void Calibration::doCancelTempCalPoint()
{
    if (calibration_state == GYRO_TEMP_CAL) {
        qDebug() << "Canceling";
        // Disconnect sensor and reset UAVO update rates
        connectSensor(GYRO, false);
        getObjectUtilManager()->setAllNonSettingsMetadata(originalMetaData);

        // Reenable gyro bias correction
        AttitudeSettings *attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
        Q_ASSERT(attitudeSettings);
        AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();
        attitudeSettings->setData(attitudeSettingsData);
        attitudeSettings->updated();
        Thread::usleep(100000); // Sleep 100ms to make sure the new settings values are sent

        // Reset all sensor values
        resetSensorCalibrationToOriginalValues();

        calibration_state = IDLE;
        emit showTempCalMessage(tr("Temperature calibration timed out"));
        emit tempCalProgressChanged(0);
        emit toggleControls(true);

        timer.stop();
        disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
    }
}

/**
 * @brief Calibration::setTempCalRange Set the range for calibration
 * @param r The number of degrees that must be spanned for calibration
 * to terminate
 */
void Calibration::setTempCalRange(int r)
{
    MIN_TEMPERATURE_RANGE = r;
}


/**
 * @brief Calibration::storeYawOrientationMeasurement Store an accelerometer
 * measurement.
 * @return true if enough data is collected
 */
bool Calibration::storeYawOrientationMeasurement(UAVObject *obj)
{
    Accels * accels = Accels::GetInstance(getObjectManager());

    // Accumulate samples until we have _at least_ NUM_SENSOR_UPDATES_YAW_ORIENTATION samples
    if(obj->getObjID() == Accels::OBJID) {
        Accels::DataFields accelsData = accels->getData();
        accel_accum_x.append(accelsData.x);
        accel_accum_y.append(accelsData.y);
        accel_accum_z.append(accelsData.z);
    }

    // update the progress indicator
    emit yawOrientationProgressChanged((double)accel_accum_x.size() / NUM_SENSOR_UPDATES_YAW_ORIENTATION * 100);

    // If we have enough samples, then stop sampling and compute the biases
    if (accel_accum_x.size() >= NUM_SENSOR_UPDATES_YAW_ORIENTATION) {
        timer.stop();
        disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

        // Get the existing attitude settings
        AttitudeSettings *attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
        AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();

        // Use sensor data without rotation, as it has already been rotated on-board.
        double a_body[3] = { listMean(accel_accum_x), listMean(accel_accum_y), listMean(accel_accum_z) };

        // Temporary variable
        double psi;

        // Solve "a_sensor = Rot(phi, theta, psi) *[0;0;-g]" for the roll (phi) and pitch (theta) values.
        // Recall that phi is in [-pi, pi] and theta is in [-pi/2, pi/2]
        psi = atan2(a_body[1], -a_body[0]);

        attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] += psi * RAD2DEG * 100.0; // Scale by 100 because units are [100*deg]

        // Wrap to [-pi, pi]
        while (attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] > 180*100)  // Scale by 100 because units are [100*deg]
            attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] -= 360*100;
        while (attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] < -180*100)
            attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] += 360*100;

        attitudeSettings->setData(attitudeSettingsData);
        attitudeSettings->updated();

        // Inform the system that the calibration process has completed
        emit calibrationCompleted();

        return true;
    }
    return false;
}


/**
 * @brief Calibration::storeLevelingMeasurement Store a measurement and if there
 * is enough data compute the level angle and gyro zero
 * @return true if enough data is collected
 */
bool Calibration::storeLevelingMeasurement(UAVObject *obj) {
    Accels * accels = Accels::GetInstance(getObjectManager());
    Gyros * gyros = Gyros::GetInstance(getObjectManager());

    // Accumulate samples until we have _at least_ NUM_SENSOR_UPDATES_LEVELING samples
    if(obj->getObjID() == Accels::OBJID) {
        Accels::DataFields accelsData = accels->getData();
        accel_accum_x.append(accelsData.x);
        accel_accum_y.append(accelsData.y);
        accel_accum_z.append(accelsData.z);
    } else if (obj->getObjID() == Gyros::OBJID) {
        Gyros::DataFields gyrosData = gyros->getData();
        gyro_accum_x.append(gyrosData.x);
        gyro_accum_y.append(gyrosData.y);
        gyro_accum_z.append(gyrosData.z);
        gyro_accum_temp.append(gyrosData.temperature);
    }

    // update the progress indicator
    emit levelingProgressChanged((float) qMin(accel_accum_x.size(),  gyro_accum_x.size()) / NUM_SENSOR_UPDATES_LEVELING * 100);

    // If we have enough samples, then stop sampling and compute the biases
    if (accel_accum_x.size() >= NUM_SENSOR_UPDATES_LEVELING && gyro_accum_x.size() >= NUM_SENSOR_UPDATES_LEVELING) {
        timer.stop();
        disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

        float x_gyro_bias = listMean(gyro_accum_x);
        float y_gyro_bias = listMean(gyro_accum_y);
        float z_gyro_bias = listMean(gyro_accum_z);
        float temp = listMean(gyro_accum_temp);

        // Get the existing attitude settings
        AttitudeSettings::DataFields attitudeSettingsData = AttitudeSettings::GetInstance(getObjectManager())->getData();
        SensorSettings::DataFields sensorSettingsData = SensorSettings::GetInstance(getObjectManager())->getData();

        // Inverse rotation of sensor data, from body frame into sensor frame
        double a_body[3] = { listMean(accel_accum_x), listMean(accel_accum_y), listMean(accel_accum_z) };
        double a_sensor[3]; //! Store the sensor data without any rotation
        double Rsb[3][3];  // The initial body-frame to sensor-frame rotation
        double rpy[3] = { attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] * DEG2RAD / 100.0,
                          attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] * DEG2RAD / 100.0,
                          attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] * DEG2RAD / 100.0};
        Euler2R(rpy, Rsb);
        rotate_vector(Rsb, a_body, a_sensor, false);

        // Temporary variables
        double psi, theta, phi;
        Q_UNUSED(psi);
        // Keep existing yaw rotation
        psi = attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] * DEG2RAD / 100.0;

        // Solve "a_sensor = Rot(phi, theta, psi) *[0;0;-g]" for the roll (phi) and pitch (theta) values.
        // Recall that phi is in [-pi, pi] and theta is in [-pi/2, pi/2]
        phi = atan2f(-a_sensor[1], -a_sensor[2]);
        theta = atanf(-a_sensor[0] / (sinf(phi)*a_sensor[1] + cosf(phi)*a_sensor[2]));

        attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] = phi * RAD2DEG * 100.0;
        attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] = theta * RAD2DEG * 100.0;

        if (zeroVertical) {
            // If requested, calculate the offset in the z accelerometer that
            // would make it reflect gravity

            // Rotate the accel measurements to the new body frame
            double rpy[3] = { attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] * DEG2RAD / 100.0,
                              attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] * DEG2RAD / 100.0,
                              attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] * DEG2RAD / 100.0};
            double a_body_new[3];
            Euler2R(rpy, Rsb);
            rotate_vector(Rsb, a_sensor, a_body_new, false);

            // Compute the new offset to make it average accelLength(GRAVITY)
            sensorSettingsData.ZAccelOffset += -(a_body_new[2] + accelLength);
        }

        // Rotate the gyro bias from the body frame into the sensor frame
        double gyro_sensor[3];
        double gyro_body[3] = {x_gyro_bias, y_gyro_bias, z_gyro_bias};
        rotate_vector(Rsb, gyro_body, gyro_sensor, false);

        // Store these new biases, accounting for any temperature coefficients
        sensorSettingsData.XGyroTempCoeff[0] = gyro_sensor[0] -
                temp * sensorSettingsData.XGyroTempCoeff[1] -
                pow(temp,2) * sensorSettingsData.XGyroTempCoeff[2] -
                pow(temp,3) * sensorSettingsData.XGyroTempCoeff[3];
        sensorSettingsData.YGyroTempCoeff[0] = gyro_sensor[1] -
                temp * sensorSettingsData.YGyroTempCoeff[1] -
                pow(temp,2) * sensorSettingsData.YGyroTempCoeff[2] -
                pow(temp,3) * sensorSettingsData.YGyroTempCoeff[3];
        sensorSettingsData.ZGyroTempCoeff[0] = gyro_sensor[2] -
                temp * sensorSettingsData.ZGyroTempCoeff[1] -
                pow(temp,2) * sensorSettingsData.ZGyroTempCoeff[2] -
                pow(temp,3) * sensorSettingsData.ZGyroTempCoeff[3];
        SensorSettings::GetInstance(getObjectManager())->setData(sensorSettingsData);

        // We offset the gyro bias by current bias to help precision
        // Disable gyro bias correction to see raw data
        AttitudeSettings *attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
        Q_ASSERT(attitudeSettings);
        attitudeSettingsData.BiasCorrectGyro = AttitudeSettings::BIASCORRECTGYRO_TRUE;
        attitudeSettings->setData(attitudeSettingsData);
        attitudeSettings->updated();

        // After recomputing the level for a frame, zero the trim settings
        TrimAnglesSettings *trimSettings = TrimAnglesSettings::GetInstance(getObjectManager());
        Q_ASSERT(trimSettings);
        TrimAnglesSettings::DataFields trim = trimSettings->getData();
        trim.Pitch = 0;
        trim.Roll = 0;
        trimSettings->setData(trim);

        // Inform the system that the calibration process has completed
        emit calibrationCompleted();

        return true;
    }
    return false;
}

/**
  * Grab a sample of accel or mag data while in this position and
  * store it for averaging.
  * @return true If enough data is averaged at this position
  */
bool Calibration::storeSixPointMeasurement(UAVObject * obj, int position)
{
    // Position is specified 1-6, but used as an index
    Q_ASSERT(position >= 1 && position <= 6);
    position --;

    if( calibrateAccels && obj->getObjID() == Accels::OBJID ) {
        Accels * accels = Accels::GetInstance(getObjectManager());
        Q_ASSERT(accels);
        Accels::DataFields accelsData = accels->getData();

        accel_accum_x.append(accelsData.x);
        accel_accum_y.append(accelsData.y);
        accel_accum_z.append(accelsData.z);
    }

    if( calibrateMags && obj->getObjID() == Magnetometer::OBJID) {
        Magnetometer * mag = Magnetometer::GetInstance(getObjectManager());
        Q_ASSERT(mag);
        Magnetometer::DataFields magData = mag->getData();

        mag_accum_x.append(magData.x);
        mag_accum_y.append(magData.y);
        mag_accum_z.append(magData.z);
    }

    // Update progress bar
    float progress_percentage;
    if(calibrateAccels && !calibrateMags)
        progress_percentage = (float) accel_accum_x.size() / NUM_SENSOR_UPDATES_SIX_POINT * 100;
    else if(!calibrateAccels && calibrateMags)
        progress_percentage = (float) mag_accum_x.size() / NUM_SENSOR_UPDATES_SIX_POINT * 100;
    else
        progress_percentage = fminf(mag_accum_x.size(), accel_accum_x.size()) / NUM_SENSOR_UPDATES_SIX_POINT * 100;
    emit sixPointProgressChanged(progress_percentage);

    // If enough data is collected, average it for this position
    if((!calibrateAccels || accel_accum_x.size() >= NUM_SENSOR_UPDATES_SIX_POINT) &&
            (!calibrateMags || mag_accum_x.size() >= NUM_SENSOR_UPDATES_SIX_POINT)) {

        // Store the average accelerometer value in that position
        if (calibrateAccels) {
            accel_data_x[position] = listMean(accel_accum_x);
            accel_data_y[position] = listMean(accel_accum_y);
            accel_data_z[position] = listMean(accel_accum_z);
            accel_accum_x.clear();
            accel_accum_y.clear();
            accel_accum_z.clear();
        }

        // Store the average magnetometer value in that position
        if (calibrateMags) {
            mag_data_x[position] = listMean(mag_accum_x);
            mag_data_y[position] = listMean(mag_accum_y);
            mag_data_z[position] = listMean(mag_accum_z);
            mag_accum_x.clear();
            mag_accum_y.clear();
            mag_accum_z.clear();
        }

        // Indicate all data collected for this position
        return true;
    }
    return false;
}

/**
 * @brief Calibration::configureTempCurves
 * @param x
 * @param y
 * @param z
 */
void Calibration::configureTempCurves(TempCompCurve *x,
                                      TempCompCurve *y,
                                      TempCompCurve *z)
{
    xCurve = x;
    yCurve = y;
    zCurve = z;
}

/**
  * Grab a sample of gyro data with the temperautre
  * @return true If enough data is averaged at this position
  */
bool Calibration::storeTempCalMeasurement(UAVObject * obj)
{
    if (obj->getObjID() == Gyros::OBJID) {
        Gyros *gyros = Gyros::GetInstance(getObjectManager());
        Q_ASSERT(gyros);
        Gyros::DataFields gyrosData = gyros->getData();
        gyro_accum_x.append(gyrosData.x);
        gyro_accum_y.append(gyrosData.y);
        gyro_accum_z.append(gyrosData.z);
        gyro_accum_temp.append(gyrosData.temperature);
    }

    double range = listMax(gyro_accum_temp) - listMin(gyro_accum_temp);
    emit tempCalProgressChanged((float) range / MIN_TEMPERATURE_RANGE * 100);

    if ((gyro_accum_temp.size() % 10) == 0) {
        updateTempCompCalibrationDisplay();
    }

    // If enough data is collected, average it for this position
    if(range >= MIN_TEMPERATURE_RANGE) {
        return true;
    }

    return false;
}

/**
 * @brief Calibration::updateTempCompCalibrationDisplay
 */
void Calibration::updateTempCompCalibrationDisplay()
{
    unsigned int n_samples = gyro_accum_temp.size();

    // Construct the matrix of temperature.
    Eigen::Matrix<double, Eigen::Dynamic, 4> X(n_samples, 4);

    // And the matrix of gyro samples.
    Eigen::Matrix<double, Eigen::Dynamic, 3> Y(n_samples, 3);

    for (unsigned i = 0; i < n_samples; ++i) {
        X(i,0) = 1;
        X(i,1) = gyro_accum_temp[i];
        X(i,2) = pow(gyro_accum_temp[i],2);
        X(i,3) = pow(gyro_accum_temp[i],3);
        Y(i,0) = gyro_accum_x[i];
        Y(i,1) = gyro_accum_y[i];
        Y(i,2) = gyro_accum_z[i];
    }

    // Solve Y = X * B

    Eigen::Matrix<double, 4, 3> result;
    // Use the cholesky-based Penrose pseudoinverse method.
    (X.transpose() * X).ldlt().solve(X.transpose()*Y, &result);

    QList<double> xCoeffs, yCoeffs, zCoeffs;
    xCoeffs.clear();
    xCoeffs.append(result(0,0));
    xCoeffs.append(result(1,0));
    xCoeffs.append(result(2,0));
    xCoeffs.append(result(3,0));
    yCoeffs.clear();
    yCoeffs.append(result(0,1));
    yCoeffs.append(result(1,1));
    yCoeffs.append(result(2,1));
    yCoeffs.append(result(3,1));
    zCoeffs.clear();
    zCoeffs.append(result(0,2));
    zCoeffs.append(result(1,2));
    zCoeffs.append(result(2,2));
    zCoeffs.append(result(3,2));

    if (xCurve != NULL)
        xCurve->plotData(gyro_accum_temp, gyro_accum_x, xCoeffs);
    if (yCurve != NULL)
        yCurve->plotData(gyro_accum_temp, gyro_accum_y, yCoeffs);
    if (zCurve != NULL)
        zCurve->plotData(gyro_accum_temp, gyro_accum_z, zCoeffs);

}

/**
 * @brief Calibration::tempCalProgressChanged Compute a polynominal fit to all
 * of the temperature data and each gyro channel
 * @return
 */
int Calibration::computeTempCal()
{
    timer.stop();
    disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

    // Reenable gyro bias correction
    AttitudeSettings *attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
    Q_ASSERT(attitudeSettings);
    AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();
    attitudeSettingsData.BiasCorrectGyro = AttitudeSettings::BIASCORRECTGYRO_TRUE;
    attitudeSettings->setData(attitudeSettingsData);
    attitudeSettings->updated();

    unsigned int n_samples = gyro_accum_temp.size();

    // Construct the matrix of temperature.
    Eigen::Matrix<double, Eigen::Dynamic, 4> X(n_samples, 4);

    // And the matrix of gyro samples.
    Eigen::Matrix<double, Eigen::Dynamic, 3> Y(n_samples, 3);

    for (unsigned i = 0; i < n_samples; ++i) {
        X(i,0) = 1;
        X(i,1) = gyro_accum_temp[i];
        X(i,2) = pow(gyro_accum_temp[i],2);
        X(i,3) = pow(gyro_accum_temp[i],3);
        Y(i,0) = gyro_accum_x[i];
        Y(i,1) = gyro_accum_y[i];
        Y(i,2) = gyro_accum_z[i];
    }

    // Solve Y = X * B

    Eigen::Matrix<double, 4, 3> result;
    // Use the cholesky-based Penrose pseudoinverse method.
    (X.transpose() * X).ldlt().solve(X.transpose()*Y, &result);

    //qDebug() << "Solution" << result;
    qDebug() << "Solution: ";
    qDebug() << "[" << result(0,0) << " " << result(0,1) << " " << result(0,2) << "]";
    qDebug() << "[" << result(1,0) << " " << result(1,1) << " " << result(1,2) << "]";
    qDebug() << "[" << result(2,0) << " " << result(2,1) << " " << result(2,2) << "]";
    qDebug() << "[" << result(3,0) << " " << result(3,1) << " " << result(3,2) << "]";

    // Store the results
    SensorSettings * sensorSettings = SensorSettings::GetInstance(getObjectManager());
    Q_ASSERT(sensorSettings);
    SensorSettings::DataFields sensorSettingsData = sensorSettings->getData();
    sensorSettingsData.XGyroTempCoeff[0] = result(0,0);
    sensorSettingsData.XGyroTempCoeff[1] = result(1,0);
    sensorSettingsData.XGyroTempCoeff[2] = result(2,0);
    sensorSettingsData.XGyroTempCoeff[3] = result(3,0);
    sensorSettingsData.YGyroTempCoeff[0] = result(0,1);
    sensorSettingsData.YGyroTempCoeff[1] = result(1,1);
    sensorSettingsData.YGyroTempCoeff[2] = result(2,1);
    sensorSettingsData.YGyroTempCoeff[3] = result(3,1);
    sensorSettingsData.ZGyroTempCoeff[0] = result(0,2);
    sensorSettingsData.ZGyroTempCoeff[1] = result(1,2);
    sensorSettingsData.ZGyroTempCoeff[2] = result(2,2);
    sensorSettingsData.ZGyroTempCoeff[3] = result(3,2);
    sensorSettings->setData(sensorSettingsData);

    QList<double> xCoeffs, yCoeffs, zCoeffs;
    xCoeffs.clear();
    xCoeffs.append(result(0,0));
    xCoeffs.append(result(1,0));
    xCoeffs.append(result(2,0));
    xCoeffs.append(result(3,0));
    yCoeffs.clear();
    yCoeffs.append(result(0,1));
    yCoeffs.append(result(1,1));
    yCoeffs.append(result(2,1));
    yCoeffs.append(result(3,1));
    zCoeffs.clear();
    zCoeffs.append(result(0,2));
    zCoeffs.append(result(1,2));
    zCoeffs.append(result(2,2));
    zCoeffs.append(result(3,2));

    if (xCurve != NULL)
        xCurve->plotData(gyro_accum_temp, gyro_accum_x, xCoeffs);
    if (yCurve != NULL)
        yCurve->plotData(gyro_accum_temp, gyro_accum_y, yCoeffs);
    if (zCurve != NULL)
        zCurve->plotData(gyro_accum_temp, gyro_accum_z, zCoeffs);

    emit tempCalProgressChanged(0);

    // Inform the system that the calibration process has completed
    emit calibrationCompleted();

    return CALIBRATION_SUCCESS;
}

/**
 * Util function to get a pointer to the object manager
 * @return pointer to the UAVObjectManager
 */
UAVObjectManager* Calibration::getObjectManager() {
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager * objMngr = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objMngr);
    return objMngr;
}

/**
 * @brief Calibration::getObjectUtilManager Utility function to get a pointer to the object manager utilities
 * @return pointer to the UAVObjectUtilManager
 */
UAVObjectUtilManager* Calibration::getObjectUtilManager() {
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectUtilManager *utilMngr = pm->getObject<UAVObjectUtilManager>();
    Q_ASSERT(utilMngr);
    return utilMngr;
}

/**
 * Utility function which calculates the Mean value of a list of values
 * @param list list of double values
 * @returns Mean value of the list of parameter values
 */
double Calibration::listMean(QList<double> list)
{
    double accum = 0;
    for(int i = 0; i < list.size(); i++)
        accum += list[i];
    return accum / list.size();
}

/**
 * Utility function which calculates the Mean value of a list of values
 * @param list list of double values
 * @returns Mean value of the list of parameter values
 */
double Calibration::listMin(QList<double> list)
{
    double min = list[0];
    for(int i = 0; i < list.size(); i++)
        min = qMin(min, list[i]);
    return min;
}

/**
 * Utility function which calculates the Mean value of a list of values
 * @param list list of double values
 * @returns Mean value of the list of parameter values
 */
double Calibration::listMax(QList<double> list)
{
    double max = list[0];
    for(int i = 0; i < list.size(); i++)
        max = qMax(max, list[i]);
    return max;
}

/**
  * Computes the scale and bias for the accelerometer and mag once all the data
  * has been collected in 6 positions.
  */
int Calibration::computeScaleBias()
{
    SensorSettings * sensorSettings = SensorSettings::GetInstance(getObjectManager());
    Q_ASSERT(sensorSettings);
    SensorSettings::DataFields sensorSettingsData = sensorSettings->getData();

    // Regardless of calibration result, set board rotations back to user settings
    AttitudeSettings * attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
    Q_ASSERT(attitudeSettings);
    AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] = initialBoardRotation[0];
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] = initialBoardRotation[1];
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] = initialBoardRotation[2];
    attitudeSettings->setData(attitudeSettingsData);

    bool good_calibration = true;

    //Assign calibration data
    if (calibrateAccels) {
        good_calibration = true;

        qDebug() << "Accel measurements";
        for(int i = 0; i < 6; i++)
            qDebug() << accel_data_x[i] << ", " << accel_data_y[i] << ", " << accel_data_z[i] << ";";

        // Solve for accelerometer calibration
        double S[3], b[3];
        SixPointInConstFieldCal(accelLength, accel_data_x, accel_data_y, accel_data_z, S, b);

        sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_X] += (-sign(S[0]) * b[0]);
        sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Y] += (-sign(S[1]) * b[1]);
        sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Z] += (-sign(S[2]) * b[2]);

        sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_X] *= fabs(S[0]);
        sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Y] *= fabs(S[1]);
        sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Z] *= fabs(S[2]);

        // Check the accel calibration is good
        good_calibration &= sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_X] ==
                sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_X];
        good_calibration &= sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Y] ==
                sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Y];
        good_calibration &= sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Z] ==
                sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Z];
        good_calibration &= (sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_X] ==
                             sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_X]);
        good_calibration &= (sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Y] ==
                             sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Y]);
        good_calibration &= (sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Z] ==
                             sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Z]);

        //This can happen if, for instance, HomeLocation.g_e == 0
        if((S[0]+S[1]+S[2])<0.0001){
            good_calibration=false;
        }

        if (good_calibration) {
            qDebug()<<  "Accel bias: " << sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_X] << " " << sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Y] << " " << sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Z];
        } else {
            return ACCELEROMETER_FAILED;
        }
    }

    if (calibrateMags) {
        good_calibration = true;

        qDebug() << "Mag measurements";
        for(int i = 0; i < 6; i++)
            qDebug() << mag_data_x[i] << ", " << mag_data_y[i] << ", " << mag_data_z[i] << ";";

        // Work out the average vector length as nominal since mag scale normally close to 1
        // and we don't use the magnitude anyway.  Avoids requiring home location.
        double m_x = 0, m_y = 0, m_z = 0, len = 0;
        for (int i = 0; i < 6; i++) {
            m_x += mag_data_x[i];
            m_y += mag_data_x[i];
            m_z += mag_data_x[i];
        }
        m_x /= 6;
        m_y /= 6;
        m_z /= 6;
        for (int i = 0; i < 6; i++) {
            len += sqrt(pow(mag_data_x[i] - m_x,2) + pow(mag_data_y[i] - m_y,2) + pow(mag_data_z[i] - m_z,2));
        }
        len /= 6;

        // Solve for magnetometer calibration
        double S[3], b[3];
        SixPointInConstFieldCal(len, mag_data_x, mag_data_y, mag_data_z, S, b);

        //Assign calibration data
        sensorSettingsData.MagBias[SensorSettings::MAGBIAS_X] += (-sign(S[0]) * b[0]);
        sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Y] += (-sign(S[1]) * b[1]);
        sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Z] += (-sign(S[2]) * b[2]);
        sensorSettingsData.MagScale[SensorSettings::MAGSCALE_X] *= fabs(S[0]);
        sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Y] *= fabs(S[1]);
        sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Z] *= fabs(S[2]);

        // Check the mag calibration is good
        good_calibration &= sensorSettingsData.MagBias[SensorSettings::MAGBIAS_X] ==
                sensorSettingsData.MagBias[SensorSettings::MAGBIAS_X];
        good_calibration &= sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Y] ==
                sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Y];
        good_calibration &= sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Z]  ==
                sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Z] ;
        good_calibration &= sensorSettingsData.MagScale[SensorSettings::MAGSCALE_X] ==
                sensorSettingsData.MagScale[SensorSettings::MAGSCALE_X];
        good_calibration &= sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Y] ==
                sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Y];
        good_calibration &= sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Z] ==
                sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Z];

        //This can happen if, for instance, HomeLocation.g_e == 0
        if((S[0]+S[1]+S[2])<0.0001){
            good_calibration=false;
        }

        if (good_calibration) {
            qDebug()<<  "Mag bias: " << sensorSettingsData.MagBias[SensorSettings::MAGBIAS_X] << " " << sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Y]  << " " << sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Z];
        } else {
            return MAGNETOMETER_FAILED;
        }
    }

    // If we've made it this far, it's because good_calibration == true
    sensorSettings->setData(sensorSettingsData);

    // Inform the system that the calibration process has completed
    emit calibrationCompleted();

    return CALIBRATION_SUCCESS;
}

/**
 * @brief Calibration::resetToOriginalValues Resets the accelerometer and magnetometer setting to their pre-calibration values
 */
void Calibration::resetSensorCalibrationToOriginalValues()
{
    // Write original board rotation settings back to device
    AttitudeSettings * attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
    Q_ASSERT(attitudeSettings);
    AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();

    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] = initialBoardRotation[0];
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] = initialBoardRotation[1];
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] = initialBoardRotation[2];
    attitudeSettings->setData(attitudeSettingsData);
    attitudeSettings->updated();


    //Write the original accelerometer values back to the device
    SensorSettings * sensorSettings = SensorSettings::GetInstance(getObjectManager());
    Q_ASSERT(sensorSettings);
    SensorSettings::DataFields sensorSettingsData = sensorSettings->getData();

    if (calibrateAccels) {
        sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_X] = initialAccelsScale[0];
        sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Y] = initialAccelsScale[1];
        sensorSettingsData.AccelScale[SensorSettings::ACCELSCALE_Z] = initialAccelsScale[2];
        sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_X] = initialAccelsBias[0];
        sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Y] = initialAccelsBias[1];
        sensorSettingsData.AccelBias[SensorSettings::ACCELBIAS_Z] = initialAccelsBias[2];
    }

    if (calibrateMags) {
        //Write the original magnetometer values back to the device
        sensorSettingsData.MagScale[SensorSettings::MAGSCALE_X] = initialMagsScale[0];
        sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Y] = initialMagsScale[1];
        sensorSettingsData.MagScale[SensorSettings::MAGSCALE_Z] = initialMagsScale[2];
        sensorSettingsData.MagBias[SensorSettings::MAGBIAS_X] = initialMagsBias[0];
        sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Y] = initialMagsBias[1];
        sensorSettingsData.MagBias[SensorSettings::MAGBIAS_Z] = initialMagsBias[2];
    }

    sensorSettings->setData(sensorSettingsData);
    sensorSettings->updated();
}

/**
 * @brief Compute a rotation matrix from a set of euler angles
 * @param[in] rpy The euler angles in roll, pitch, yaw
 * @param[out] Rbe The rotation matrix to take a matrix from
 *       0,0,0 to that rotation
 */
void Calibration::Euler2R(double rpy[3], double Rbe[3][3])
{
    double sF = sin(rpy[0]), cF = cos(rpy[0]);
    double sT = sin(rpy[1]), cT = cos(rpy[1]);
    double sP = sin(rpy[2]), cP = cos(rpy[2]);

    Rbe[0][0] = cT*cP;
    Rbe[0][1] = cT*sP;
    Rbe[0][2] = -sT;
    Rbe[1][0] = sF*sT*cP - cF*sP;
    Rbe[1][1] = sF*sT*sP + cF*cP;
    Rbe[1][2] = cT*sF;
    Rbe[2][0] = cF*sT*cP + sF*sP;
    Rbe[2][1] = cF*sT*sP - sF*cP;
    Rbe[2][2] = cT*cF;
}

/**
 * @brief Rotate a vector by the rotation matrix, optionally trasposing
 * @param[in] R the rotation matrix
 * @param[in] vec The vector to rotate by this matrix
 * @param[out] vec_out The rotated vector
 * @param[in] transpose Optionally transpose the rotation matrix first (reverse the rotation)
 */
void Calibration::rotate_vector(double R[3][3], const double vec[3], double vec_out[3], bool transpose = true)
{
    if (!transpose){
        vec_out[0] = R[0][0] * vec[0] + R[0][1] * vec[1] + R[0][2] * vec[2];
        vec_out[1] = R[1][0] * vec[0] + R[1][1] * vec[1] + R[1][2] * vec[2];
        vec_out[2] = R[2][0] * vec[0] + R[2][1] * vec[1] + R[2][2] * vec[2];
    }
    else {
        vec_out[0] = R[0][0] * vec[0] + R[1][0] * vec[1] + R[2][0] * vec[2];
        vec_out[1] = R[0][1] * vec[0] + R[1][1] * vec[1] + R[2][1] * vec[2];
        vec_out[2] = R[0][2] * vec[0] + R[1][2] * vec[1] + R[2][2] * vec[2];
    }
}

/**
 * @name LinearEquationsSolving
 * @brief Uses Gaussian Elimination to solve linear equations of the type Ax=b
 *
 * @note Matrix code snarfed from: http://www.hlevkin.com/NumAlg/LinearEquations.c
 *
 * @return 0 if system not solving
 * @param[in] nDim - system dimension
 * @param[in] pfMatr - matrix with coefficients
 * @param[in] pfVect - vector with free members
 * @param[out] pfSolution - vector with system solution
 * @param[out] pfMatr becames trianglular after function call
 * @param[out] pfVect changes after function call
 * @author Henry Guennadi Levkin
 */
int Calibration::LinearEquationsSolving(int nDim, double* pfMatr, double* pfVect, double* pfSolution)
{
    double fMaxElem;
    double fAcc;

    int i , j, k, m;

    for(k=0; k<(nDim-1); k++) // base row of matrix
    {
        // search of line with max element
        fMaxElem = fabs( pfMatr[k*nDim + k] );
        m = k;
        for(i=k+1; i<nDim; i++)
        {
            if(fMaxElem < fabs(pfMatr[i*nDim + k]) )
            {
                fMaxElem = pfMatr[i*nDim + k];
                m = i;
            }
        }

        // permutation of base line (index k) and max element line(index m)
        if(m != k)
        {
            for(i=k; i<nDim; i++)
            {
                fAcc               = pfMatr[k*nDim + i];
                pfMatr[k*nDim + i] = pfMatr[m*nDim + i];
                pfMatr[m*nDim + i] = fAcc;
            }
            fAcc = pfVect[k];
            pfVect[k] = pfVect[m];
            pfVect[m] = fAcc;
        }

        if( pfMatr[k*nDim + k] == 0.) return 0; // needs improvement !!!

        // triangulation of matrix with coefficients
        for(j=(k+1); j<nDim; j++) // current row of matrix
        {
            fAcc = - pfMatr[j*nDim + k] / pfMatr[k*nDim + k];
            for(i=k; i<nDim; i++)
            {
                pfMatr[j*nDim + i] = pfMatr[j*nDim + i] + fAcc*pfMatr[k*nDim + i];
            }
            pfVect[j] = pfVect[j] + fAcc*pfVect[k]; // free member recalculation
        }
    }

    for(k=(nDim-1); k>=0; k--)
    {
        pfSolution[k] = pfVect[k];
        for(i=(k+1); i<nDim; i++)
        {
            pfSolution[k] -= (pfMatr[k*nDim + i]*pfSolution[i]);
        }
        pfSolution[k] = pfSolution[k] / pfMatr[k*nDim + k];
    }

    return 1;
}

/**
 * @name SixPointInConstFieldCal
 * @brief Compute the scale and bias assuming the data comes from six orientations
 *        in a constant field
 *
 *   x, y, z are vectors of six measurements
 *
 * Computes sensitivity and offset such that:
 *
 * c = S * A + b
 *
 * where c is the measurement, S is the sensitivity, b is the bias offset, and
 * A is the field being measured  expressed as a ratio of the measured value
 * to the field strength. aka a direction cosine.
 *
 * A is what we really want and it is computed using the equation:
 *
 * A = (c - b)/S
 *
 */
int Calibration::SixPointInConstFieldCal( double ConstMag, double x[6], double y[6], double z[6], double S[3], double b[3] )
{
    int i;
    double A[5][5];
    double f[5], c[5];
    double xp, yp, zp, Sx;

    // Fill in matrix A -
    // write six difference-in-magnitude equations of the form
    // Sx^2(x2^2-x1^2) + 2*Sx*bx*(x2-x1) + Sy^2(y2^2-y1^2) + 2*Sy*by*(y2-y1) + Sz^2(z2^2-z1^2) + 2*Sz*bz*(z2-z1) = 0
    // or in other words
    // 2*Sx*bx*(x2-x1)/Sx^2  + Sy^2(y2^2-y1^2)/Sx^2  + 2*Sy*by*(y2-y1)/Sx^2  + Sz^2(z2^2-z1^2)/Sx^2  + 2*Sz*bz*(z2-z1)/Sx^2  = (x1^2-x2^2)
    for (i=0;i<5;i++){
        A[i][0] = 2.0 * (x[i+1] - x[i]);
        A[i][1] = y[i+1]*y[i+1] - y[i]*y[i];
        A[i][2] = 2.0 * (y[i+1] - y[i]);
        A[i][3] = z[i+1]*z[i+1] - z[i]*z[i];
        A[i][4] = 2.0 * (z[i+1] - z[i]);
        f[i]    = x[i]*x[i] - x[i+1]*x[i+1];
    }

    // solve for c0=bx/Sx, c1=Sy^2/Sx^2; c2=Sy*by/Sx^2, c3=Sz^2/Sx^2, c4=Sz*bz/Sx^2
    if (  !LinearEquationsSolving( 5, (double *)A, f, c) ) return 0;

    // use one magnitude equation and c's to find Sx - doesn't matter which - all give the same answer
    xp = x[0]; yp = y[0]; zp = z[0];
    Sx = sqrt(ConstMag*ConstMag / (xp*xp + 2*c[0]*xp + c[0]*c[0] + c[1]*yp*yp + 2*c[2]*yp + c[2]*c[2]/c[1] + c[3]*zp*zp + 2*c[4]*zp + c[4]*c[4]/c[3]));

    S[0] = Sx;
    b[0] = Sx*c[0];
    S[1] = sqrt(c[1]*Sx*Sx);
    b[1] = c[2]*Sx*Sx/S[1];
    S[2] = sqrt(c[3]*Sx*Sx);
    b[2] = c[4]*Sx*Sx/S[2];

    return 1;
}
